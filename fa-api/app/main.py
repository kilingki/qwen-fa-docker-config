import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.aligner import ForcedAlignerService
from app.config import Settings, load_settings
from app.lifecycle import Lifecycle, StatusError
from app.schemas import AlignmentError, ClientError, parse_payload

logger = logging.getLogger("fa")

_UNAVAILABLE = {"status": "loading", "model_loaded": False}


def create_app(
    settings: Settings | None = None,
    service: ForcedAlignerService | None = None,
    *,
    load_on_startup: bool = False,
    memory: Callable[[], int] | None = None,
) -> FastAPI:
    del load_on_startup
    if service is None:
        service = ForcedAlignerService(settings or load_settings())
    app = FastAPI()
    app.state.service = service
    app.state.lifecycle = Lifecycle(service, memory=memory)
    app.state.inference_lock = asyncio.Lock()

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": _format_http_validation(exc)})

    @app.get("/health")
    async def health() -> JSONResponse:
        if not app.state.lifecycle.ready():
            return JSONResponse(status_code=503, content=_UNAVAILABLE)
        return JSONResponse(content={"status": "ok", "model_loaded": True})

    @app.get("/control/status")
    async def control_status() -> JSONResponse:
        try:
            return JSONResponse(content=app.state.lifecycle.snapshot())
        except StatusError as exc:
            return _control_error(500, "STATUS_FAILED", str(exc))

    @app.post("/control/load")
    async def control_load(request: Request) -> JSONResponse:
        rejected = await _reject_control_body(request)
        if rejected is not None:
            return rejected
        return await _run_lifecycle(app.state.lifecycle.start_load, app.state.lifecycle.run_load)

    @app.post("/control/unload")
    async def control_unload(request: Request) -> JSONResponse:
        rejected = await _reject_control_body(request)
        if rejected is not None:
            return rejected
        return await _run_lifecycle(
            app.state.lifecycle.start_unload,
            app.state.lifecycle.run_unload,
        )

    @app.post("/align")
    async def align(
        file: UploadFile = File(...),
        payload: str = Form(...),
    ) -> JSONResponse:
        if not app.state.lifecycle.ready():
            return JSONResponse(status_code=503, content=_UNAVAILABLE)

        directory = tempfile.mkdtemp(prefix="fa-align-")
        accepted = False
        try:
            try:
                path = await _save_upload(file, directory)
                parsed = decode_payload(payload)
                await asyncio.to_thread(app.state.service.validate_request, path, parsed)
            except ClientError as exc:
                logger.warning("rejected alignment request: %s", exc.detail)
                return JSONResponse(status_code=400, content={"detail": exc.detail})
            except Exception:
                logger.exception("alignment failed")
                return JSONResponse(status_code=500, content={"detail": "alignment failed"})

            current = asyncio.current_task()
            if current is not None and getattr(current, "cancelling", lambda: 0)():
                return JSONResponse(status_code=503, content=_UNAVAILABLE)
            if not app.state.lifecycle.try_accept():
                return JSONResponse(status_code=503, content=_UNAVAILABLE)
            accepted = True
            try:
                result = await complete_alignment(
                    app.state.inference_lock,
                    directory,
                    lambda: app.state.service.align_chunks(path, parsed),
                    after_cleanup=app.state.lifecycle.finish_request,
                )
            except ClientError as exc:
                logger.warning("rejected alignment request: %s", exc.detail)
                return JSONResponse(status_code=400, content={"detail": exc.detail})
            except AlignmentError as exc:
                logger.exception("alignment failed")
                return JSONResponse(status_code=500, content={"detail": exc.detail})
            except Exception:
                logger.exception("alignment failed")
                return JSONResponse(status_code=500, content={"detail": "alignment failed"})
            return JSONResponse(content=result)
        finally:
            if not accepted:
                shutil.rmtree(directory, ignore_errors=True)

    return app


def _control_error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


async def _reject_control_body(request: Request) -> JSONResponse | None:
    raw = await request.body()
    if not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _control_error(400, "BAD_REQUEST", "request body must be empty or {}")
    if data != {}:
        return _control_error(400, "BAD_REQUEST", "request body must be empty or {}")
    return None


async def _run_lifecycle(start: Callable[[], tuple[str, Any]], run: Callable[[Any], None]) -> JSONResponse:
    kind, payload = start()
    if kind == "done":
        return JSONResponse(content=payload)
    if kind == "conflict":
        code, message = payload
        return _control_error(409, code, message)
    operation = payload
    if kind == "run":
        threading.Thread(target=run, args=(operation,), daemon=False).start()
    status_code, body = await asyncio.to_thread(operation.wait)
    return JSONResponse(status_code=status_code, content=body)


def decode_payload(raw: str) -> Any:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ClientError("payload is not valid JSON") from exc
    return parse_payload(data)


async def complete_alignment(
    lock: asyncio.Lock,
    directory: str,
    work: Callable[[], Any],
    after_cleanup: Callable[[], None] | None = None,
) -> Any:
    try:
        return await execute_locked(lock, work)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        if after_cleanup is not None:
            after_cleanup()


async def execute_locked(lock: asyncio.Lock, work: Callable[[], Any]) -> Any:
    """Run work on one worker thread while holding lock.

    HTTP 취소가 worker thread를 멈추지는 않는다. 취소되어도 thread가 끝난 뒤에만
    락을 풀어, 다음 요청이 같은 모델에 겹치지 않게 한다.
    """
    await lock.acquire()
    task = asyncio.create_task(asyncio.to_thread(work))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await _wait_for_worker(task)
        raise
    finally:
        lock.release()


async def _wait_for_worker(task: asyncio.Task) -> None:
    current = asyncio.current_task()
    while not task.done():
        _clear_cancellation(current)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("alignment worker failed during cancellation", exc_info=exc)


def _clear_cancellation(task: asyncio.Task | None) -> None:
    if task is None or not hasattr(task, "uncancel"):
        return
    cancelling = getattr(task, "cancelling", None)
    if cancelling is None:
        task.uncancel()
        return
    while cancelling():
        task.uncancel()


async def _save_upload(upload: UploadFile, directory: str) -> str:
    path = os.path.join(directory, "input.wav")
    try:
        with open(path, "wb") as handle:
            while True:
                block = await upload.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
    finally:
        await upload.close()
    return path


def _format_http_validation(exc: RequestValidationError) -> str:
    messages: list[str] = []
    for err in exc.errors():
        loc = [str(part) for part in err.get("loc", ()) if part != "body"]
        place = ".".join(loc) or "request"
        messages.append(f"{place}: {err.get('msg', 'invalid value')}")
    return "; ".join(messages) if messages else "invalid request"


app = create_app()
