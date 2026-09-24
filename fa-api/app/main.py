import asyncio
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.aligner import ForcedAlignerService
from app.config import Settings, load_settings
from app.schemas import AlignmentError, ClientError, parse_payload

logger = logging.getLogger("fa")


@asynccontextmanager
async def lifespan(app: FastAPI):
    service: ForcedAlignerService = app.state.service
    if app.state.load_on_startup and not service.loaded:
        await asyncio.to_thread(service.load)
    yield


def create_app(
    settings: Settings | None = None,
    service: ForcedAlignerService | None = None,
    *,
    load_on_startup: bool = True,
) -> FastAPI:
    if service is None:
        service = ForcedAlignerService(settings or load_settings())
    app = FastAPI(lifespan=lifespan)
    app.state.service = service
    app.state.inference_lock = asyncio.Lock()
    app.state.load_on_startup = load_on_startup

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": _format_http_validation(exc)})

    @app.get("/health")
    async def health() -> JSONResponse:
        if not app.state.service.loaded:
            return JSONResponse(
                status_code=503,
                content={"status": "loading", "model_loaded": False},
            )
        return JSONResponse(content={"status": "ok", "model_loaded": True})

    @app.post("/align")
    async def align(
        file: UploadFile = File(...),
        payload: str = Form(...),
    ) -> JSONResponse:
        if not app.state.service.loaded:
            return JSONResponse(
                status_code=503,
                content={"status": "loading", "model_loaded": False},
            )

        directory = tempfile.mkdtemp(prefix="fa-align-")
        handed_off = False
        try:
            try:
                path = await _save_upload(file, directory)
                parsed = decode_payload(payload)
            except ClientError as exc:
                logger.warning("rejected alignment request: %s", exc.detail)
                return JSONResponse(status_code=400, content={"detail": exc.detail})
            except Exception:
                logger.exception("alignment failed")
                return JSONResponse(status_code=500, content={"detail": "alignment failed"})

            handed_off = True
            try:
                result = await complete_alignment(
                    app.state.inference_lock,
                    directory,
                    lambda: app.state.service.align_chunks(path, parsed),
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
            if not handed_off:
                shutil.rmtree(directory, ignore_errors=True)

    return app


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
) -> Any:
    try:
        return await execute_locked(lock, work)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


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
