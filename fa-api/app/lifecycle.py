import gc
import threading
from collections.abc import Callable
from typing import Any

MemoryProbe = Callable[[], int]


class StatusError(RuntimeError):
    pass


class _Operation:
    def __init__(self) -> None:
        self._done = threading.Event()
        self.status_code = 200
        self.body: dict[str, Any] = {}

    def finish(self, status_code: int, body: dict[str, Any]) -> None:
        self.status_code = status_code
        self.body = body
        self._done.set()

    def wait(self) -> tuple[int, dict[str, Any]]:
        self._done.wait()
        return self.status_code, self.body


def default_memory_allocated() -> int:
    try:
        import torch
    except Exception:
        return 0
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_allocated())


def _empty_cache() -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    clear_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear_workspaces is not None:
        clear_workspaces()
    torch.cuda.empty_cache()


class Lifecycle:
    """Process-local model lifecycle. GPU work stays outside this lock."""

    def __init__(self, service: Any, memory: MemoryProbe | None = None) -> None:
        self.service = service
        self._memory = memory or default_memory_allocated
        self._lock = threading.Lock()
        self.state = "unloaded"
        self.residency = "not_resident"
        self.active_requests = 0
        self.last_error: dict[str, str] | None = None
        self._baseline: int | None = None
        self._operation: _Operation | None = None
        self.status_broken = False
        if getattr(service, "loaded", False):
            self.state = "ready"
            self.residency = "resident"
            self._baseline = self._memory()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self.status_broken:
                raise StatusError("lifecycle status is unavailable")
            return self._view()

    def ready(self) -> bool:
        with self._lock:
            return self.state == "ready" and self.residency == "resident"

    def try_accept(self) -> bool:
        with self._lock:
            if self.state != "ready" or self.residency != "resident":
                return False
            self.active_requests += 1
            return True

    def finish_request(self) -> None:
        with self._lock:
            if self.active_requests > 0:
                self.active_requests -= 1

    def start_load(self) -> tuple[str, Any]:
        with self._lock:
            if self.state == "loading" and self._operation is not None:
                return "wait", self._operation
            if self.state == "unloading":
                return "conflict", ("LIFECYCLE_CONFLICT", "unload in progress")
            if self.state == "ready":
                return "done", self._view()
            if self.state == "failed" and self.residency != "not_resident":
                return "conflict", (
                    "LIFECYCLE_CONFLICT",
                    "release residency before load",
                )
            self.state = "loading"
            self._operation = _Operation()
            return "run", self._operation

    def run_load(self, operation: _Operation) -> None:
        baseline = self._memory()
        failure: str | None = None
        try:
            self.service.load()
        except Exception as exc:
            failure = str(exc) or "load failed"
        if failure is not None:
            self._fail_load(operation, baseline, RuntimeError(failure))
            return
        with self._lock:
            self._baseline = baseline
            self.state = "ready"
            self.residency = "resident"
            self.last_error = None
            self.service.loaded = True
            if self._operation is operation:
                self._operation = None
            body = self._view()
        operation.finish(200, body)

    def start_unload(self) -> tuple[str, Any]:
        with self._lock:
            if self.state == "unloading" and self._operation is not None:
                return "wait", self._operation
            if self.active_requests > 0:
                return "conflict", ("BUSY", "runtime has active inference requests")
            if self.state == "loading":
                return "conflict", ("LIFECYCLE_CONFLICT", "load in progress")
            if self.state == "unloaded":
                return "done", self._view()
            if self.state == "failed" and self.residency == "not_resident":
                self.service.release()
                self.service.loaded = False
                self.state = "unloaded"
                self.residency = "not_resident"
                self.last_error = None
                self._baseline = None
                return "done", self._view()
            if self.state == "ready" or (
                self.state == "failed" and self.residency in ("resident", "unknown")
            ):
                self.state = "unloading"
                self._operation = _Operation()
                return "run", self._operation
            return "conflict", ("LIFECYCLE_CONFLICT", "lifecycle conflict")

    def run_unload(self, operation: _Operation) -> None:
        baseline = self._baseline
        self.service.release()
        gc.collect()
        _empty_cache()
        allocated = self._memory()
        released = baseline is not None and allocated <= baseline
        message = "model memory was not released"
        with self._lock:
            if released:
                self.state = "unloaded"
                self.residency = "not_resident"
                self.last_error = None
                self.service.loaded = False
                self._baseline = None
                status_code = 200
                body = self._view()
            else:
                self.state = "failed"
                self.residency = "resident" if self.service.holds_model() else "unknown"
                self.last_error = {"code": "UNLOAD_FAILED", "message": message}
                self.service.loaded = False
                status_code = 500
                body = {"error": {"code": "UNLOAD_FAILED", "message": message}}
            if self._operation is operation:
                self._operation = None
        operation.finish(status_code, body)

    def _fail_load(self, operation: _Operation, baseline: int, exc: BaseException) -> None:
        self.service.release()
        gc.collect()
        _empty_cache()
        allocated = self._memory()
        residency = "not_resident" if allocated <= baseline else "unknown"
        message = str(exc) or "load failed"
        with self._lock:
            self._baseline = baseline
            self.state = "failed"
            self.residency = residency
            self.last_error = {"code": "LOAD_FAILED", "message": message}
            self.service.loaded = False
            if self._operation is operation:
                self._operation = None
        operation.finish(500, {"error": {"code": "LOAD_FAILED", "message": message}})

    def _view(self) -> dict[str, Any]:
        error = None if self.last_error is None else dict(self.last_error)
        return {
            "state": self.state,
            "residency": self.residency,
            "active_requests": self.active_requests,
            "last_error": error,
        }
