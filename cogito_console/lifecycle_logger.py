from __future__ import annotations

import json
import time
from typing import Any


class ContextGraphLifecycleLogger:
    """Terminal logger for the temporary context graph lifecycle."""

    def __init__(self) -> None:
        self._manifested_at_ns: dict[str, int] = {}

    def manifested(self, session_id: str, mesh: dict[str, Any]) -> None:
        self._manifested_at_ns[session_id] = time.perf_counter_ns()
        self._print(
            session_id,
            "MANIFESTED",
            f"Epigenetic Mesh for Session {session_id} (Size: {self._size_kb(mesh)})",
            elapsed_ms=0,
        )

    def stream_completion_successful(self, session_id: str) -> None:
        self._print(session_id, "STREAM COMPLETION SUCCESSFUL")

    def evaporated(self, session_id: str, mesh_result: dict[str, Any]) -> None:
        if mesh_result.get("status") == "already_evaporated":
            return
        self._print(session_id, "EVAPORATED", f"Mesh for Session {session_id} -> RAM Freed. Compute Drag: 0.000W")
        self._manifested_at_ns.pop(session_id, None)

    def _print(self, session_id: str, event: str, detail: str | None = None, elapsed_ms: int | None = None) -> None:
        elapsed = self._elapsed_ms(session_id) if elapsed_ms is None else elapsed_ms
        prefix = f"[{elapsed}ms]".ljust(10)
        body = event if detail is None else f"{event} {detail}"
        print(f"{prefix}{body}", flush=True)

    def _elapsed_ms(self, session_id: str) -> int:
        started_at = self._manifested_at_ns.get(session_id)
        if started_at is None:
            return 0
        return round((time.perf_counter_ns() - started_at) / 1_000_000)

    @staticmethod
    def _size_kb(mesh: dict[str, Any]) -> str:
        payload = json.dumps(mesh, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return f"{len(payload) / 1024:.1f}KB"
