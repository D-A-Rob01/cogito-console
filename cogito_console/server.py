from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .graph import build_context_graph, build_rewrite_graph, build_router_graph, build_token_graph, router_validation_events
from .interceptor import LatentBeamInterceptor
from .lifecycle_logger import ContextGraphLifecycleLogger
from .memory import EpigeneticMemoryController
from .metrics import MetricEngine
from .persistence import PersistenceStore
from .providers import ProviderStreamPacket, ProviderUnavailableError, OpenAIAdapter, adapter_from_env
from .router import ConfluenceRouter
from .schemas import RewriteEvent, make_evidence
from .transaction_log import TokenTransactionLog


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Cogito Console", version="0.2.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

router_layer = ConfluenceRouter()
interceptor = LatentBeamInterceptor()
memory_controller = EpigeneticMemoryController()
lifecycle_logger = ContextGraphLifecycleLogger()
provider_adapter = adapter_from_env(interceptor)
persistence_store = PersistenceStore.from_env()


@dataclass
class RuntimeSession:
    session_id: str
    prompt: str = ""
    system_context: str = ""
    sub_context: dict[str, Any] = field(default_factory=dict)
    token_log: TokenTransactionLog = field(default_factory=TokenTransactionLog)
    packets: dict[int, dict[str, Any]] = field(default_factory=dict)
    stream_task: asyncio.Task[None] | None = None
    cancellation_token: StreamCancellationToken | None = None
    stream_buffer: list[dict[str, Any]] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_started_at: float = field(default_factory=time.perf_counter)
    run_started_at: float = field(default_factory=time.perf_counter)
    rewrites: int = 0
    router_events: list[dict[str, Any]] = field(default_factory=list)
    graph_nodes: list[dict[str, Any]] = field(default_factory=list)
    graph_edges: list[dict[str, Any]] = field(default_factory=list)

    def begin_stream(self) -> "StreamCancellationToken":
        self.cancellation_token = StreamCancellationToken(self.run_id)
        self.stream_buffer.clear()
        return self.cancellation_token

    def request_cancel(self, reason: str) -> None:
        if self.cancellation_token is not None:
            self.cancellation_token.cancel(reason)

    def flush_stream_buffer(self) -> int:
        flushed = len(self.stream_buffer)
        self.stream_buffer.clear()
        return flushed


class StreamCancellationToken:
    """Per-run cancellation handle used to break the streaming loop before a rewrite."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.reason: str | None = None
        self._event = asyncio.Event()

    def cancel(self, reason: str = "cancelled") -> None:
        self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "Cogito Console", "provider": provider_adapter.mode}


@app.get("/api/sessions")
async def list_sessions() -> list[dict[str, Any]]:
    return persistence_store.list_sessions()


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    session = persistence_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return session


@app.get("/api/sessions/{session_id}/events")
async def get_session_events(session_id: str) -> dict[str, list[dict[str, Any]]]:
    return persistence_store.get_events(session_id)


@app.get("/api/sessions/{session_id}/graph")
async def get_session_graph(session_id: str) -> dict[str, list[dict[str, Any]]]:
    return persistence_store.get_graph(session_id)


@app.get("/api/sessions/{session_id}/metrics")
async def get_session_metrics(session_id: str) -> list[dict[str, Any]]:
    return persistence_store.get_metrics(session_id)


@app.get("/api/providers")
async def get_providers() -> dict[str, Any]:
    openai_adapter = OpenAIAdapter()
    return {
        "active": provider_adapter.describe(),
        "available": [
            provider_adapter.describe(),
            openai_adapter.describe(),
        ],
        "synthetic_policy": "Synthetic demo metrics are emitted only with status='synthetic' and evidence records.",
    }


@app.get("/api/evidence/{evidence_id}")
async def get_evidence(evidence_id: str) -> dict[str, Any]:
    evidence = persistence_store.get_evidence(evidence_id)
    if evidence is None:
        raise HTTPException(status_code=404, detail="evidence not found")
    return evidence


@app.websocket("/ws/cogito/{session_id}")
async def cogito_socket(websocket: WebSocket, session_id: str) -> None:
    await stream_socket(websocket, session_id)


async def stream_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = RuntimeSession(session_id=session_id)
    await websocket.send_json(
        {
            "type": "session_open",
            "session_id": session_id,
            "provider": provider_adapter.describe(),
        }
    )

    try:
        while True:
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)
            message_type = message.get("type")

            if message_type == "start":
                await start_session(websocket, session, message.get("prompt", ""))
            elif message_type == "steer":
                await steer_session(websocket, session, message)
            elif message_type == "evaporate":
                await cancel_stream(session, websocket, notify=True)
                await websocket.send_json({"type": "mesh_evaporated", "mesh": evaporate_session_mesh(session.session_id)})
            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})
            else:
                await websocket.send_json({"type": "error", "message": f"Unknown message type: {message_type}"})
    except WebSocketDisconnect:
        await cancel_stream(session)
        evaporate_session_mesh(session.session_id)


async def start_session(websocket: WebSocket, session: RuntimeSession, prompt: str) -> None:
    await cancel_stream(session)
    evaporate_session_mesh(session.session_id)
    session.prompt = prompt.strip() or "Build Cogito Console and expose steering hooks."
    session.token_log.reset()
    session.packets = {}
    session.stream_buffer.clear()
    session.run_id = uuid.uuid4().hex
    session.session_started_at = time.perf_counter()
    session.run_started_at = time.perf_counter()
    session.rewrites = 0
    session.router_events = []
    session.graph_nodes = []
    session.graph_edges = []
    persistence_store.upsert_session(
        session.session_id,
        prompt=session.prompt,
        provider_name=provider_adapter.name,
        payload={"provider": provider_adapter.describe()},
    )
    persistence_store.upsert_run(
        session.run_id,
        session_id=session.session_id,
        payload={"provider": provider_adapter.describe(), "rewrite_index": session.rewrites},
    )

    await websocket.send_json({"type": "router_started", "session_id": session.session_id})
    session.sub_context = await router_layer.analyze_sub_perceptions(session.prompt)
    router_events, router_evidence = router_validation_events(session.sub_context)
    session.router_events = [event.to_dict() for event in router_events]
    router_nodes, router_edges = build_router_graph(
        session_id=session.session_id,
        prompt=session.prompt,
        router_events=router_events,
    )
    session.graph_nodes.extend(node.to_dict() for node in router_nodes)
    session.graph_edges.extend(edge.to_dict() for edge in router_edges)
    persistence_store.save_evidence_records(session.session_id, router_evidence)
    persistence_store.save_metrics(
        session.session_id,
        "router",
        [event.confidence for event in router_events],
    )
    persistence_store.save_graph(session.session_id, router_nodes, router_edges)
    await websocket.send_json(
        {
            "type": "router_complete",
            "sub_context": session.sub_context,
            "router_events": session.router_events,
            "evidence": [record.to_dict() for record in router_evidence],
            "graph": {
                "nodes": [node.to_dict() for node in router_nodes],
                "edges": [edge.to_dict() for edge in router_edges],
            },
        }
    )

    mesh = memory_controller.manifest_temporary_mesh(session.session_id, session.sub_context)
    lifecycle_logger.manifested(session.session_id, mesh)
    context_nodes, context_edges = build_context_graph(session.session_id, mesh)
    session.graph_nodes.extend(node.to_dict() for node in context_nodes)
    session.graph_edges.extend(edge.to_dict() for edge in context_edges)
    persistence_store.save_graph(session.session_id, context_nodes, context_edges)
    await websocket.send_json(
        {
            "type": "mesh_manifested",
            "mesh": mesh,
            "graph": {
                "nodes": [node.to_dict() for node in context_nodes],
                "edges": [edge.to_dict() for edge in context_edges],
            },
        }
    )

    session.system_context = f"You are guided by these hidden sub-perceptions: {session.sub_context}"
    session.stream_task = asyncio.create_task(stream_run(websocket, session))


async def steer_session(websocket: WebSocket, session: RuntimeSession, message: dict[str, Any]) -> None:
    token_id = int(message["token_id"])
    alternative = message["alternative"]
    old_run_id = session.run_id
    previous_text = session.token_log.text()

    await cancel_stream(session, websocket, notify=True)
    session.run_id = uuid.uuid4().hex
    session.run_started_at = time.perf_counter()
    try:
        rewrite = session.token_log.rewrite_at(token_id, alternative, session.run_id)
    except IndexError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        return

    session.rewrites += 1
    session.packets = session.token_log.packets()
    rewrite_event = RewriteEvent(
        session_id=session.session_id,
        old_run_id=old_run_id,
        new_run_id=session.run_id,
        token_id=token_id,
        chosen_token=rewrite.replacement,
        previous_text=previous_text,
        rewritten_text=rewrite.text,
        operator_action_timestamp=time.time(),
        reason="operator selected alternative token",
        rationale=alternative.get("rationale"),
    )
    rewrite_payload = rewrite_event.to_dict()
    persistence_store.upsert_run(
        session.run_id,
        session_id=session.session_id,
        payload={
            "provider": provider_adapter.describe(),
            "rewrite_index": session.rewrites,
            "rewrite_event": rewrite_payload,
        },
    )
    persistence_store.save_rewrite_event(rewrite_payload)
    rewrite_nodes, rewrite_edges = build_rewrite_graph(rewrite_payload)
    session.graph_nodes.extend(node.to_dict() for node in rewrite_nodes)
    session.graph_edges.extend(edge.to_dict() for edge in rewrite_edges)
    persistence_store.save_graph(session.session_id, rewrite_nodes, rewrite_edges)

    if memory_controller.get_mesh(session.session_id) is None:
        mesh = memory_controller.manifest_temporary_mesh(session.session_id, session.sub_context)
        lifecycle_logger.manifested(session.session_id, mesh)
        context_nodes, context_edges = build_context_graph(session.session_id, mesh)
        session.graph_nodes.extend(node.to_dict() for node in context_nodes)
        session.graph_edges.extend(edge.to_dict() for edge in context_edges)
        persistence_store.save_graph(session.session_id, context_nodes, context_edges)
        await websocket.send_json(
            {
                "type": "mesh_manifested",
                "mesh": mesh,
                "graph": {
                    "nodes": [node.to_dict() for node in context_nodes],
                    "edges": [edge.to_dict() for edge in context_edges],
                },
            }
        )

    await websocket.send_json(
        {
            "type": "history_rewritten",
            "token_id": token_id,
            "replacement": rewrite.replacement,
            "alternative": alternative,
            "history": rewrite.history,
            "text": rewrite.text,
            "run_id": session.run_id,
            "rewrite_event": rewrite_payload,
            "graph": {
                "nodes": [node.to_dict() for node in rewrite_nodes],
                "edges": [edge.to_dict() for edge in rewrite_edges],
            },
        }
    )
    session.stream_task = asyncio.create_task(stream_run(websocket, session))


async def stream_run(websocket: WebSocket, session: RuntimeSession) -> None:
    run_id = session.run_id
    cancellation_token = session.begin_stream()
    packet_source = provider_adapter.stream(
        prompt=session.prompt,
        system_context=session.system_context,
        history=session.token_log.history(),
        session_id=session.session_id,
        run_id=run_id,
    )
    try:
        async for provider_packet in stream_with_cancellation(packet_source, cancellation_token):
            if run_id != session.run_id or cancellation_token.cancelled:
                return
            packet = provider_packet.to_legacy_packet()
            session.stream_buffer.append(packet)
            session.token_log.append_packet(packet)
            packet["committed_history"] = session.token_log.history()
            packet["text"] = session.token_log.text()
            token_nodes, token_edges = build_token_graph(provider_packet.token_event, provider_adapter.name)
            session.graph_nodes.extend(node.to_dict() for node in token_nodes)
            session.graph_edges.extend(edge.to_dict() for edge in token_edges)
            packet["graph"] = {
                "nodes": [node.to_dict() for node in token_nodes],
                "edges": [edge.to_dict() for edge in token_edges],
            }
            session.packets[packet["token_id"]] = packet
            persistence_store.save_token_packet(session.session_id, packet)
            persistence_store.save_graph(session.session_id, token_nodes, token_edges)
            await websocket.send_json({"type": "token", "packet": packet})
            session.stream_buffer.clear()

        if run_id == session.run_id:
            duration_ms = (time.perf_counter() - session.run_started_at) * 1000
            duration_evidence = make_evidence(
                "runtime",
                "Python runtime",
                "time.perf_counter",
                caveat="Measured for this server-side streaming run.",
            )
            speed_evidence = make_evidence(
                "runtime",
                "Python runtime",
                "token_count / run_duration",
                caveat="Measured from server runtime timestamps and accepted token count.",
            )
            rewrite_evidence = make_evidence(
                "operator",
                "WebSocket steer events",
                "session.rewrites",
                caveat="Counts accepted rewrites in this session.",
            )
            run_metrics = [
                MetricEngine.session_duration_ms(duration_ms, duration_evidence.evidence_id),
                MetricEngine.tokens_per_second(len(session.token_log), duration_ms, speed_evidence.evidence_id),
                MetricEngine.rewrite_count(session.rewrites, rewrite_evidence.evidence_id),
            ]
            persistence_store.save_evidence_records(
                session.session_id,
                [duration_evidence, speed_evidence, rewrite_evidence],
            )
            persistence_store.save_metrics(session.session_id, f"run:{run_id}", run_metrics)
            await websocket.send_json(
                {
                    "type": "stream_complete",
                    "run_id": run_id,
                    "text": session.token_log.text(),
                    "metrics": [metric.to_dict() for metric in run_metrics],
                    "evidence": [
                        duration_evidence.to_dict(),
                        speed_evidence.to_dict(),
                        rewrite_evidence.to_dict(),
                    ],
                }
            )
            lifecycle_logger.stream_completion_successful(session.session_id)
            await asyncio.sleep(0.2)
            await websocket.send_json({"type": "mesh_evaporated", "mesh": evaporate_session_mesh(session.session_id)})
    except asyncio.CancelledError:
        raise
    except ProviderUnavailableError as exc:
        await websocket.send_json(
            {
                "type": "provider_unavailable",
                "provider": provider_adapter.describe(),
                "message": str(exc),
            }
        )
    except WebSocketDisconnect:
        session.request_cancel("disconnect")
        return


async def cancel_stream(session: RuntimeSession, websocket: WebSocket | None = None, notify: bool = False) -> None:
    task = session.stream_task
    if task is None or task.done():
        session.stream_task = None
        session.flush_stream_buffer()
        return

    run_id = session.run_id
    session.request_cancel("steer")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        session.stream_task = None

    flushed_packets = session.flush_stream_buffer()
    if notify and websocket is not None:
        await websocket.send_json({"type": "stream_cancelled", "run_id": run_id, "flushed_packets": flushed_packets})


async def stream_with_cancellation(
    packet_source: AsyncIterator[ProviderStreamPacket],
    cancellation_token: StreamCancellationToken,
) -> AsyncIterator[ProviderStreamPacket]:
    try:
        async for packet in packet_source:
            if cancellation_token.cancelled:
                break
            yield packet
            if cancellation_token.cancelled:
                break
    finally:
        aclose = getattr(packet_source, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await aclose()


def evaporate_session_mesh(session_id: str) -> dict[str, Any]:
    result = memory_controller.evaporate_mesh(session_id)
    lifecycle_logger.evaporated(session_id, result)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("cogito_console.server:app", host="127.0.0.1", port=8000, reload=True)
