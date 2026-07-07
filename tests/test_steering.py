import asyncio
import socket

import pytest
import uvicorn
import websockets

from auratrack.interceptor import LatentBeamInterceptor
from auratrack.lifecycle_logger import ContextGraphLifecycleLogger
from auratrack.memory import EpigeneticMemoryController
from auratrack.router import ConfluenceRouter
from auratrack.transaction_log import TokenTransactionLog, normalize_splice_token


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_rewrite_history_truncates_and_replaces_selected_token():
    log = TokenTransactionLog()
    log.append_packet({"token_id": 0, "token": "Confluence ", "run_id": "run-a"})
    log.append_packet({"token_id": 1, "token": "Router ", "run_id": "run-a"})
    log.append_packet({"token_id": 2, "token": "validates ", "run_id": "run-a"})
    log.append_packet({"token_id": 3, "token": "intent ", "run_id": "run-a"})

    rewritten = log.rewrite_at(2, {"token": "skeptical   ", "logprob": -0.3}, "run-b")

    assert rewritten.history == ["Confluence ", "Router ", "skeptical "]
    assert log.text() == "Confluence Router skeptical "


def test_splice_token_normalizes_whitespace_and_semantic_markers():
    assert normalize_splice_token("Confluence ", "  <intent>   ") == "<intent></intent> "
    assert normalize_splice_token("Confluence ", "  branch(   ") == "branch() "
    assert normalize_splice_token("Confluence", "   Router   ") == " Router "


@pytest.mark.anyio
async def test_interceptor_stream_exposes_hooks():
    interceptor = LatentBeamInterceptor(token_delay=0)

    packet = None
    async for packet in interceptor.stream_with_steering_hooks("Build AuraTrack", "constraint context"):
        break

    assert packet is not None
    assert packet["source"] == "stream_with_steering_hooks"
    assert packet["alternatives"]
    assert "hidden_state" in packet
    assert "entropy" in packet["hidden_state"]


@pytest.mark.anyio
async def test_router_runs_parallel_micro_validations():
    router = ConfluenceRouter()

    result = await router.analyze_sub_perceptions("FastAPI websocket with latent beam logprobs")

    assert {"semantic_vibe", "relationship_context", "constraint_bounds", "architecture_pressure"} <= set(result)
    assert result["constraint_bounds"]["confidence"] > 0.7


def test_memory_mesh_evaporates():
    controller = EpigeneticMemoryController()
    mesh = controller.manifest_temporary_mesh("session-a", {"constraint_bounds": {"confidence": 0.9}})

    assert mesh["status"] == "active"
    assert controller.get_mesh("session-a") is not None

    released = controller.evaporate_mesh("session-a")

    assert released["status"] == "evaporated"
    assert released["released_nodes"] > 0
    assert controller.get_mesh("session-a") is None


def test_lifecycle_logger_prints_mesh_lifecycle(capsys):
    logger = ContextGraphLifecycleLogger()
    mesh = {"session_id": "session-a", "nodes": [{"id": "n1"}], "edges": []}

    logger.manifested("session-a", mesh)
    logger.stream_completion_successful("session-a")
    logger.evaporated("session-a", {"status": "evaporated"})

    output = capsys.readouterr().out
    assert "[0ms]" in output
    assert "MANIFESTED Epigenetic Mesh for Session session-a" in output
    assert "STREAM COMPLETION SUCCESSFUL" in output
    assert "EVAPORATED Mesh for Session session-a -> RAM Freed. Compute Drag: 0.000W" in output


@pytest.mark.anyio
async def test_concurrent_websocket_steering_keeps_session_state_isolated():
    from auratrack import server as server_module

    original_delay = server_module.interceptor.token_delay
    server_module.interceptor.token_delay = 0.005
    port = _free_tcp_port()
    config = uvicorn.Config(server_module.app, host="127.0.0.1", port=port, log_level="critical", lifespan="off")
    uvicorn_server = uvicorn.Server(config)
    server_task = asyncio.create_task(uvicorn_server.serve())
    await _wait_for_port(port)

    ready_queue: asyncio.Queue[str] = asyncio.Queue()
    steer_now = asyncio.Event()

    try:
        clients = [
            asyncio.create_task(_steered_client(port, index, ready_queue, steer_now))
            for index in range(5)
        ]
        ready_sessions = [await asyncio.wait_for(ready_queue.get(), timeout=10) for _ in range(5)]
        assert len(set(ready_sessions)) == 5

        steer_now.set()
        results = await asyncio.gather(*clients)
    finally:
        uvicorn_server.should_exit = True
        await asyncio.wait_for(server_task, timeout=10)
        server_module.interceptor.token_delay = original_delay

    all_markers = {result["marker"] for result in results}
    assert len({result["session_id"] for result in results}) == 5
    assert len({result["run_id"] for result in results}) == 5

    for result in results:
        own_marker = result["marker"]
        foreign_markers = all_markers - {own_marker}
        assert result["history"] == [own_marker]
        assert result["text"] == own_marker
        assert result["resumed_token_id"] == 1
        assert result["cancelled_run_id"] != result["run_id"]
        assert not any(marker in result["text"] for marker in foreign_markers)


async def _steered_client(
    port: int,
    index: int,
    ready_queue: asyncio.Queue[str],
    steer_now: asyncio.Event,
) -> dict[str, object]:
    session_id = f"load-session-{index}"
    marker = f"session_{index}_marker "
    uri = f"ws://127.0.0.1:{port}/ws/auratrack/{session_id}"

    async with websockets.connect(uri) as websocket:
        await websocket.recv()
        await websocket.send(
            _json_message(
                {
                    "type": "start",
                    "prompt": f"State-isolation load test prompt for session {index}.",
                }
            )
        )

        first_packet = None
        while first_packet is None:
            message = await _receive_json(websocket)
            if message["type"] == "token":
                first_packet = message["packet"]

        await ready_queue.put(session_id)
        await asyncio.wait_for(steer_now.wait(), timeout=10)
        await websocket.send(
            _json_message(
                {
                    "type": "steer",
                    "token_id": first_packet["token_id"],
                    "alternative": {
                        "token": f"  {marker}   ",
                        "logprob": -0.01 - index,
                        "rank": 1,
                        "rationale": f"unique marker for {session_id}",
                    },
                }
            )
        )

        cancelled_run_id = None
        rewrite = None
        resumed_packet = None
        while resumed_packet is None:
            message = await _receive_json(websocket)
            if message["type"] == "stream_cancelled":
                cancelled_run_id = message["run_id"]
            elif message["type"] == "history_rewritten":
                rewrite = message
            elif rewrite and message["type"] == "token":
                resumed_packet = message["packet"]

        return {
            "session_id": session_id,
            "marker": marker,
            "history": rewrite["history"],
            "text": rewrite["text"],
            "run_id": rewrite["run_id"],
            "cancelled_run_id": cancelled_run_id,
            "resumed_token_id": resumed_packet["token_id"],
        }


async def _receive_json(websocket):
    import json

    return json.loads(await asyncio.wait_for(websocket.recv(), timeout=10))


def _json_message(message: dict[str, object]) -> str:
    import json

    return json.dumps(message)


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


async def _wait_for_port(port: int) -> None:
    for _ in range(100):
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.05)
    raise TimeoutError(f"server did not start on port {port}")
