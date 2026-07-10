import asyncio
import socket

import pytest
import uvicorn
import websockets

from cogito_console.graph import build_token_graph
from cogito_console.interceptor import LatentBeamInterceptor
from cogito_console.lifecycle_logger import ContextGraphLifecycleLogger
from cogito_console.memory import EpigeneticMemoryController
from cogito_console.metrics import MetricEngine
from cogito_console.persistence import PersistenceStore
from cogito_console.providers import LocalDemoAdapter, OpenAIAdapter, ProviderUnavailableError
from cogito_console.router import ConfluenceRouter
from cogito_console.schemas import MetricValue
from cogito_console.transaction_log import TokenTransactionLog, normalize_splice_token


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
    async for packet in interceptor.stream_with_steering_hooks("Build Cogito Console", "constraint context"):
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


def test_metric_engine_formulas_and_status_propagation():
    real_logprob = MetricValue(
        name="token_logprob",
        value=-0.25,
        unit="logprob",
        status="real",
        evidence_id="ev-real",
        display_label="Logprob",
        plain_explanation="provider value",
    )
    synthetic_logprob = MetricValue(
        name="token_logprob",
        value=-1.0,
        unit="logprob",
        status="synthetic",
        evidence_id="ev-synth",
        display_label="Logprob",
        plain_explanation="demo value",
    )

    probability = MetricEngine.token_probability(real_logprob, "ev-derived")
    synthetic_probability = MetricEngine.token_probability(synthetic_logprob, "ev-synth-derived")
    margin = MetricEngine.alternative_margin(real_logprob, synthetic_logprob, "ev-margin")
    entropy = MetricEngine.entropy_from_top_logprobs([real_logprob, synthetic_logprob], "ev-entropy")

    assert probability.status == "derived"
    assert probability.value == pytest.approx(0.77880078)
    assert synthetic_probability.status == "synthetic"
    assert margin.status == "synthetic"
    assert margin.value == pytest.approx(0.75)
    assert entropy.status == "synthetic"
    assert entropy.value > 0


@pytest.mark.anyio
async def test_local_demo_packets_label_all_synthetic_metrics():
    adapter = LocalDemoAdapter(LatentBeamInterceptor(token_delay=0))
    packet = None
    async for provider_packet in adapter.stream(
        "Build Cogito",
        "constraint context",
        [],
        session_id="audit-session",
        run_id="run-a",
    ):
        packet = provider_packet.to_legacy_packet()
        break

    assert packet is not None
    assert packet["token_event"]["logprob"]["status"] == "synthetic"
    assert packet["token_event"]["probability"]["status"] == "synthetic"
    assert packet["alternatives"]
    assert packet["hidden_state_metrics"]

    for metric in _collect_metric_dicts(packet):
        assert metric["status"] in {"real", "derived", "synthetic"}
        assert metric["evidence_id"]
        assert metric["plain_explanation"]


@pytest.mark.anyio
async def test_openai_adapter_missing_key_fails_gracefully():
    adapter = OpenAIAdapter(api_key="")

    assert adapter.is_available() is False
    assert "missing OPENAI_API_KEY" in adapter.unavailable_reason()
    with pytest.raises(ProviderUnavailableError):
        stream = adapter.stream(
            "hello",
            "system",
            [],
            session_id="missing-key",
            run_id="run-a",
        )
        await stream.__anext__()


def test_sqlite_persistence_roundtrip(tmp_path):
    store = PersistenceStore(tmp_path / "cogito.sqlite")
    packet = {
        "run_id": "run-a",
        "token_id": 0,
        "token": "Cogito ",
        "metrics": {
            "token_logprob": {
                "name": "token_logprob",
                "value": -0.5,
                "unit": "logprob",
                "status": "synthetic",
                "evidence_id": "ev-a",
                "display_label": "Logprob",
                "plain_explanation": "demo value",
            }
        },
        "alternatives": [],
        "evidence": [
            {
                "evidence_id": "ev-a",
                "source_kind": "synthetic",
                "source_name": "test",
                "source_field": "packet.logprob",
                "formula": None,
                "caveat": "demo",
                "created_at": 1.0,
            }
        ],
    }

    store.upsert_session("session-a", prompt="prompt", provider_name="demo")
    store.upsert_run("run-a", session_id="session-a")
    store.save_token_packet("session-a", packet)

    assert store.get_session("session-a")["session_id"] == "session-a"
    assert store.get_events("session-a")["tokens"][0]["payload"]["token"] == "Cogito "
    assert store.get_evidence("ev-a")["payload"]["source_kind"] == "synthetic"
    assert store.get_metrics("session-a")[0]["payload"]["plain_explanation"] == "demo value"


@pytest.mark.anyio
async def test_graph_nodes_and_edges_preserve_metric_status():
    adapter = LocalDemoAdapter(LatentBeamInterceptor(token_delay=0))
    provider_packet = None
    async for item in adapter.stream(
        "Build graph",
        "constraint context",
        [],
        session_id="graph-session",
        run_id="run-a",
    ):
        provider_packet = item
        break

    nodes, edges = build_token_graph(provider_packet.token_event, adapter.name)

    token_nodes = [node for node in nodes if node.node_type == "token"]
    alternative_nodes = [node for node in nodes if node.node_type == "alternative"]
    assert token_nodes[0].status == "synthetic"
    assert alternative_nodes
    assert all(node.evidence_ids for node in alternative_nodes)
    assert any(edge.edge_type == "alternative_to" for edge in edges)


@pytest.mark.anyio
async def test_concurrent_websocket_steering_keeps_session_state_isolated():
    from cogito_console import server as server_module

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
    uri = f"ws://127.0.0.1:{port}/ws/cogito/{session_id}"

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


def _collect_metric_dicts(value):
    metrics = []
    if isinstance(value, dict):
        if {"status", "evidence_id", "plain_explanation"} <= set(value):
            metrics.append(value)
        for child in value.values():
            metrics.extend(_collect_metric_dicts(child))
    elif isinstance(value, list):
        for child in value:
            metrics.extend(_collect_metric_dicts(child))
    return metrics
