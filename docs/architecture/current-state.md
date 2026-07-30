# Cogito Console baseline audit and v0.3 state

Audit date: 2026-07-13
Baseline commit: `18c9b82`
Baseline tests: `12 passed`

## Runtime shape

At the audited baseline, Cogito Console 0.2 was a FastAPI application with a static HTML/CSS/JavaScript client. A websocket session owned a `RuntimeSession`, a provider stream, a transaction log, an in-memory temporary context mesh, graph projections, and persistence calls.

```text
browser
  -> /ws/cogito/{session_id}
  -> server.RuntimeSession
  -> ConfluenceRouter (synthetic prompt checks)
  -> EpigeneticMemoryController (runtime-only context mesh)
  -> ProviderAdapter.stream(...)
  -> ProviderStreamPacket
  -> TokenTransactionLog + graph projection + SQLite
  -> websocket token/history/complete events
```

REST endpoints expose sessions, events, graph rows, metrics, evidence, and a minimal provider list. Static assets are mounted at `/static`.

## Provider flow

`providers.py` contains one abstract adapter and two concrete paths:

- `LocalDemoAdapter`: deterministic token text, alternatives, logprob-like scores, and signal values. All model-like measurements are correctly tagged `SYNTHETIC`; runtime latency and byte length are `REAL`.
- `OpenAIAdapter`: optional Chat Completions stream. It preserves returned logprobs and alternatives as `REAL`, derives probabilities, and fails visibly when credentials or the SDK are unavailable.

`adapter_from_env()` selects a single process-wide provider at import time. There is no runtime provider selection, model registry, capability contract, or real local model adapter yet.

## Measurement and evidence flow

`schemas.py` provides dataclasses for evidence, metrics, token events, alternatives, router checks, rewrites, and graph elements. `metrics.py` contains explicit formulas for probability, margins, entropy, timing, speed, and graph/session summaries.

Strengths:

- metrics carry status and evidence identifiers;
- derived formulas are documented in evidence records;
- demo model signals remain synthetic;
- raw provider payloads are retained for inspection.

Gaps against the v0.3 mission:

- `UNAVAILABLE` is not a metric status;
- evidence does not name the capability that produced or withheld a value;
- measurements do not carry their collection timestamp directly;
- missing model telemetry is represented by absence rather than an observable event;
- there is no versioned normalized telemetry envelope;
- entropy is currently computed from renormalized top-k values, not the complete distribution;
- there is no model/tokenizer revision or generation-configuration record.

## Persistence flow

`PersistenceStore` creates SQLite tables directly on startup for sessions, runs, token and alternative events, rewrites, graph nodes and edges, evidence, and metrics. Packet persistence recursively stores packet evidence and measurements.

Gaps:

- no schema version/migration ledger;
- no normalized telemetry-event table;
- no capability snapshots, layer summaries, resource samples, or unavailable-measurement table;
- no bounded JSONL export or local purge command.

## Steering flow

The browser sends a token index and selected alternative. The server cancels the active stream, flushes its transient buffer, rewrites `TokenTransactionLog`, creates a new run ID, persists a `RewriteEvent`, and restarts the provider with the modified history.

The mechanics are real server operations, but the current model semantics depend on the provider:

- demo mode is a `SYNTHETIC` narrative/UI splice;
- a provider that truly conditions on the rewritten prefix can be labeled branch replay;
- the current rewrite record does not explicitly enumerate invalidated, reused, and recomputed work.

## Browser and graph flow

The existing client is a responsive three-column dark console with prompt controls, a causal graph, token stream, event timeline, inspector, alternatives, and a signal panel. The graph is assembled client-side from legacy packets rather than rendered from the persisted normalized event stream.

The v0.3 redesign keeps the recognizable console and graph but makes token time, mechanics, capability absence, evidence, and replay provenance the primary reading order. The accepted visual specification is stored outside Git in the task visualization folder.

## Initial implementation boundary

The v0.3 implementation will:

1. add capabilities, `UNAVAILABLE`, and a versioned event envelope without breaking legacy packets;
2. add a lazy, bounded Transformers reference provider behind the existing interface;
3. measure full-distribution token statistics, per-layer last-token summaries, runtime timing, RAM, and CPU;
4. expose capability and unavailable records before and during a run;
5. persist/export normalized events and replay provenance;
6. restructure the static client around the measured-inference workflow;
7. retain deterministic demo mode as an unmistakably synthetic test surface.

llama.cpp, small-MoE routing, and Colibri remain follow-on milestones after the first complete measured-inference slice.

## Implemented v0.3 state

The measured-inference slice now adds a per-session provider selection, a complete capability handshake, explicit unavailable measurements, normalized schema-versioned telemetry, bounded SQLite/JSONL persistence, and a local purge command. The lazy Transformers reference adapter manually decodes a registry-selected SmolLM2 checkpoint and emits real token IDs, logprobs, top alternatives, prefill/decode timing, and resource samples. Probability, entropy, top-choice margin, and layer L2 summaries remain explicitly derived from real tensors.

The browser now reads in this order: reproducible run strip, token lane, model mechanics, measured path, capability matrix, intervention ledger, and evidence inspector. A selected alternative is recorded as a true branch replay only for a provider that declares and performs replay; demo mode remains a synthetic narrative splice.

The recorded local experiment is in `docs/experiments/v0.3-smollm2-measured-inference.md`. The accepted visual concept and browser-verification screenshots are retained in the Codex task's visualization artifacts rather than the repository.
