# AuraTrack

AuraTrack is an introspective orchestration and synaptic steering framework. It turns a single prompt into a multi-layer, steerable inference stream:

1. **Confluence Router** runs parallel micro-validations before generation.
2. **Latent Beam Interceptor** streams accepted tokens, top-logprob alternatives, and hidden-state probes.
3. **Epigenetic Memory Controller** manifests a temporary context graph and evaporates it after the task completes.

The included frontend connects to FastAPI over a websocket, renders the live graph, and lets you click an alternative logprob token to rewrite model history mid-stream. Accepted tokens render as a glowing central highway; alternatives branch laterally as probability-weighted ghost pathways.

## Run

```powershell
cd outputs\auratrack
python -m uvicorn auratrack.server:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## Steering Flow

The browser sends:

```json
{
  "type": "steer",
  "token_id": 7,
  "alternative": { "token": "skeptical ", "logprob": -1.12 }
}
```

The server triggers a per-run cancellation token, flushes any uncommitted stream buffer, splices a structured token transaction log at `token_id`, inserts the selected alternative token, emits `history_rewritten`, and resumes `stream_with_steering_hooks` from that exact junction.

The splice layer normalizes trailing whitespace and closes simple open semantic markers such as `<intent>` or `branch(` before generation resumes.

## Context Graph Lifecycle

AuraTrack prints the exact lifecycle of each temporary context mesh:

```text
[0ms]     MANIFESTED Epigenetic Mesh for Session lifecycle-demo (Size: 2.4KB)
[14815ms] STREAM COMPLETION SUCCESSFUL
[15030ms] EVAPORATED Mesh for Session lifecycle-demo -> RAM Freed. Compute Drag: 0.000W
```

## Test

```powershell
python -m pytest
```

The test suite includes a five-client websocket load test that makes concurrent steering interventions and validates strict session isolation across session IDs.

This demo uses a deterministic local engine so it runs without an API key. The adapter boundary is in `LatentBeamInterceptor`: a provider that exposes real token logprobs or hidden states can replace the local packet generator while keeping the websocket protocol and steering semantics intact.
