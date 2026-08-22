# Cogito Console

Cogito Console v0.3 is a local, evidence-aware inference console. It runs a small free model on CPU, displays measured token decisions and runtime behavior, and lets an operator replay a continuation from a selected alternate token.

The governing rule is simple: no unlabeled fake telemetry.

- `REAL` means a value came directly from a model runtime, provider, operator action, or system measurement.
- `DERIVED` means Cogito computed the value from named real inputs and records the formula.
- `SYNTHETIC` means a deterministic demo fixture produced the value.
- `UNAVAILABLE` means the active backend cannot expose the measurement; no estimate is substituted.

## v0.3 measured inference

The reference backend uses `HuggingFaceTB/SmolLM2-360M-Instruct` in CPU `float32` mode. Model weights stay in the user's Hugging Face cache and are never added to this repository.

For every emitted token, the backend retains:

| Displayed value | Status | Source or formula |
| --- | --- | --- |
| Token ID and selected-token logprob | `REAL` | Reference tokenizer ID and `log_softmax` of the model's unfiltered last-position logits |
| Top alternatives and their logprobs | `REAL` | Top-k entries from the same unfiltered distribution |
| Token probability | `DERIVED` | `exp(selected_token_logprob)` |
| Full-vocabulary entropy | `DERIVED` | `-sum(p_i * log2(p_i))` |
| Top-1/top-2 margin | `DERIVED` | `probability(top_1) - probability(top_2)` |
| Time to first token and token latency | `REAL` | Local monotonic runtime clock; a cold TTFT includes model loading |
| RAM and CPU | `REAL` | Local `psutil` process and host samples |
| Per-layer hidden-state magnitude and change | `DERIVED` | L2 norm of each real last-position hidden vector and its difference from the previous layer |

Full logits and hidden tensors are discarded after compact summaries are emitted. Attention values, normalized KV-cache byte counts, MoE routing, and activation steering are explicitly unavailable in v0.3.

## Run locally

Python 3.11 or newer is required. On Windows, an isolated environment at a short path avoids long-path problems while installing PyTorch:

```powershell
py -3 -m venv C:\Users\$env:USERNAME\.venvs\cogito-console
& C:\Users\$env:USERNAME\.venvs\cogito-console\Scripts\python.exe -m pip install -U pip
& C:\Users\$env:USERNAME\.venvs\cogito-console\Scripts\python.exe -m pip install -e ".[dev,local]"
& C:\Users\$env:USERNAME\.venvs\cogito-console\Scripts\python.exe -m uvicorn cogito_console.server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Cogito selects the local Transformers backend in the interface when its dependencies are installed. The first run downloads and loads the registered model; later runs reuse the local cache.

The server generates a temporary access token at startup and prints it in the local terminal. Enter that token on the sign-in page. The token is exchanged for an HttpOnly, SameSite=Strict session cookie; it is never placed in the URL. Set `COGITO_ACCESS_TOKEN` to a stable URL-safe value of at least 32 characters when a fixed token is required, including any multi-worker deployment.

Useful environment settings:

```powershell
$env:COGITO_PROVIDER = "transformers" # demo | transformers | openai
$env:COGITO_LOCAL_MODEL = "smollm2-360m-instruct"
$env:COGITO_MAX_INPUT_TOKENS = "512"
$env:COGITO_MAX_NEW_TOKENS = "32"
$env:COGITO_PERSIST = "true"
$env:COGITO_DB_PATH = ".\data\cogito.sqlite"
$env:COGITO_ACCESS_TOKEN = "replace-with-at-least-32-url-safe-characters"
$env:COGITO_TRUSTED_HOSTS = "127.0.0.1,localhost"
```

Generation settings chosen in the interface are captured per run. Model revision, tokenizer revision, dtype, quantization, seed, prompt-token count, and a reproducibility fingerprint are included in token evidence.

## Local access security

Cogito rejects untrusted `Host` headers before routing requests, requires authentication for static assets and every `/api` endpoint, and requires both authentication and an exact same-origin `Origin` header before accepting a WebSocket. `/health`, the local sign-in page, and its token-exchange endpoint are the only unauthenticated HTTP surfaces; the token exchange itself requires an exact same-origin request.

- Keep the service bound to `127.0.0.1` unless a separately secured reverse proxy and explicit trusted hostname are configured.
- Add only exact hostnames or IPv4 addresses to `COGITO_TRUSTED_HOSTS`; wildcards and ports are rejected.
- Set `COGITO_SECURE_COOKIE=true` when the trusted deployment uses HTTPS.
- Non-browser API clients may send `Authorization: Bearer <token>`. WebSocket clients must also send an `Origin` whose host and port exactly match the request authority.

## Backends and steering

- `transformers` performs manual autoregressive decoding and exposes real logits, alternatives, compact hidden-state summaries, and system samples. Selecting an alternate token re-encodes the modified prefix and recomputes the suffix; the ledger records it as `branch_replay`.
- `demo` is a deterministic interface fixture. Its generated values remain `SYNTHETIC`, and its path change is labeled `narrative_ui_splice` rather than true resampling.
- `openai` is optional. It fails visibly without `OPENAI_API_KEY` and never claims access to hidden states or raw logits.

Every backend publishes its capability matrix before generation. Conditional or unsupported measurements are persisted as `measurement.unavailable` events with the backend's stated limitation.

## Audit and export

- `GET /api/providers`
- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `GET /api/sessions/{session_id}/events`
- `GET /api/sessions/{session_id}/metrics`
- `GET /api/sessions/{session_id}/telemetry`
- `GET /api/sessions/{session_id}/export.jsonl`
- `GET /api/evidence/{evidence_id}`

Telemetry uses schema version `1.0`. SQLite event payloads are bounded to 64 KiB by default and JSONL exports to 10,000 events by default; both limits are configurable.

To preview or remove only repository-local databases, traces, and generated artifacts:

```powershell
cogito-purge --dry-run
cogito-purge
```

The command prints every removed path, its byte count, and a total. It does not touch the global model cache.

## Verify

```powershell
python -m pytest -q
```

The suite covers formulas and status propagation, the reference-provider telemetry fixture, capability completeness, explicit unavailable signals, SQLite/JSONL persistence, bounded purge behavior, graph provenance, concurrent WebSocket sessions, and steering cancellation.

The recorded Astræos experiment and machine-specific observations live in [`docs/experiments/v0.3-smollm2-measured-inference.md`](docs/experiments/v0.3-smollm2-measured-inference.md). Architecture decisions and later backend work are tracked under [`docs/`](docs/).
