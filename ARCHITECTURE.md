# Cogito Console Architecture

Cogito Console is built around a single invariant: every visible metric must carry provenance.

## Event Pipeline

1. `ConfluenceRouter` runs local prompt checks.
2. Router outputs are converted into `RouterValidationEvent` objects with synthetic confidence metrics.
3. `EpigeneticMemoryController` manifests a temporary session context graph.
4. A `ProviderAdapter` streams token events.
5. `MetricEngine` computes real, derived, or synthetic metrics.
6. `TokenTransactionLog` appends accepted tokens.
7. WebSocket packets preserve legacy fields while adding typed `token_event`, `metrics`, `evidence`, and graph fragments.
8. `PersistenceStore` writes the session to SQLite.
9. The frontend renders the live graph and inspector from the packet evidence.

## Provider Adapters

`ProviderAdapter` defines the boundary:

```python
async def stream(prompt, system_context, history, *, session_id, run_id):
    ...
```

Adapters emit `ProviderStreamPacket`, which contains a typed `TokenEvent`, evidence records, metric values, and optional synthetic signal metrics.

Implemented adapters:

- `LocalDemoAdapter`: wraps the deterministic `LatentBeamInterceptor`. Demo logprobs, alternatives, and probe signals are labeled `synthetic`.
- `OpenAIAdapter`: reads `OPENAI_API_KEY` and `COGITO_MODEL`. It fails gracefully without a key. It uses provider logprobs/top alternatives only when returned by the API and does not claim hidden-state access.

## Evidence Model

`EvidenceRecord` answers:

- Where did this value come from?
- Which source field produced it?
- Was a formula used?
- What caveat applies?

`MetricValue` carries:

- `status`: `real`, `derived`, or `synthetic`
- `evidence_id`
- plain explanation
- optional expert explanation

The frontend should never render a metric without status and evidence.

## Metric Engine

`MetricEngine` computes:

- `token_probability = exp(logprob)`
- `alternative_margin = selected_logprob - alternative_logprob`
- `entropy_from_top_logprobs`
- `confidence_gap`
- `tokens_per_second`
- `token_latency_ms`
- `rewrite_count`
- `branch_depth`
- `graph_density`
- `session_duration_ms`

Synthetic propagation is conservative: a probability derived from a synthetic logprob remains synthetic.

## Persistence

SQLite is managed by `PersistenceStore`.

Tables:

- `sessions`
- `runs`
- `token_events`
- `alternative_events`
- `rewrite_events`
- `graph_nodes`
- `graph_edges`
- `evidence_records`
- `metric_values`

Default path:

```text
./data/cogito.sqlite
```

Set `COGITO_PERSIST=false` to disable persistence.

## Graph Semantics

Nodes:

- prompt
- router
- provider
- token
- alternative
- rewrite
- context
- system

Edges:

- `routed_by`
- `constrained_by`
- `generated_by`
- `alternative_to`
- `selected_by_operator`
- `caused_rewrite`
- `held_in_context`
- `derived_from`

Every graph element must explain why it exists. Decorative-only nodes are out of bounds.

## Rewrite Flow

When a `type: steer` message arrives:

1. The server cancels the active stream with `StreamCancellationToken`.
2. It flushes the uncommitted buffer.
3. It splices `TokenTransactionLog` at `token_id`.
4. It normalizes trailing whitespace and closes simple semantic markers.
5. It emits `RewriteEvent`.
6. It starts a new run from the rewritten history.
7. The UI collapses the old branch and opens Compare mode.

## Package

The Python package is `cogito_console`. The GitHub repository slug is `cogito-console`.
