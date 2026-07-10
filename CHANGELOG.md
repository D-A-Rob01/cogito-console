# Changelog

## v0.2.0 - Real Instrumentation Foundation

- Established Cogito Console as the product name and `cogito_console` as the Python package.
- Added a typed schema layer for evidence records, metrics, token events, alternatives, router validations, rewrites, graph nodes, and graph edges.
- Added `MetricEngine` with documented formulas and conservative synthetic status propagation.
- Added provider adapter boundary with `LocalDemoAdapter` and an `OpenAIAdapter` scaffold that fails gracefully without `OPENAI_API_KEY`.
- Preserved deterministic demo mode and legacy websocket packet fields.
- Added SQLite persistence for sessions, runs, token events, alternatives, rewrites, graph nodes, graph edges, evidence records, and metric values.
- Added REST audit endpoints for sessions, events, graph data, metrics, providers, and evidence lookup.
- Rebuilt graph semantics around prompt checks, provider output, accepted tokens, alternatives, rewrites, and temporary session context.
- Added UI evidence badges, Expert Mode, raw data expansion, branch comparison, provider status, and plain-language labels.
- Added tests for metric formulas, evidence/status propagation, provider fallback, persistence, graph construction, rewrite/session isolation, and concurrent websocket steering.

### Caveats

- Demo logprobs, alternatives, signal-panel values, router confidence, and context mesh weights are synthetic.
- OpenAI adapter support depends on provider/model logprob availability and the optional official SDK.
- Hidden-state access is not claimed for OpenAI mode.
- The Python import package is `cogito_console`.
