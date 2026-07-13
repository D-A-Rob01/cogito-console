# Codex Mission: Turn Cogito Console into a Real Local Inference Observatory

## Mandate

Advance Cogito Console from an auditable steering prototype into a portfolio-worthy local AI observability laboratory that reports **real measured model and runtime values wherever technically accessible**, derives additional metrics only through documented formulas, and labels simulations honestly.

The project must remain useful at every stage. Do not postpone all value until a giant model or exotic backend works.

The architectural principle is:

> **Cogito Console is the observatory. Models and inference engines are replaceable specimens.**

Colibrì is an optional advanced backend and stress-test target. It is not the foundation, the only supported engine, or a prerequisite for the first credible release.

---

## Standing authorization

Within the `D-A-Rob01/cogito-console` repository and its project-scoped development environment, Codex is authorized to proceed without repeatedly requesting routine approval for:

- inspecting, creating, editing, moving, and deleting project files;
- refactoring the application architecture while preserving tested behavior;
- installing free/open-source development dependencies in isolated environments;
- adding provider adapters, local runtimes, tests, fixtures, migrations, scripts, and documentation;
- running tests, linters, benchmarks, local servers, and browser verification;
- creating branches, commits, and draft pull requests;
- cloning or forking open-source dependencies for inspection or project-scoped experimental patches;
- downloading openly licensed small model artifacts needed for testing, provided a single model download does not exceed 10 GB and total new model storage remains below 20 GB;
- making reasonable technical decisions and documenting them in architecture decision records;
- continuing all unblocked work when one optional backend is unavailable.

Do not interrupt the workflow for ordinary implementation choices. Choose the safest reversible option, document it, test it, and continue.

This standing authorization does **not** authorize Codex to:

- spend money or create paid cloud resources;
- expose, print, commit, or transmit secrets, API keys, private conversations, or personal files;
- weaken system security, disable antivirus, open public network ports, or change account permissions;
- delete repositories, cloud resources, user documents, or external data;
- push changes to third-party upstream repositories without explicit approval;
- download GLM-5.2, the approximately 370 GB Colibrì checkpoint, or any model larger than 10 GB during the initial implementation;
- make invasive driver, firmware, partition, registry, or operating-system changes.

When a blocked action falls outside authorization, record it in `docs/BLOCKERS.md`, implement every non-blocked component, and leave an exact command or procedure for the user rather than stopping the entire mission.

---

## Known target machine

Design and benchmark the first local version for **Astræos**:

- Windows 11 x64
- Intel Core Ultra 7 256V
- 16 GB RAM
- Intel Arc 140V integrated GPU
- approximately 581 GB free storage at the time of planning

Assume CPU-first portability. Intel GPU/SYCL/OpenVINO acceleration may be explored behind capability detection, but the application must still function without it.

Do not assume NVIDIA CUDA.

---

## Non-negotiable epistemic contract

Preserve and strengthen the existing rule:

- `REAL`: directly emitted by a model, inference engine, provider, operating system, or measured runtime event.
- `DERIVED`: calculated from real inputs using a visible documented formula.
- `SYNTHETIC`: generated solely for demos, UI testing, or unavailable-signal simulation.
- `UNAVAILABLE`: requested telemetry the active backend cannot expose.

Add `UNAVAILABLE` as a first-class status rather than representing absent data as `null` without explanation.

Every displayed value must include:

1. status;
2. source/provider/runtime;
3. evidence identifier;
4. collection timestamp;
5. formula, when derived;
6. caveat or known limitation;
7. backend capability that produced or withheld it.

Missing access is itself an observable finding. Never infer hidden state, attention, routing, confidence, or causation from visual resemblance.

---

## Product thesis

Cogito Console should let an operator examine and intervene in the **observable mechanics of inference** without pretending that observable mechanics equal consciousness or complete reasoning.

The system should eventually compare several layers of evidence:

1. **Generation surface** — emitted tokens, token IDs, logits/logprobs, top alternatives, entropy, sampling configuration.
2. **Internal activation summaries** — hidden-state norms and changes, layer-wise similarities, attention summaries where exposed.
3. **Mixture-of-Experts routing** — selected experts, routing weights, expert reuse, load balance, cache hits, and disk/RAM placement where exposed.
4. **Memory and context** — KV-cache allocation, prefix reuse, context growth, evictions, persistence, and slot ownership.
5. **Runtime mechanics** — prefill/decode latency, time to first token, inter-token latency, tokens per second, memory, CPU/GPU utilization, disk reads, queue wait, cancellation, and restart cost.
6. **Intervention provenance** — what the operator changed, which path was invalidated, what was recomputed, and how the continuation diverged.
7. **Epistemic tests** — calibration, consistency, abstention, answer correctness, and the difference between model-reported confidence and probability-derived uncertainty.

The graph must represent evidence and causal execution order, not decorative neurological metaphor.

---

## Required architecture

Refactor toward a backend-neutral capability architecture.

### 1. Capability handshake

Every provider must declare a machine-readable capability record, including at minimum:

- streaming;
- token IDs;
- selected-token logprobs;
- top-k alternative tokens;
- raw logits;
- hidden states;
- attentions;
- MoE router logits;
- selected experts and routing weights;
- KV-cache metrics;
- request queue metrics;
- system resource telemetry;
- deterministic seed support;
- grammar/structured-output support;
- steering method supported;
- known granularity and limitations.

Expose this through `/api/providers` and render it in the UI before a run begins.

### 2. Normalized telemetry envelope

Create a versioned schema that can accept token-, layer-, expert-, request-, and system-level events without forcing every backend to emit every field.

Recommended event families:

- `run.started`
- `provider.capabilities`
- `prompt.encoded`
- `prefill.started` / `prefill.completed`
- `token.sampled`
- `token.committed`
- `layer.summary`
- `attention.summary`
- `router.decision`
- `expert.loaded`
- `expert.cache_hit`
- `kv.updated`
- `resource.sample`
- `steer.requested`
- `run.cancelled`
- `run.resumed`
- `run.completed`
- `measurement.unavailable`

Do not store enormous full tensors by default. Store compact summaries plus optional artifact references. Add explicit retention and sampling policies.

### 3. Provider separation

Implement or preserve these provider classes behind one interface:

#### A. Demo provider

Retain deterministic synthetic mode for demonstrations and UI tests. Make its synthetic status visually impossible to miss.

#### B. OpenAI provider

Use only values actually returned by the API. Retain graceful degradation and provider-evidence capture.

#### C. Reference instrumented local provider

This is the first scientifically useful local backend.

Use a small openly licensed transformer that runs within Astræos’s 16 GB RAM budget. Prefer a 1B–3B model for the first implementation. The exact model is replaceable and must be selected through a registry rather than hard-coded into the product identity.

Use a runtime that allows direct access to:

- token IDs;
- raw output logits;
- selected-token logprob;
- top-k alternatives;
- entropy and margin derived from logits;
- per-layer hidden-state summaries;
- attention summaries when feasible;
- exact generation and timing configuration.

A Python/Transformers reference implementation is acceptable even if it is slower, because its purpose is instrumentation fidelity and testability. Use inference mode, bounded context, tensor-summary policies, and explicit memory guards.

#### D. llama.cpp provider

Add a practical local GGUF provider through `llama-server` or a thin native binding. Capture real logprobs and runtime telemetry supported by the server. Treat unavailable hidden-state or routing data honestly.

Use this for responsive local interaction and hardware-friendly model experiments.

#### E. Small MoE provider

Add a research backend for a manageable open Mixture-of-Experts model, preferably **OLMoE-1B-7B-Instruct or an equivalently small open MoE** that can be quantized to fit Astræos.

The goal is real expert-routing instrumentation:

- chosen experts per layer/token;
- router weights/logits;
- expert load distribution;
- expert reuse across adjacent tokens;
- routing entropy;
- cache or load behavior where the runtime exposes it.

Choose the implementation route that produces the most defensible telemetry on the target machine:

1. direct Transformers instrumentation;
2. a project-scoped llama.cpp instrumentation patch;
3. the smaller OLMoE engine present in Colibrì;
4. another open runtime with clearer hooks.

Document the tradeoff and preserve the provider interface so the choice can change later.

#### F. Colibrì provider

Implement the **adapter, capability declaration, fixtures, and integration tests now**, but do not download GLM-5.2 during the initial mission.

Colibrì should remain an external sibling service or separately cloned runtime. Do not vendor hundreds of gigabytes into Cogito Console and never commit model weights.

Expected topology:

```text
Cogito Console
  ├─ UI and experiment ledger
  ├─ provider adapters
  ├─ normalized telemetry/event store
  └─ Colibrì adapter ──HTTP/SSE or telemetry side channel──> Colibrì runtime
                                                         └─ model weights outside Git
```

The first Colibrì adapter may consume its OpenAI-compatible chat endpoint, health endpoint, queue headers, usage data, and any runtime statistics available through stdout/files. Then design a versioned telemetry sidecar or patch proposal for deeper values such as expert routing, cache hits, KV usage, disk reads, and per-token timing.

Colibrì is considered successful when the adapter contract and replay fixtures work without the giant checkpoint. A future hardware gate may enable the real model.

---

## Early display-worthy release

Deliver an early release that is recognizably valuable without Colibrì or a giant model.

### Target: `v0.3 — Measured Inference`

A user should be able to run a small local model and see:

- live generated tokens;
- real selected-token logprob;
- real top-k alternatives;
- real token probability derived from logprob;
- entropy and selected-vs-runner-up margin;
- time to first token and per-token latency;
- model and tokenizer identity, revision, quantization, and generation settings;
- system RAM/CPU measurements;
- at least one real layer-wise hidden-state summary;
- evidence and formulas for every displayed metric;
- an explicit capability panel showing what the backend cannot expose;
- a steering/replay demonstration with provenance that distinguishes true resampling from UI-only path editing.

This release is portfolio-worthy if it is honest, reproducible, visually legible, and accompanied by a recorded experiment—not if it merely has many glowing nodes.

### Target: `v0.4 — Routed Thought`

Use a small local MoE to show real expert-routing events and compare them across prompts and steering interventions.

### Target: `v0.5 — Backend Laboratory`

Run the same experiment against at least three provider types and produce a comparison report:

- closed API provider;
- small instrumented dense local model;
- small instrumented MoE;
- optional Colibrì protocol fixture.

---

## Steering semantics

Separate several kinds of “steering” in the UI and data model:

1. **Branch replay** — choose an alternative token and regenerate from the modified prefix.
2. **Sampling intervention** — change temperature/top-p/top-k/seed and rerun.
3. **Logit intervention** — apply a documented token bias before sampling, only when the backend supports it.
4. **Activation intervention** — experimental manipulation of hidden states or expert routing, only in explicitly instrumented local runtimes.
5. **Narrative/UI splice** — synthetic demonstration only; must never be shown as a model-native branch.

For each intervention, record exactly which computation was cancelled, reused, invalidated, and recomputed.

---

## Confidence and epistemic measurement

Do not use the word “confidence” as one undifferentiated score.

Implement separate measurements:

- selected-token probability;
- token entropy;
- top-1/top-2 margin;
- sequence log-likelihood where meaningful;
- model-verbalized confidence;
- repeated-sample agreement;
- semantic answer agreement;
- benchmark correctness;
- calibration error after evaluation;
- abstention threshold and risk/coverage.

Label verbalized confidence as a model claim, not an internal probability.

Create a small evaluation harness that can run reproducible tests and persist:

- prompts and expected answers;
- exact model/runtime/config revisions;
- outputs and telemetry;
- correctness;
- calibration metrics;
- environment and hardware metadata.

Start with a compact, quick suite. Do not make full benchmark completion a prerequisite for the product.

---

## Data and storage

Extend SQLite or introduce a compatible event-store layer with migrations.

At minimum support:

- schema versions;
- providers and capability snapshots;
- models and immutable model revisions;
- runs and generation settings;
- token events;
- alternatives;
- layer summaries;
- router decisions;
- resource samples;
- interventions;
- evidence records;
- unavailable measurements;
- artifacts and hashes;
- benchmark items and evaluation results.

Use compressed external artifacts for large tensors or traces. Store hashes and provenance in SQLite. Add export to JSONL for research use.

---

## User interface

Preserve the graph, but restructure the experience around comprehension:

- **Run strip:** prompt, backend, model, status, timing, and reproducibility fingerprint.
- **Token lane:** emitted tokens with probability/entropy/latency overlays.
- **Mechanics lane:** layer/router/cache/runtime events aligned to token time.
- **Evidence inspector:** raw source, formula, caveat, and status.
- **Capability matrix:** what this backend can actually reveal.
- **Intervention ledger:** before/after prefixes, invalidated work, and divergence.
- **Comparison mode:** align two runs by prompt and token/semantic segment.
- **Plain mode / Expert mode:** preserve accessibility without concealing evidence.

Avoid implying that neuron-like graphics represent literal neurons unless they do.

---

## Security and privacy

- Bind local services to `127.0.0.1` by default.
- Never commit model weights, caches, databases, prompts, API keys, or generated private content.
- Add and test `.gitignore` coverage.
- Redact secrets from logs and evidence payloads.
- Add bounded trace sizes and retention controls.
- Treat persisted KV caches and prompts as sensitive.
- Provide a one-command local data purge that reports exactly what it removed.

---

## Engineering process

### First pass: audit and plan

1. Run the existing test suite.
2. Map the current provider, metric, evidence, graph, persistence, and websocket flows.
3. Write `docs/architecture/current-state.md`.
4. Write architecture decision records for the telemetry schema, local reference runtime, and model selection.
5. Create a milestone checklist in `docs/ROADMAP.md`.

### Implementation order

1. Add capability and `UNAVAILABLE` schemas.
2. Add normalized versioned telemetry events.
3. Implement the small reference instrumented local provider.
4. Surface real logits/top alternatives/entropy/hidden-state summaries.
5. Add system resource telemetry.
6. Refine steering provenance and replay semantics.
7. Add llama.cpp provider.
8. Add small-MoE routing instrumentation.
9. Add Colibrì adapter contract, fixtures, and sidecar design.
10. Add benchmark/calibration harness.
11. Polish UI and documentation.

### Quality gates

For every milestone:

- tests pass;
- new metrics include provenance and status;
- no synthetic value appears without an unmistakable label;
- memory use is bounded on a 16 GB machine;
- failure modes are displayed rather than hidden;
- documentation explains what is measured and what remains unknowable;
- screenshots or a short demo capture prove the feature visibly works.

Use focused commits. Prefer a draft pull request with a running checklist over one enormous opaque change.

---

## Definition of done for this mission

The mission is complete when:

1. Cogito Console runs locally on Astræos with a free small model.
2. The UI displays real token probabilities/alternatives and real runtime metrics.
3. At least one internal model signal is directly measured and visibly distinguished from derived interpretation.
4. A small MoE experiment emits real routing telemetry, or a precise documented blocker and working instrumentation fixture exists.
5. The Colibrì adapter and telemetry-extension design exist without requiring the GLM-5.2 download.
6. The same prompt can be compared across backends with capability-aware missing-data handling.
7. Steering actions are reproducible and provenance-complete.
8. Tests, architecture docs, setup instructions, and a portfolio demonstration are complete.
9. Every claim in the README survives the question: **“Where did this number come from?”**

---

## Final orientation

Do not optimize for pretending Cogito Console can see every thought.

Optimize for making every accessible computational event inspectable, every transformation auditable, every absence explicit, and every intervention reproducible.

Build the credible instrument first. Let the titan enter the laboratory later.
