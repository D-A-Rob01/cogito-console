# Cogito Console roadmap

Updated: 2026-07-13

## v0.3 - Measured Inference

- [x] Preserve the standing mission in `CODEX_MISSION.md`.
- [x] Audit tests, provider flow, evidence/metrics, graph, persistence, websocket steering, and UI.
- [x] Record the current architecture and initial decisions.
- [ ] Add first-class provider capabilities and `UNAVAILABLE` measurements.
- [ ] Add the versioned normalized telemetry envelope and SQLite migration.
- [ ] Add JSONL trace export and bounded local data purge.
- [ ] Add a lazy reference local provider with registry-selected model identity.
- [ ] Emit real token IDs, logits-derived logprob/top-k/probability/entropy/margin.
- [ ] Emit real prefill, time-to-first-token, per-token latency, RAM, and CPU telemetry.
- [ ] Emit real per-layer last-token hidden-state summaries.
- [ ] Record exact generation settings, revisions, environment, and fingerprint.
- [ ] Make capabilities and unavailable signals visible before a run.
- [ ] Align token lane, mechanics lane, evidence inspector, and replay ledger.
- [ ] Distinguish true branch replay from synthetic narrative splice.
- [ ] Add a quick reproducible evaluation/experiment fixture.
- [ ] Run automated, API, websocket, and browser QA.
- [ ] Add screenshots and a recorded experiment.
- [ ] Publish focused commits and a draft pull request.

## v0.4 - Routed Thought

- [ ] Add a bounded small-MoE provider or a working replay fixture plus documented hardware blocker.
- [ ] Emit selected experts, router weights, routing entropy, reuse, and load balance.
- [ ] Compare routing before and after an intervention.

## v0.5 - Backend Laboratory

- [ ] Add a practical llama.cpp provider.
- [ ] Add the Colibri adapter contract, fixtures, and telemetry sidecar proposal without GLM-5.2.
- [ ] Compare the same prompt across closed API, dense local, small-MoE, and fixture backends.
- [ ] Add compact correctness/calibration evaluation and comparison export.

## Known later work

- Native Intel acceleration is capability-detected research, never a v0.3 requirement.
- Large model downloads and GLM-5.2 remain prohibited for the initial mission.
- Full tensor retention remains opt-in and external-artifact-backed if it is ever introduced.
