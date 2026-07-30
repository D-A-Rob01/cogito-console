# ADR 0003: Registry-selected SmolLM2 reference model

Status: Accepted
Date: 2026-07-13

## Context

The mission prefers a 1B-3B reference model, but Astræos has 16 GB of shared system memory and the first milestone must remain easy to install, test, and demonstrate. The instrument must not be branded around one checkpoint.

## Decision

Use a model registry and select `HuggingFaceTB/SmolLM2-360M-Instruct` as the initial default specimen.

Reasons:

- Apache-2.0 model card;
- native Transformers causal-LM support;
- small enough for conservative CPU-first loading and repeated instrumented forward passes;
- hidden states and raw logits are available through the standard model output;
- the checkpoint is replaceable through `COGITO_LOCAL_MODEL` and registry entries.

Default guardrails:

- float32 CPU inference;
- at most 512 input tokens and 32 new tokens unless configured lower;
- five stored alternatives per token;
- layer summaries only, never full hidden-state persistence;
- resolved model and tokenizer commit hashes recorded when available;
- model files remain in the external Hugging Face cache and are ignored by Git.

## Consequences

The 360M model is below the mission's preferred 1B-3B range. This is an intentional hardware-risk reduction for the first observable release, not a permanent ceiling. A 1.7B registry entry can be evaluated after the 360M path is stable and measured on Astræos.
