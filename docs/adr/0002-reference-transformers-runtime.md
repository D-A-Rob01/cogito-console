# ADR 0002: Direct Transformers reference runtime

Status: Accepted
Date: 2026-07-13

## Context

The first credible release needs real logits, alternatives, entropy, timing, and at least one internal activation summary on a 16 GB Windows machine. OpenAI-compatible servers often omit hidden states; llama.cpp is the later responsive path, not the best initial instrumentation path.

## Decision

Implement a lazy CPU-first `TransformersReferenceAdapter` using direct PyTorch forward passes and a bounded manual autoregressive loop.

The adapter will:

- import optional dependencies only when selected;
- load one registry-selected causal language model per process;
- cap input and output tokens;
- run under inference mode with KV cache enabled;
- compute selected-token logprob, probability, full-distribution entropy, top-1/top-2 probability margin, and top-k alternatives from real logits;
- summarize each layer's last-token L2 norm and layer-to-layer L2 change without retaining tensors;
- record prefill time, time to first token, decode latency, process RSS, system RAM, and CPU;
- serialize access to the model so concurrent sessions cannot corrupt cache state;
- make attention collection opt-in because full attention tensors are expensive and some optimized attention paths do not return them.

The default runtime is deliberately slower than a native engine. Instrumentation fidelity and reproducibility are the priority for v0.3.

## Consequences

- first startup downloads a small openly licensed model and may take longer;
- CPU decode is suitable for short experiments, not high-throughput chat;
- the provider is testable with injected fake model/tokenizer objects without downloading weights;
- llama.cpp remains a separate provider for later responsive experiments.
