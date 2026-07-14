from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from .metrics import MetricEngine
from .model_registry import ReferenceModelSpec, get_reference_model
from .providers import (
    ProviderAdapter,
    ProviderStreamPacket,
    ProviderUnavailableError,
    _cap,
    _capabilities,
)
from .schemas import (
    AlternativeTokenEvent,
    EvidenceRecord,
    MetricValue,
    ProviderCapabilityRecord,
    TelemetryEvent,
    TokenEvent,
    make_evidence,
)


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    max_input_tokens: int = 512
    max_new_tokens: int = 32
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = False
    seed: int = 1337
    top_alternatives: int = 5

    @classmethod
    def from_env(cls, spec: ReferenceModelSpec) -> "GenerationSettings":
        return cls(
            max_input_tokens=int(
                os.getenv("COGITO_MAX_INPUT_TOKENS", str(spec.default_max_input_tokens))
            ),
            max_new_tokens=int(
                os.getenv("COGITO_MAX_NEW_TOKENS", str(spec.default_max_new_tokens))
            ),
            temperature=float(os.getenv("COGITO_TEMPERATURE", "1.0")),
            top_k=int(os.getenv("COGITO_TOP_K", "0")),
            top_p=float(os.getenv("COGITO_TOP_P", "1.0")),
            do_sample=os.getenv("COGITO_DO_SAMPLE", "false").lower() == "true",
            seed=int(os.getenv("COGITO_SEED", "1337")),
            top_alternatives=int(os.getenv("COGITO_TOP_ALTERNATIVES", "5")),
        ).validated(spec)

    def validated(self, spec: ReferenceModelSpec) -> "GenerationSettings":
        if not 1 <= self.max_input_tokens <= min(spec.max_context_tokens, 2048):
            raise ValueError("max_input_tokens must be between 1 and 2048")
        if not 1 <= self.max_new_tokens <= 128:
            raise ValueError("max_new_tokens must be between 1 and 128")
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 1 <= self.top_alternatives <= 20:
            raise ValueError("top_alternatives must be between 1 and 20")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    registry_key: str
    model_id: str
    requested_revision: str
    model_revision: str
    tokenizer_revision: str
    parameter_count: int
    license: str
    dtype: str
    quantization: str
    device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LayerSummary:
    layer_index: int
    l2_norm: float
    delta_l2: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReferenceAlternative:
    token_id: int
    text: str
    logprob: float
    probability: float


@dataclass(frozen=True, slots=True)
class ReferenceStep:
    position: int
    tokenizer_token_id: int
    text: str
    selected_logprob: float
    selected_probability: float
    sampling_probability: float
    entropy_bits: float
    top1_top2_margin: float
    alternatives: list[ReferenceAlternative]
    layer_summaries: list[LayerSummary]
    resource_sample: dict[str, float | int | None]
    latency_ms: float
    time_to_first_token_ms: float | None
    prefill_ms: float | None
    cold_load_ms: float | None
    prompt_tokens: int
    raw_logit_summary: dict[str, float]
    model_identity: ModelIdentity
    generation_settings: GenerationSettings
    run_fingerprint: str


class TransformersReferenceRuntime:
    """Bounded manual autoregressive loop that retains summaries, never tensors."""

    def __init__(
        self,
        spec: ReferenceModelSpec | None = None,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        torch_module: Any | None = None,
        psutil_module: Any | None = None,
    ) -> None:
        self.spec = spec or get_reference_model()
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.psutil = psutil_module
        self.process = None
        self.identity: ModelIdentity | None = None

    @staticmethod
    def missing_dependencies() -> list[str]:
        return [
            package
            for package in ("torch", "transformers", "psutil")
            if importlib.util.find_spec(package) is None
        ]

    def ensure_loaded(self) -> float:
        if self.identity is not None:
            return 0.0
        started = time.perf_counter()
        if self.torch is None:
            import torch

            self.torch = torch
        if self.psutil is None:
            import psutil

            self.psutil = psutil
        if self.model is None or self.tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.spec.model_id,
                revision=self.spec.revision,
                trust_remote_code=self.spec.trust_remote_code,
                use_fast=True,
            )
            import transformers

            dtype_key = (
                "dtype"
                if int(transformers.__version__.split(".", 1)[0]) >= 5
                else "torch_dtype"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.spec.model_id,
                revision=self.spec.revision,
                trust_remote_code=self.spec.trust_remote_code,
                **{dtype_key: getattr(self.torch, self.spec.torch_dtype)},
            )
        self.model.to("cpu")
        self.model.eval()
        self.process = self.psutil.Process()
        self.process.cpu_percent(interval=None)
        model_revision = str(
            getattr(getattr(self.model, "config", None), "_commit_hash", None)
            or self.spec.revision
        )
        tokenizer_revision = str(
            getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
            or getattr(self.tokenizer, "_commit_hash", None)
            or model_revision
        )
        dtype = str(next(self.model.parameters()).dtype).replace("torch.", "")
        self.identity = ModelIdentity(
            registry_key=self.spec.key,
            model_id=self.spec.model_id,
            requested_revision=self.spec.revision,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            parameter_count=self.spec.parameter_count,
            license=self.spec.license,
            dtype=dtype,
            quantization="none",
            device="cpu",
        )
        return (time.perf_counter() - started) * 1000

    def generate(
        self,
        prompt: str,
        system_context: str,
        history: list[str],
        settings: GenerationSettings,
    ) -> Iterator[ReferenceStep]:
        request_started = time.perf_counter()
        cold_load_ms = self.ensure_loaded()
        assert self.identity is not None
        prompt_text = self._format_prompt(prompt, system_context, history)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=settings.max_input_tokens,
        )
        input_ids = encoded["input_ids"].to("cpu")
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = self.torch.ones_like(input_ids)
        attention_mask = attention_mask.to("cpu")
        prompt_tokens = int(input_ids.shape[-1])
        generator = self.torch.Generator(device="cpu")
        generator.manual_seed(settings.seed)
        fingerprint = _fingerprint(
            prompt_text,
            self.identity,
            settings,
        )

        prefill_started = time.perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=False,
            )
        prefill_ms = (time.perf_counter() - prefill_started) * 1000
        logits = outputs.logits[0, -1, :]
        hidden_states = outputs.hidden_states
        past_key_values = outputs.past_key_values
        first_step = True
        position_offset = len(history)
        step_started = prefill_started

        for generated_index in range(settings.max_new_tokens):
            (
                selected_id,
                selected_logprob,
                selected_probability,
                sampling_probability,
                entropy_bits,
                top1_top2_margin,
                alternatives,
                raw_logit_summary,
            ) = self._sample(logits, settings, generator)
            eos_ids = self._eos_token_ids()
            if selected_id in eos_ids:
                break
            token_text = self.tokenizer.decode(
                [selected_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            layer_summaries = self._summarize_layers(hidden_states)
            latency_ms = (time.perf_counter() - step_started) * 1000
            ttft_ms = (
                (time.perf_counter() - request_started) * 1000
                if first_step
                else None
            )
            yield ReferenceStep(
                position=position_offset + generated_index,
                tokenizer_token_id=selected_id,
                text=token_text,
                selected_logprob=selected_logprob,
                selected_probability=selected_probability,
                sampling_probability=sampling_probability,
                entropy_bits=entropy_bits,
                top1_top2_margin=top1_top2_margin,
                alternatives=alternatives,
                layer_summaries=layer_summaries,
                resource_sample=self._resource_sample(),
                latency_ms=latency_ms,
                time_to_first_token_ms=ttft_ms,
                prefill_ms=prefill_ms if first_step else None,
                cold_load_ms=cold_load_ms if first_step else None,
                prompt_tokens=prompt_tokens,
                raw_logit_summary=raw_logit_summary,
                model_identity=self.identity,
                generation_settings=settings,
                run_fingerprint=fingerprint,
            )
            first_step = False

            step_started = time.perf_counter()
            next_input = self.torch.tensor([[selected_id]], dtype=self.torch.long)
            attention_mask = self.torch.cat(
                [
                    attention_mask,
                    self.torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=-1,
            )
            with self.torch.inference_mode():
                outputs = self.model(
                    input_ids=next_input,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=False,
                )
            logits = outputs.logits[0, -1, :]
            hidden_states = outputs.hidden_states
            past_key_values = outputs.past_key_values

    def _format_prompt(
        self,
        prompt: str,
        system_context: str,
        history: list[str],
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": system_context
                or "You are a concise local model running under measurement.",
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                base = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return str(base) + "".join(history)
            except (ValueError, TypeError, AttributeError):
                pass
        return (
            f"System: {messages[0]['content']}\nUser: {prompt}\nAssistant: "
            + "".join(history)
        )

    def _sample(
        self,
        logits: Any,
        settings: GenerationSettings,
        generator: Any,
    ) -> tuple[
        int,
        float,
        float,
        float,
        float,
        float,
        list[ReferenceAlternative],
        dict[str, float],
    ]:
        raw_logits = logits.detach().float().cpu()
        model_logprobs = self.torch.log_softmax(raw_logits, dim=-1)
        model_probabilities = model_logprobs.exp()
        top_two = self.torch.topk(model_probabilities, k=2)
        top1_top2_margin = float((top_two.values[0] - top_two.values[1]).item())
        entropy_bits = float(
            -self.torch.sum(
                model_probabilities * model_logprobs / self.torch.log(self.torch.tensor(2.0))
            ).item()
        )

        if settings.do_sample:
            sampler_logits = raw_logits / settings.temperature
            sampler_logits = self._apply_top_k_top_p(
                sampler_logits, settings.top_k, settings.top_p
            )
            sampler_probabilities = self.torch.softmax(sampler_logits, dim=-1)
            selected_tensor = self.torch.multinomial(
                sampler_probabilities,
                num_samples=1,
                generator=generator,
            )
            selected_id = int(selected_tensor.item())
            sampling_probability = float(sampler_probabilities[selected_id].item())
        else:
            selected_id = int(self.torch.argmax(raw_logits).item())
            sampling_probability = 1.0

        selected_logprob = float(model_logprobs[selected_id].item())
        selected_probability = float(model_probabilities[selected_id].item())
        requested = min(settings.top_alternatives + 1, int(raw_logits.shape[-1]))
        top = self.torch.topk(model_logprobs, k=requested)
        alternatives: list[ReferenceAlternative] = []
        for token_id, logprob in zip(top.indices.tolist(), top.values.tolist()):
            if int(token_id) == selected_id:
                continue
            alternatives.append(
                ReferenceAlternative(
                    token_id=int(token_id),
                    text=self.tokenizer.decode(
                        [int(token_id)],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    logprob=float(logprob),
                    probability=float(self.torch.exp(self.torch.tensor(logprob)).item()),
                )
            )
            if len(alternatives) >= settings.top_alternatives:
                break

        raw_logit_summary = {
            "minimum": float(raw_logits.min().item()),
            "maximum": float(raw_logits.max().item()),
            "mean": float(raw_logits.mean().item()),
        }
        return (
            selected_id,
            selected_logprob,
            selected_probability,
            sampling_probability,
            entropy_bits,
            top1_top2_margin,
            alternatives,
            raw_logit_summary,
        )

    def _apply_top_k_top_p(self, logits: Any, top_k: int, top_p: float) -> Any:
        filtered = logits.clone()
        if top_k > 0:
            cutoff = self.torch.topk(
                filtered, k=min(top_k, int(filtered.shape[-1]))
            ).values[-1]
            filtered[filtered < cutoff] = -float("inf")
        if top_p < 1.0:
            sorted_logits, sorted_indices = self.torch.sort(filtered, descending=True)
            sorted_probabilities = self.torch.softmax(sorted_logits, dim=-1)
            cumulative = self.torch.cumsum(sorted_probabilities, dim=-1)
            remove = cumulative > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            filtered[sorted_indices[remove]] = -float("inf")
        return filtered

    def _summarize_layers(self, hidden_states: Any) -> list[LayerSummary]:
        summaries: list[LayerSummary] = []
        previous = None
        for layer_index, hidden_state in enumerate(hidden_states or ()):
            vector = hidden_state[0, -1, :].detach().float().cpu()
            norm = float(self.torch.linalg.vector_norm(vector).item())
            delta = (
                float(self.torch.linalg.vector_norm(vector - previous).item())
                if previous is not None
                else None
            )
            summaries.append(
                LayerSummary(
                    layer_index=layer_index,
                    l2_norm=norm,
                    delta_l2=delta,
                )
            )
            previous = vector
        return summaries

    def _resource_sample(self) -> dict[str, float | int | None]:
        if self.process is None:
            return {
                "process_rss_bytes": None,
                "process_cpu_percent": None,
                "system_cpu_percent": None,
                "system_memory_used_bytes": None,
                "system_memory_percent": None,
            }
        memory = self.psutil.virtual_memory()
        return {
            "process_rss_bytes": int(self.process.memory_info().rss),
            "process_cpu_percent": float(self.process.cpu_percent(interval=None)),
            "system_cpu_percent": float(self.psutil.cpu_percent(interval=None)),
            "system_memory_used_bytes": int(memory.used),
            "system_memory_percent": float(memory.percent),
        }

    def _eos_token_ids(self) -> set[int]:
        value = getattr(self.tokenizer, "eos_token_id", None)
        if value is None:
            return set()
        if isinstance(value, int):
            return {value}
        return {int(item) for item in value}


class TransformersReferenceAdapter(ProviderAdapter):
    name = "Transformers reference local"
    mode = "transformers"

    def __init__(
        self,
        *,
        spec: ReferenceModelSpec | None = None,
        settings: GenerationSettings | None = None,
        runtime: TransformersReferenceRuntime | Any | None = None,
    ) -> None:
        self.spec = spec or get_reference_model()
        self.settings = settings or GenerationSettings.from_env(self.spec)
        self.runtime = runtime or TransformersReferenceRuntime(self.spec)
        self._run_lock = asyncio.Lock()

    def is_available(self) -> bool:
        if not isinstance(self.runtime, TransformersReferenceRuntime):
            return True
        return not self.runtime.missing_dependencies()

    def unavailable_reason(self) -> str | None:
        if self.is_available():
            return None
        missing = ", ".join(self.runtime.missing_dependencies())
        return (
            "Transformers reference adapter unavailable: install the local extras "
            f"(missing: {missing})."
        )

    def capability_record(self) -> ProviderCapabilityRecord:
        return ProviderCapabilityRecord(
            provider_id="transformers",
            provider_name=self.name,
            mode=self.mode,
            capabilities=_capabilities(
                streaming=_cap("streaming", "supported", "token", status="real"),
                token_ids=_cap("token_ids", "supported", "token", status="real"),
                selected_token_logprobs=_cap(
                    "selected_token_logprobs", "supported", "token", status="real"
                ),
                top_k_alternatives=_cap(
                    "top_k_alternatives", "supported", "token", status="real"
                ),
                raw_logits=_cap(
                    "raw_logits",
                    "supported",
                    "token decision",
                    "Full tensors are summarized and discarded after each step.",
                    "real",
                ),
                hidden_states=_cap(
                    "hidden_states",
                    "supported",
                    "layer per token decision",
                    "Only last-position L2 norm and layer-to-layer delta are retained.",
                    "real",
                ),
                attentions=_cap(
                    "attentions",
                    "conditional",
                    "layer per token decision",
                    "Disabled in v0.3 to bound memory; no attention value is inferred.",
                    "unavailable",
                ),
                kv_cache_metrics=_cap(
                    "kv_cache_metrics",
                    "conditional",
                    "run",
                    "KV cache is used, but byte accounting is not yet normalized.",
                    "unavailable",
                ),
                system_resource_telemetry=_cap(
                    "system_resource_telemetry",
                    "supported",
                    "token",
                    "CPU and RAM describe the local host and Cogito process.",
                    "real",
                ),
                deterministic_seed=_cap(
                    "deterministic_seed", "supported", "run", status="real"
                ),
                branch_replay=_cap(
                    "branch_replay",
                    "supported",
                    "run prefix",
                    "The modified prefix is re-encoded and the continuation recomputed.",
                    "real",
                ),
                sampling_intervention=_cap(
                    "sampling_intervention",
                    "supported",
                    "run",
                    "Settings are fixed for a run and persisted with its fingerprint.",
                    "real",
                ),
            ),
            steering_methods=["branch_replay", "sampling_intervention"],
        )

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "model": self.spec.to_dict(),
                "generation_defaults": self.settings.to_dict(),
                "runtime_policy": (
                    "CPU-first manual decode; compact summaries retained; tensors discarded."
                ),
            }
        )
        return payload

    async def stream(
        self,
        prompt: str,
        system_context: str,
        history: list[str],
        *,
        session_id: str,
        run_id: str,
        generation_settings: dict[str, Any] | None = None,
    ):
        if not self.is_available():
            raise ProviderUnavailableError(self.unavailable_reason() or "local runtime unavailable")
        async with self._run_lock:
            settings = self._settings_for_run(generation_settings)
            iterator = self.runtime.generate(
                prompt,
                system_context,
                history,
                settings,
            )
            while True:
                step = await asyncio.to_thread(_next_or_none, iterator)
                if step is None:
                    break
                yield self._packet_from_step(step, session_id=session_id, run_id=run_id)

    def _settings_for_run(
        self, overrides: dict[str, Any] | None
    ) -> GenerationSettings:
        if not overrides:
            return self.settings
        allowed = set(self.settings.to_dict())
        unknown = set(overrides) - allowed
        if unknown:
            raise ProviderUnavailableError(
                f"Unsupported generation settings: {', '.join(sorted(unknown))}"
            )
        values = self.settings.to_dict()
        values.update(overrides)
        try:
            return GenerationSettings(**values).validated(self.spec)
        except (TypeError, ValueError) as exc:
            raise ProviderUnavailableError(f"Invalid generation settings: {exc}") from exc

    def _packet_from_step(
        self,
        step: ReferenceStep,
        *,
        session_id: str,
        run_id: str,
    ) -> ProviderStreamPacket:
        evidence: list[EvidenceRecord] = []
        metrics: list[MetricValue] = []

        logprob_evidence = _evidence(
            evidence,
            self.name,
            "model.logits[last_position]",
            "selected_token_logprobs",
            caveat=(
                "Logprob uses the unfiltered model distribution. Sampling transforms are "
                "recorded separately when enabled."
            ),
        )
        logprob = _metric(
            name="token_logprob",
            value=step.selected_logprob,
            unit="natural_log_probability",
            status="real",
            evidence=logprob_evidence,
            label="Selected-token logprob",
            explanation="The model log-probability assigned to the emitted token.",
        )
        metrics.append(logprob)

        probability_evidence = _evidence(
            evidence,
            self.name,
            "model.logits[last_position]",
            "raw_logits",
            source_kind="derived",
            formula="selected_token_probability = exp(selected_token_logprob)",
        )
        probability = MetricEngine.token_probability(
            logprob, probability_evidence.evidence_id
        )
        metrics.append(probability)

        token_id_evidence = _evidence(
            evidence,
            self.name,
            "tokenizer token id",
            "token_ids",
        )
        tokenizer_token_id = _metric(
            name="tokenizer_token_id",
            value=step.tokenizer_token_id,
            unit="token_id",
            status="real",
            evidence=token_id_evidence,
            label="Tokenizer token ID",
            explanation="The exact vocabulary ID emitted by the reference tokenizer.",
        )
        metrics.append(tokenizer_token_id)

        entropy_evidence = _evidence(
            evidence,
            self.name,
            "full model probability distribution",
            "raw_logits",
            source_kind="derived",
            formula="entropy_bits = -sum(p_i * log2(p_i)) over the full vocabulary",
        )
        entropy = _metric(
            name="token_entropy_bits",
            value=step.entropy_bits,
            unit="bits",
            status="derived",
            evidence=entropy_evidence,
            label="Token entropy",
            explanation="Uncertainty across the full next-token distribution.",
        )
        metrics.append(entropy)

        margin_evidence = _evidence(
            evidence,
            self.name,
            "two largest full-distribution probabilities",
            "raw_logits",
            source_kind="derived",
            formula="top1_top2_margin = probability(top_1) - probability(top_2)",
        )
        margin = _metric(
            name="top1_top2_probability_margin",
            value=step.top1_top2_margin,
            unit="probability",
            status="derived",
            evidence=margin_evidence,
            label="Top-1 / top-2 margin",
            explanation="Probability distance between the two most likely tokens.",
        )
        metrics.append(margin)

        latency_evidence = _evidence(
            evidence,
            "Python perf counter",
            "manual decode step",
            "system_resource_telemetry",
            caveat="First-token latency includes the prefill forward pass.",
        )
        latency = _metric(
            name="token_latency_ms",
            value=step.latency_ms,
            unit="ms",
            status="real",
            evidence=latency_evidence,
            label="Per-token latency",
            explanation="Elapsed local runtime time for this token decision.",
        )
        metrics.append(latency)

        byte_evidence = _evidence(
            evidence,
            "Python runtime",
            "len(token.encode('utf-8'))",
            "streaming",
        )
        byte_length = _metric(
            name="byte_length",
            value=len(step.text.encode("utf-8")),
            unit="bytes",
            status="real",
            evidence=byte_evidence,
            label="Byte length",
            explanation="UTF-8 byte length of the emitted token text.",
        )
        metrics.append(byte_length)

        if step.time_to_first_token_ms is not None:
            ttft_evidence = _evidence(
                evidence,
                "Python perf counter",
                "request start to first sampled token",
                "system_resource_telemetry",
                caveat="A cold run includes model load and model-cache initialization.",
            )
            metrics.append(
                _metric(
                    name="time_to_first_token_ms",
                    value=step.time_to_first_token_ms,
                    unit="ms",
                    status="real",
                    evidence=ttft_evidence,
                    label="Time to first token",
                    explanation="Elapsed time from local request start to first token.",
                )
            )

        alternatives = self._alternatives(step, logprob, evidence, metrics)
        hidden_metrics, layer_events = self._layer_summaries(
            step, session_id, run_id, evidence
        )
        metrics.extend(hidden_metrics.values())
        resource_metrics, resource_event = self._resource_metrics(
            step, session_id, run_id, evidence
        )
        metrics.extend(resource_metrics)

        telemetry_events = layer_events + [resource_event]
        if step.prefill_ms is not None:
            telemetry_events = [
                TelemetryEvent(
                    event_type="prompt.encoded",
                    session_id=session_id,
                    run_id=run_id,
                    status="real",
                    source_name=step.model_identity.model_id,
                    capability="token_ids",
                    payload={
                        "prompt_tokens": step.prompt_tokens,
                        "max_input_tokens": step.generation_settings.max_input_tokens,
                    },
                ),
                TelemetryEvent(
                    event_type="prefill.started",
                    session_id=session_id,
                    run_id=run_id,
                    status="real",
                    source_name=step.model_identity.model_id,
                    capability="streaming",
                    payload={"prompt_tokens": step.prompt_tokens},
                ),
                TelemetryEvent(
                    event_type="prefill.completed",
                    session_id=session_id,
                    run_id=run_id,
                    status="real",
                    source_name=step.model_identity.model_id,
                    capability="streaming",
                    payload={
                        "prompt_tokens": step.prompt_tokens,
                        "prefill_ms": step.prefill_ms,
                        "cold_load_ms": step.cold_load_ms,
                    },
                ),
            ] + telemetry_events

        token_event = TokenEvent(
            session_id=session_id,
            run_id=run_id,
            token_id=step.position,
            text=step.text,
            logprob=logprob,
            probability=probability,
            byte_length=byte_length,
            latency_ms=latency,
            alternatives=alternatives,
            raw_provider_payload={
                "source": "transformers_reference_runtime",
                "model_identity": step.model_identity.to_dict(),
                "generation_settings": step.generation_settings.to_dict(),
                "run_fingerprint": step.run_fingerprint,
                "prompt_tokens": step.prompt_tokens,
                "tokenizer_token_id": step.tokenizer_token_id,
                "entropy_bits": step.entropy_bits,
                "top1_top2_margin": step.top1_top2_margin,
                "sampling_probability": step.sampling_probability,
                "raw_logit_summary": step.raw_logit_summary,
                "layer_summaries": [
                    summary.to_dict() for summary in step.layer_summaries
                ],
                "resource_sample": step.resource_sample,
            },
        )
        packet = ProviderStreamPacket(
            token_event=token_event,
            evidence=evidence,
            metrics=metrics,
            hidden_state_metrics=hidden_metrics,
            telemetry_events=telemetry_events,
        )
        packet.ensure_telemetry_events(self.name)
        return packet

    def _alternatives(
        self,
        step: ReferenceStep,
        selected_logprob: MetricValue,
        evidence: list[EvidenceRecord],
        metrics: list[MetricValue],
    ) -> list[AlternativeTokenEvent]:
        result: list[AlternativeTokenEvent] = []
        for rank, alternative in enumerate(step.alternatives, start=1):
            logprob_evidence = _evidence(
                evidence,
                self.name,
                "top-k model logits",
                "top_k_alternatives",
            )
            logprob = _metric(
                name=f"alternative_{rank}_logprob",
                value=alternative.logprob,
                unit="natural_log_probability",
                status="real",
                evidence=logprob_evidence,
                label="Alternative logprob",
                explanation="Model log-probability for this alternate token.",
            )
            metrics.append(logprob)
            probability_evidence = _evidence(
                evidence,
                self.name,
                "top-k model logits",
                "top_k_alternatives",
                source_kind="derived",
                formula="alternative_probability = exp(alternative_logprob)",
            )
            probability = MetricEngine.token_probability(
                logprob, probability_evidence.evidence_id
            )
            metrics.append(probability)
            margin_evidence = _evidence(
                evidence,
                self.name,
                "selected and alternate logprobs",
                "top_k_alternatives",
                source_kind="derived",
                formula="selected_logprob - alternative_logprob",
            )
            alternative_margin = MetricEngine.alternative_margin(
                selected_logprob, logprob, margin_evidence.evidence_id
            )
            metrics.append(alternative_margin)
            result.append(
                AlternativeTokenEvent(
                    token=alternative.text,
                    rank=rank,
                    logprob=logprob,
                    probability=probability,
                    margin_from_selected=alternative_margin,
                    rationale=(
                        f"Tokenizer ID {alternative.token_id}; full-distribution rank {rank + 1}."
                    ),
                    status="real",
                )
            )
        return result

    def _layer_summaries(
        self,
        step: ReferenceStep,
        session_id: str,
        run_id: str,
        evidence: list[EvidenceRecord],
    ) -> tuple[dict[str, MetricValue], list[TelemetryEvent]]:
        metrics: dict[str, MetricValue] = {}
        events: list[TelemetryEvent] = []
        for summary in step.layer_summaries:
            norm_evidence = _evidence(
                evidence,
                step.model_identity.model_id,
                f"hidden_states[{summary.layer_index}][last_position]",
                "hidden_states",
                source_kind="derived",
                formula="l2_norm = sqrt(sum(hidden_state_i ** 2))",
                caveat=(
                    "The vector is the last decision-state position used to select this token; "
                    "the full vector is discarded."
                ),
            )
            norm = _metric(
                name=f"layer_{summary.layer_index}_l2_norm",
                value=summary.l2_norm,
                unit="l2_norm",
                status="derived",
                evidence=norm_evidence,
                label=f"Layer {summary.layer_index} L2 norm",
                explanation="Magnitude of the real last-position hidden-state vector.",
            )
            metrics[f"layer_{summary.layer_index}_l2_norm"] = norm
            evidence_ids = [norm.evidence_id]
            delta = None
            if summary.delta_l2 is not None:
                delta_evidence = _evidence(
                    evidence,
                    step.model_identity.model_id,
                    f"hidden_states[{summary.layer_index}] - hidden_states[{summary.layer_index - 1}]",
                    "hidden_states",
                    source_kind="derived",
                    formula="delta_l2 = ||hidden_state_layer - hidden_state_previous_layer||_2",
                    caveat="Summary of real hidden states; full tensors are discarded.",
                )
                delta = _metric(
                    name=f"layer_{summary.layer_index}_delta_l2",
                    value=summary.delta_l2,
                    unit="l2_norm",
                    status="derived",
                    evidence=delta_evidence,
                    label=f"Layer {summary.layer_index} change",
                    explanation="Layer-to-layer change in the real hidden-state vector.",
                )
                metrics[f"layer_{summary.layer_index}_delta_l2"] = delta
                evidence_ids.append(delta.evidence_id)
            events.append(
                TelemetryEvent(
                    event_type="layer.summary",
                    session_id=session_id,
                    run_id=run_id,
                    status="derived",
                    source_name=step.model_identity.model_id,
                    capability="hidden_states",
                    payload={
                        "token_id": step.position,
                        "layer_index": summary.layer_index,
                        "l2_norm": norm.to_dict(),
                        "delta_l2": delta.to_dict() if delta else None,
                        "source_tensor_status": "real",
                    },
                    token_id=step.position,
                    layer_index=summary.layer_index,
                    evidence_ids=evidence_ids,
                )
            )
        return metrics, events

    def _resource_metrics(
        self,
        step: ReferenceStep,
        session_id: str,
        run_id: str,
        evidence: list[EvidenceRecord],
    ) -> tuple[list[MetricValue], TelemetryEvent]:
        metrics: list[MetricValue] = []
        for key, value in step.resource_sample.items():
            if value is None:
                continue
            unit = "bytes" if key.endswith("_bytes") else "percent"
            record = _evidence(
                evidence,
                "psutil",
                key,
                "system_resource_telemetry",
                caveat="Host measurements are sampled after each local token decision.",
            )
            metrics.append(
                _metric(
                    name=key,
                    value=value,
                    unit=unit,
                    status="real",
                    evidence=record,
                    label=key.replace("_", " ").title(),
                    explanation="Measured local host or Cogito process resource value.",
                )
            )
        return metrics, TelemetryEvent(
            event_type="resource.sample",
            session_id=session_id,
            run_id=run_id,
            status="real",
            source_name="psutil",
            capability="system_resource_telemetry",
            payload={
                "token_id": step.position,
                "sample": step.resource_sample,
            },
            token_id=step.position,
            evidence_ids=[metric.evidence_id for metric in metrics],
        )


def _evidence(
    target: list[EvidenceRecord],
    source_name: str,
    source_field: str,
    capability: str,
    *,
    source_kind: str = "provider",
    formula: str | None = None,
    caveat: str | None = None,
) -> EvidenceRecord:
    record = make_evidence(
        source_kind,
        source_name,
        source_field,
        formula=formula,
        caveat=caveat,
        capability=capability,
    )
    target.append(record)
    return record


def _metric(
    *,
    name: str,
    value: int | float | str | bool | None,
    unit: str | None,
    status: str,
    evidence: EvidenceRecord,
    label: str,
    explanation: str,
) -> MetricValue:
    return MetricValue(
        name=name,
        value=value,
        unit=unit,
        status=status,
        evidence_id=evidence.evidence_id,
        display_label=label,
        plain_explanation=explanation,
        expert_explanation=evidence.caveat,
        source_name=evidence.source_name,
        capability=evidence.capability or "",
        formula=evidence.formula,
        caveat=evidence.caveat,
    )


def _fingerprint(
    prompt_text: str,
    identity: ModelIdentity,
    settings: GenerationSettings,
) -> str:
    payload = {
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "model": identity.to_dict(),
        "generation": settings.to_dict(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]


def _next_or_none(iterator: Iterator[ReferenceStep]) -> ReferenceStep | None:
    try:
        return next(iterator)
    except StopIteration:
        return None
