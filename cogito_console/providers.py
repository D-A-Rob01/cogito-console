from __future__ import annotations

import os
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from .interceptor import LatentBeamInterceptor
from .metrics import MetricEngine
from .schemas import (
    CAPABILITY_NAMES,
    AlternativeTokenEvent,
    EvidenceRecord,
    MetricValue,
    ProviderCapability,
    ProviderCapabilityRecord,
    TelemetryEvent,
    TokenEvent,
    make_evidence,
    metric_dict,
)


ProviderMode = Literal[
    "demo",
    "openai",
    "transformers",
    "llama_cpp",
    "moe",
    "colibri",
    "mock",
]


class ProviderUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class ProviderStreamPacket:
    token_event: TokenEvent
    evidence: list[EvidenceRecord] = field(default_factory=list)
    metrics: list[MetricValue] = field(default_factory=list)
    hidden_state_metrics: dict[str, MetricValue] = field(default_factory=dict)
    telemetry_events: list[TelemetryEvent] = field(default_factory=list)

    def to_legacy_packet(self, provider_name: str | None = None) -> dict[str, Any]:
        source_name = provider_name or str(
            (self.token_event.raw_provider_payload or {}).get("source", "provider_adapter")
        )
        self.ensure_telemetry_events(source_name)
        raw_payload = dict(self.token_event.raw_provider_payload or {})
        packet = {
            "source": raw_payload.get("source", "provider_adapter"),
            "run_id": self.token_event.run_id,
            "token_id": self.token_event.token_id,
            "token": self.token_event.text,
            "logprob": self.token_event.logprob.value if self.token_event.logprob else None,
            "probability": self.token_event.probability.value if self.token_event.probability else None,
            "alternatives": [self._alternative_to_legacy(alt) for alt in self.token_event.alternatives],
            "hidden_state": raw_payload.get("hidden_state", {}),
            "hidden_state_metrics": {
                key: metric.to_dict() for key, metric in self.hidden_state_metrics.items()
            },
            "text": raw_payload.get("text", self.token_event.text),
            "committed_history": raw_payload.get("committed_history", []),
            "token_event": self.token_event.to_dict(),
            "evidence": [record.to_dict() for record in self.evidence],
            "metrics": {metric.name: metric.to_dict() for metric in self.metrics},
            "telemetry_events": [event.to_dict() for event in self.telemetry_events],
            "raw_provider_payload": raw_payload,
        }
        return packet

    def ensure_telemetry_events(self, provider_name: str) -> list[TelemetryEvent]:
        evidence_by_id = {record.evidence_id: record for record in self.evidence}
        for metric in self._all_metrics():
            evidence = evidence_by_id.get(metric.evidence_id)
            if evidence is not None:
                metric.source_name = metric.source_name or evidence.source_name
                metric.capability = metric.capability or evidence.capability or _metric_capability(metric.name)
                metric.formula = metric.formula or evidence.formula
                metric.caveat = metric.caveat or evidence.caveat
            else:
                metric.source_name = metric.source_name or provider_name
                metric.capability = metric.capability or _metric_capability(metric.name)

        if self.telemetry_events:
            return self.telemetry_events

        token_event = self.token_event
        token_status = token_event.logprob.status if token_event.logprob else "real"
        evidence_ids = [metric.evidence_id for metric in self._all_metrics()]
        compact_payload = {
            "token_id": token_event.token_id,
            "text": token_event.text,
            "logprob": metric_dict(token_event.logprob),
            "probability": metric_dict(token_event.probability),
            "latency_ms": metric_dict(token_event.latency_ms),
            "alternative_count": len(token_event.alternatives),
        }
        self.telemetry_events.extend(
            [
                TelemetryEvent(
                    event_type="token.sampled",
                    session_id=token_event.session_id,
                    run_id=token_event.run_id,
                    status=token_status,
                    source_name=provider_name,
                    capability="selected_token_logprobs",
                    payload=compact_payload,
                    token_id=token_event.token_id,
                    evidence_ids=evidence_ids,
                ),
                TelemetryEvent(
                    event_type="token.committed",
                    session_id=token_event.session_id,
                    run_id=token_event.run_id,
                    status=token_status,
                    source_name="Cogito Console transaction log",
                    capability="streaming",
                    payload={"token_id": token_event.token_id, "text": token_event.text},
                    token_id=token_event.token_id,
                    evidence_ids=evidence_ids,
                ),
            ]
        )
        for metric in self._all_metrics():
            if metric.status == "unavailable":
                self.telemetry_events.append(
                    TelemetryEvent(
                        event_type="measurement.unavailable",
                        session_id=token_event.session_id,
                        run_id=token_event.run_id,
                        status="unavailable",
                        source_name=metric.source_name or provider_name,
                        capability=metric.capability,
                        payload={"metric": metric.to_dict()},
                        token_id=token_event.token_id,
                        evidence_ids=[metric.evidence_id],
                    )
                )
        return self.telemetry_events

    def _all_metrics(self) -> list[MetricValue]:
        metrics = list(self.metrics)
        metrics.extend(self.hidden_state_metrics.values())
        for alternative in self.token_event.alternatives:
            metrics.extend(
                metric
                for metric in (
                    alternative.logprob,
                    alternative.probability,
                    alternative.margin_from_selected,
                )
                if metric is not None
            )
        unique: dict[str, MetricValue] = {}
        for metric in metrics:
            unique[f"{metric.name}:{metric.evidence_id}"] = metric
        return list(unique.values())

    @staticmethod
    def _alternative_to_legacy(alternative: AlternativeTokenEvent) -> dict[str, Any]:
        return {
            "token": alternative.token,
            "rank": alternative.rank,
            "logprob": alternative.logprob.value if alternative.logprob else None,
            "probability": alternative.probability.value if alternative.probability else None,
            "margin_from_selected": (
                alternative.margin_from_selected.value if alternative.margin_from_selected else None
            ),
            "rationale": alternative.rationale,
            "status": alternative.status,
            "metrics": {
                "logprob": metric_dict(alternative.logprob),
                "probability": metric_dict(alternative.probability),
                "margin_from_selected": metric_dict(alternative.margin_from_selected),
            },
        }


class ProviderAdapter(ABC):
    name: str
    mode: ProviderMode

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system_context: str,
        history: list[str],
        *,
        session_id: str,
        run_id: str,
    ) -> AsyncIterator[ProviderStreamPacket]:
        ...

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str | None:
        return None

    def capability_record(self) -> ProviderCapabilityRecord:
        return ProviderCapabilityRecord(
            provider_id=self.mode,
            provider_name=self.name,
            mode=self.mode,
            capabilities=_capabilities(),
        )

    def describe(self) -> dict[str, Any]:
        capability_record = self.capability_record()
        return {
            "provider_id": capability_record.provider_id,
            "name": self.name,
            "mode": self.mode,
            "available": self.is_available(),
            "unavailable_reason": self.unavailable_reason(),
            "capability_record": capability_record.to_dict(),
        }


class LocalDemoAdapter(ProviderAdapter):
    name = "Local deterministic demo"
    mode: ProviderMode = "demo"

    def __init__(
        self,
        interceptor: LatentBeamInterceptor | None = None,
        *,
        enable_synthetic_probes: bool = True,
    ) -> None:
        self.interceptor = interceptor or LatentBeamInterceptor()
        self.enable_synthetic_probes = enable_synthetic_probes

    def capability_record(self) -> ProviderCapabilityRecord:
        return ProviderCapabilityRecord(
            provider_id="demo",
            provider_name=self.name,
            mode=self.mode,
            capabilities=_capabilities(
                streaming=_cap("streaming", "supported", "token", status="synthetic"),
                token_ids=_cap(
                    "token_ids",
                    "conditional",
                    "whitespace-delimited demo unit",
                    "IDs describe demo units, not a model tokenizer.",
                    "synthetic",
                ),
                system_resource_telemetry=_cap(
                    "system_resource_telemetry", "supported", "server process", status="real"
                ),
                deterministic_seed=_cap(
                    "deterministic_seed",
                    "supported",
                    "entire demo run",
                    "The deterministic demo has no stochastic sampler.",
                    "synthetic",
                ),
            ),
            steering_methods=["narrative_ui_splice"],
        )

    async def stream(
        self,
        prompt: str,
        system_context: str,
        history: list[str],
        *,
        session_id: str,
        run_id: str,
    ) -> AsyncIterator[ProviderStreamPacket]:
        source = self.interceptor.stream_with_steering_hooks(
            prompt=prompt,
            system_context=system_context,
            committed_history=history,
            run_id=run_id,
        )
        last_seen = time.perf_counter()
        async for raw_packet in source:
            now = time.perf_counter()
            latency_ms = (now - last_seen) * 1000
            last_seen = now
            yield self._packet_from_demo_raw(raw_packet, session_id, latency_ms)

    def _packet_from_demo_raw(
        self,
        raw_packet: dict[str, Any],
        session_id: str,
        latency_ms: float,
    ) -> ProviderStreamPacket:
        evidence: list[EvidenceRecord] = []
        metrics: list[MetricValue] = []

        logprob_evidence = make_evidence(
            "synthetic",
            self.name,
            "packet.logprob",
            caveat="Deterministic demo score from LatentBeamInterceptor; not a provider logprob.",
        )
        evidence.append(logprob_evidence)
        logprob = MetricValue(
            name="token_logprob",
            value=raw_packet.get("logprob"),
            unit="logprob",
            status="synthetic",
            evidence_id=logprob_evidence.evidence_id,
            display_label="Logprob",
            plain_explanation="Demo-only score for the selected token.",
            expert_explanation="Generated by LatentBeamInterceptor._logprob for deterministic local visualization.",
        )
        metrics.append(logprob)

        probability_evidence = make_evidence(
            "synthetic",
            self.name,
            "packet.logprob",
            formula="probability = exp(logprob)",
            caveat="Computed from a synthetic demo logprob, so the probability is also synthetic.",
        )
        evidence.append(probability_evidence)
        probability = MetricEngine.token_probability(logprob, probability_evidence.evidence_id)
        metrics.append(probability)

        byte_evidence = make_evidence(
            "runtime",
            "Python runtime",
            "len(token.encode('utf-8'))",
            caveat="Measured from the token text held by the server process.",
        )
        evidence.append(byte_evidence)
        byte_length = MetricValue(
            name="byte_length",
            value=len(str(raw_packet.get("token", "")).encode("utf-8")),
            unit="bytes",
            status="real",
            evidence_id=byte_evidence.evidence_id,
            display_label="Byte length",
            plain_explanation="How large this token text is in UTF-8 bytes.",
            expert_explanation="Computed by the Python runtime from the emitted token string.",
        )
        metrics.append(byte_length)

        latency_evidence = make_evidence(
            "runtime",
            "Python runtime",
            "time.perf_counter",
            caveat="Local runtime timing; includes demo generator delay.",
        )
        evidence.append(latency_evidence)
        latency_metric = MetricEngine.token_latency_ms(latency_ms, latency_evidence.evidence_id)
        metrics.append(latency_metric)

        alternatives = [
            self._demo_alternative(alt, logprob, evidence, metrics)
            for alt in raw_packet.get("alternatives", [])
        ]
        hidden_state_metrics = self._hidden_state_metrics(raw_packet, evidence)
        metrics.extend(hidden_state_metrics.values())

        token_event = TokenEvent(
            session_id=session_id,
            run_id=str(raw_packet["run_id"]),
            token_id=int(raw_packet["token_id"]),
            text=str(raw_packet["token"]),
            logprob=logprob,
            probability=probability,
            byte_length=byte_length,
            latency_ms=latency_metric,
            alternatives=alternatives,
            raw_provider_payload=raw_packet,
        )
        return ProviderStreamPacket(
            token_event=token_event,
            evidence=evidence,
            metrics=metrics,
            hidden_state_metrics=hidden_state_metrics,
        )

    def _demo_alternative(
        self,
        alternative: dict[str, Any],
        selected_logprob: MetricValue,
        evidence: list[EvidenceRecord],
        metrics: list[MetricValue],
    ) -> AlternativeTokenEvent:
        logprob_evidence = make_evidence(
            "synthetic",
            self.name,
            "packet.alternatives[].logprob",
            caveat="Deterministic demo alternative; not returned by a provider.",
        )
        evidence.append(logprob_evidence)
        logprob = MetricValue(
            name=f"alternative_{alternative.get('rank', 0)}_logprob",
            value=alternative.get("logprob"),
            unit="logprob",
            status="synthetic",
            evidence_id=logprob_evidence.evidence_id,
            display_label="Alternative logprob",
            plain_explanation="Demo-only score for this alternate token.",
            expert_explanation="Generated locally to make steering mechanics inspectable without provider credentials.",
        )
        metrics.append(logprob)

        probability_evidence = make_evidence(
            "synthetic",
            self.name,
            "packet.alternatives[].logprob",
            formula="probability = exp(logprob)",
            caveat="Computed from a synthetic alternative logprob.",
        )
        evidence.append(probability_evidence)
        probability = MetricEngine.token_probability(logprob, probability_evidence.evidence_id)
        metrics.append(probability)

        margin_evidence = make_evidence(
            "synthetic",
            self.name,
            "packet.logprob - packet.alternatives[].logprob",
            formula="selected_logprob - alternative_logprob",
            caveat="Computed from synthetic demo logprobs.",
        )
        evidence.append(margin_evidence)
        margin = MetricEngine.alternative_margin(selected_logprob, logprob, margin_evidence.evidence_id)
        metrics.append(margin)

        return AlternativeTokenEvent(
            token=str(alternative.get("token", "")),
            rank=int(alternative.get("rank", 0)),
            logprob=logprob,
            probability=probability,
            margin_from_selected=margin,
            rationale=alternative.get("rationale"),
            status="synthetic",
        )

    def _hidden_state_metrics(
        self,
        raw_packet: dict[str, Any],
        evidence: list[EvidenceRecord],
    ) -> dict[str, MetricValue]:
        if not self.enable_synthetic_probes:
            return {}
        result: dict[str, MetricValue] = {}
        for name, value in (raw_packet.get("hidden_state") or {}).items():
            probe_evidence = make_evidence(
                "synthetic",
                self.name,
                f"hidden_state.{name}",
                caveat="Synthetic probe generated by the local demo; not a model hidden state.",
            )
            evidence.append(probe_evidence)
            result[name] = MetricValue(
                name=f"synthetic_probe_{name}",
                value=value,
                unit="score",
                status="synthetic",
                evidence_id=probe_evidence.evidence_id,
                display_label=name.replace("_", " ").title(),
                plain_explanation="A demo signal used to make the graph easier to inspect.",
                expert_explanation="This value is not returned by a provider and must not be treated as hidden-state access.",
            )
        return result


class OpenAIAdapter(ProviderAdapter):
    name = "OpenAI"
    mode: ProviderMode = "openai"

    def __init__(self, *, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or os.getenv("COGITO_MODEL") or "gpt-4.1-mini"
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def unavailable_reason(self) -> str | None:
        if not self.api_key:
            return "OpenAI adapter unavailable: missing OPENAI_API_KEY."
        return None

    def capability_record(self) -> ProviderCapabilityRecord:
        return ProviderCapabilityRecord(
            provider_id="openai",
            provider_name=self.name,
            mode=self.mode,
            capabilities=_capabilities(
                streaming=_cap("streaming", "supported", "streamed chunk", status="real"),
                selected_token_logprobs=_cap(
                    "selected_token_logprobs",
                    "conditional",
                    "provider token",
                    "Availability depends on the selected model and endpoint.",
                    "real",
                ),
                top_k_alternatives=_cap(
                    "top_k_alternatives",
                    "conditional",
                    "provider token",
                    "Availability depends on the selected model and endpoint.",
                    "real",
                ),
                request_queue_metrics=_cap(
                    "request_queue_metrics",
                    "conditional",
                    "request",
                    "Only values explicitly returned by the API are retained.",
                    "real",
                ),
                system_resource_telemetry=_cap(
                    "system_resource_telemetry",
                    "supported",
                    "Cogito server process",
                    "Does not describe OpenAI infrastructure.",
                    "real",
                ),
                deterministic_seed=_cap(
                    "deterministic_seed",
                    "conditional",
                    "request",
                    "Only exposed when supported by the selected API surface.",
                    "real",
                ),
                structured_output=_cap(
                    "structured_output",
                    "conditional",
                    "request",
                    "Not wired into the current Cogito generation UI.",
                    "real",
                ),
            ),
            steering_methods=[],
        )

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload["model"] = self.model
        payload["logprob_policy"] = "Provider logprobs are used only when returned; missing alternatives are not synthesized."
        return payload

    async def stream(
        self,
        prompt: str,
        system_context: str,
        history: list[str],
        *,
        session_id: str,
        run_id: str,
    ) -> AsyncIterator[ProviderStreamPacket]:
        if not self.api_key:
            raise ProviderUnavailableError("OpenAI adapter unavailable: missing OPENAI_API_KEY.")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ProviderUnavailableError(
                "OpenAI adapter unavailable: install the official openai Python SDK."
            ) from exc

        client = AsyncOpenAI(api_key=self.api_key)
        messages = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": prompt},
        ]
        if history:
            messages.append({"role": "assistant", "content": "".join(history)})

        last_seen = time.perf_counter()
        token_id = len(history)
        stream = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            logprobs=True,
            top_logprobs=5,
        )
        async for chunk in stream:
            raw_payload = _model_dump(chunk)
            choice = (getattr(chunk, "choices", None) or [None])[0]
            delta = getattr(choice, "delta", None)
            token = getattr(delta, "content", None)
            if not token:
                continue
            now = time.perf_counter()
            latency_ms = (now - last_seen) * 1000
            last_seen = now
            yield self._packet_from_openai_chunk(
                raw_payload=raw_payload,
                token=token,
                token_id=token_id,
                run_id=run_id,
                session_id=session_id,
                latency_ms=latency_ms,
                choice=choice,
            )
            token_id += 1

    def _packet_from_openai_chunk(
        self,
        *,
        raw_payload: dict[str, Any],
        token: str,
        token_id: int,
        run_id: str,
        session_id: str,
        latency_ms: float,
        choice: Any,
    ) -> ProviderStreamPacket:
        evidence: list[EvidenceRecord] = []
        metrics: list[MetricValue] = []
        logprob_value, top_logprobs = _extract_openai_logprobs(choice)

        logprob = None
        probability = None
        if logprob_value is not None:
            logprob_evidence = make_evidence(
                "provider",
                self.name,
                "choices[].logprobs.content[].logprob",
                caveat="Returned by the OpenAI API for this streamed chunk.",
            )
            evidence.append(logprob_evidence)
            logprob = MetricValue(
                name="token_logprob",
                value=logprob_value,
                unit="logprob",
                status="real",
                evidence_id=logprob_evidence.evidence_id,
                display_label="Logprob",
                plain_explanation="Provider-returned score for this token.",
                expert_explanation="Directly returned by OpenAI when the selected model and endpoint expose logprobs.",
            )
            metrics.append(logprob)

            probability_evidence = make_evidence(
                "derived",
                self.name,
                "choices[].logprobs.content[].logprob",
                formula="probability = exp(logprob)",
                caveat="Derived from provider logprob; provider probability may differ after top-k normalization.",
            )
            evidence.append(probability_evidence)
            probability = MetricEngine.token_probability(logprob, probability_evidence.evidence_id)
            metrics.append(probability)

        byte_evidence = make_evidence("runtime", "Python runtime", "len(token.encode('utf-8'))")
        evidence.append(byte_evidence)
        byte_length = MetricValue(
            name="byte_length",
            value=len(token.encode("utf-8")),
            unit="bytes",
            status="real",
            evidence_id=byte_evidence.evidence_id,
            display_label="Byte length",
            plain_explanation="How large this token text is in UTF-8 bytes.",
            expert_explanation="Computed by the Python runtime from the provider token string.",
        )
        metrics.append(byte_length)

        latency_evidence = make_evidence("runtime", "Python runtime", "time.perf_counter")
        evidence.append(latency_evidence)
        latency = MetricEngine.token_latency_ms(latency_ms, latency_evidence.evidence_id)
        metrics.append(latency)

        alternatives = self._openai_alternatives(top_logprobs, logprob, evidence, metrics)
        token_event = TokenEvent(
            session_id=session_id,
            run_id=run_id,
            token_id=token_id,
            text=token,
            logprob=logprob,
            probability=probability,
            byte_length=byte_length,
            latency_ms=latency,
            alternatives=alternatives,
            raw_provider_payload={
                "source": "openai_chat_completions",
                "run_id": run_id,
                "token_id": token_id,
                "token": token,
                "text": token,
                "committed_history": [],
                "raw_chunk": raw_payload,
            },
        )
        return ProviderStreamPacket(token_event=token_event, evidence=evidence, metrics=metrics)

    def _openai_alternatives(
        self,
        top_logprobs: list[dict[str, Any]],
        selected_logprob: MetricValue | None,
        evidence: list[EvidenceRecord],
        metrics: list[MetricValue],
    ) -> list[AlternativeTokenEvent]:
        alternatives: list[AlternativeTokenEvent] = []
        for index, item in enumerate(top_logprobs[:5], start=1):
            token = str(item.get("token", ""))
            value = item.get("logprob")
            if value is None or (selected_logprob and token == selected_logprob.value):
                continue
            logprob_evidence = make_evidence(
                "provider",
                self.name,
                "choices[].logprobs.content[].top_logprobs[]",
                caveat="Returned by the OpenAI API when available for this model and endpoint.",
            )
            evidence.append(logprob_evidence)
            logprob = MetricValue(
                name=f"alternative_{index}_logprob",
                value=value,
                unit="logprob",
                status="real",
                evidence_id=logprob_evidence.evidence_id,
                display_label="Alternative logprob",
                plain_explanation="Provider-returned score for this alternate token.",
                expert_explanation="No alternatives are synthesized in OpenAI mode unless a separate demo overlay is explicitly enabled.",
            )
            metrics.append(logprob)

            probability_evidence = make_evidence(
                "derived",
                self.name,
                "choices[].logprobs.content[].top_logprobs[]",
                formula="probability = exp(logprob)",
            )
            evidence.append(probability_evidence)
            probability = MetricEngine.token_probability(logprob, probability_evidence.evidence_id)
            metrics.append(probability)

            margin = None
            if selected_logprob:
                margin_evidence = make_evidence(
                    "derived",
                    self.name,
                    "selected_logprob - alternative_logprob",
                    formula="selected_logprob - alternative_logprob",
                )
                evidence.append(margin_evidence)
                margin = MetricEngine.alternative_margin(selected_logprob, logprob, margin_evidence.evidence_id)
                metrics.append(margin)

            alternatives.append(
                AlternativeTokenEvent(
                    token=token,
                    rank=index,
                    logprob=logprob,
                    probability=probability,
                    margin_from_selected=margin,
                    rationale="Provider-returned alternate token.",
                    status="real",
                )
            )
        return alternatives


def adapter_from_env(interceptor: LatentBeamInterceptor | None = None) -> ProviderAdapter:
    provider = os.getenv("COGITO_PROVIDER", "demo").strip().lower()
    if provider == "openai":
        return OpenAIAdapter()
    enable_synthetic_probes = os.getenv("COGITO_ENABLE_SYNTHETIC_PROBES", "true").lower() != "false"
    return LocalDemoAdapter(interceptor=interceptor, enable_synthetic_probes=enable_synthetic_probes)


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {"repr": repr(value)}


def _extract_openai_logprobs(choice: Any) -> tuple[float | None, list[dict[str, Any]]]:
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None)
    if not content:
        return None, []
    item = content[0]
    logprob = getattr(item, "logprob", None)
    top = []
    for top_item in getattr(item, "top_logprobs", None) or []:
        top.append(
            {
                "token": getattr(top_item, "token", None),
                "logprob": getattr(top_item, "logprob", None),
            }
        )
    return logprob, top


def _cap(
    name: str,
    support: Literal["supported", "conditional", "unsupported"],
    granularity: str,
    limitation: str | None = None,
    status: Literal["real", "derived", "synthetic", "unavailable"] | None = None,
) -> ProviderCapability:
    return ProviderCapability(
        name=name,
        support=support,
        granularity=granularity,
        limitation=limitation,
        status_when_present=status,
    )


def _capabilities(**overrides: ProviderCapability) -> dict[str, ProviderCapability]:
    capabilities = {
        name: _cap(
            name,
            "unsupported",
            "unavailable",
            "The active backend does not expose this capability.",
            "unavailable",
        )
        for name in CAPABILITY_NAMES
    }
    capabilities.update(overrides)
    return capabilities


def _metric_capability(metric_name: str) -> str:
    if "logprob" in metric_name:
        return "selected_token_logprobs"
    if metric_name.startswith("alternative_"):
        return "top_k_alternatives"
    if "probability" in metric_name or "entropy" in metric_name or "margin" in metric_name:
        return "raw_logits"
    if "hidden" in metric_name or "layer" in metric_name:
        return "hidden_states"
    if "latency" in metric_name or "speed" in metric_name or "duration" in metric_name:
        return "system_resource_telemetry"
    return "streaming"
