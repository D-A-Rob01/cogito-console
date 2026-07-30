from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SourceKind = Literal["provider", "runtime", "derived", "synthetic", "operator", "system"]
MetricStatus = Literal["real", "derived", "synthetic", "unavailable"]
GraphStatus = Literal["real", "derived", "synthetic", "unavailable", "operator", "system"]
CapabilitySupport = Literal["supported", "conditional", "unsupported"]
TelemetryEventType = Literal[
    "run.started",
    "provider.capabilities",
    "prompt.encoded",
    "prefill.started",
    "prefill.completed",
    "token.sampled",
    "token.committed",
    "layer.summary",
    "attention.summary",
    "router.decision",
    "expert.loaded",
    "expert.cache_hit",
    "kv.updated",
    "resource.sample",
    "steer.requested",
    "run.cancelled",
    "run.resumed",
    "run.completed",
    "measurement.unavailable",
]
TELEMETRY_SCHEMA_VERSION = "1.0"
CAPABILITY_NAMES = (
    "streaming",
    "token_ids",
    "selected_token_logprobs",
    "top_k_alternatives",
    "raw_logits",
    "hidden_states",
    "attentions",
    "moe_router_logits",
    "selected_experts",
    "routing_weights",
    "kv_cache_metrics",
    "request_queue_metrics",
    "system_resource_telemetry",
    "deterministic_seed",
    "structured_output",
    "branch_replay",
    "sampling_intervention",
    "logit_intervention",
    "activation_intervention",
)
NodeType = Literal[
    "prompt",
    "router",
    "token",
    "alternative",
    "rewrite",
    "context",
    "provider",
    "system",
]
EdgeType = Literal[
    "generated_by",
    "alternative_to",
    "selected_by_operator",
    "caused_rewrite",
    "derived_from",
    "constrained_by",
    "routed_by",
    "held_in_context",
]
MetricPrimitive = int | float | str | bool | None


def evidence_id(prefix: str = "ev") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass(slots=True)
class EvidenceRecord:
    evidence_id: str
    source_kind: SourceKind
    source_name: str
    source_field: str | None = None
    formula: str | None = None
    caveat: str | None = None
    capability: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MetricValue:
    name: str
    value: MetricPrimitive
    unit: str | None
    status: MetricStatus
    evidence_id: str
    display_label: str
    plain_explanation: str
    expert_explanation: str | None = None
    collected_at: float = field(default_factory=time.time)
    source_name: str = ""
    capability: str = ""
    formula: str | None = None
    caveat: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProviderCapability:
    name: str
    support: CapabilitySupport
    granularity: str
    limitation: str | None = None
    status_when_present: MetricStatus | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProviderCapabilityRecord:
    provider_id: str
    provider_name: str
    mode: str
    capabilities: dict[str, ProviderCapability]
    steering_methods: list[str] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        missing = set(CAPABILITY_NAMES) - set(self.capabilities)
        if missing:
            raise ValueError(f"capability record is missing: {', '.join(sorted(missing))}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "mode": self.mode,
            "generated_at": self.generated_at,
            "steering_methods": list(self.steering_methods),
            "capabilities": {
                name: capability.to_dict()
                for name, capability in self.capabilities.items()
            },
        }


@dataclass(slots=True)
class TelemetryEvent:
    event_type: TelemetryEventType
    session_id: str
    run_id: str
    status: MetricStatus
    source_name: str
    capability: str
    payload: dict[str, Any]
    sequence: int = 0
    token_id: int | None = None
    layer_index: int | None = None
    evidence_ids: list[str] = field(default_factory=list)
    event_id: str = field(default_factory=lambda: evidence_id("evt"))
    collected_at: float = field(default_factory=time.time)
    schema_version: str = TELEMETRY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AlternativeTokenEvent:
    token: str
    rank: int
    logprob: MetricValue | None = None
    probability: MetricValue | None = None
    margin_from_selected: MetricValue | None = None
    rationale: str | None = None
    status: MetricStatus = "synthetic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TokenEvent:
    session_id: str
    run_id: str
    token_id: int
    text: str
    logprob: MetricValue | None = None
    probability: MetricValue | None = None
    byte_length: MetricValue | None = None
    latency_ms: MetricValue | None = None
    alternatives: list[AlternativeTokenEvent] = field(default_factory=list)
    raw_provider_payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RouterValidationEvent:
    name: str
    verdict: str
    confidence: MetricValue
    signals: list[str]
    status: MetricStatus

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RewriteEvent:
    session_id: str
    old_run_id: str
    new_run_id: str
    token_id: int
    chosen_token: str
    previous_text: str
    rewritten_text: str
    operator_action_timestamp: float
    reason: str | None = None
    rationale: str | None = None
    intervention_type: str = "narrative_ui_splice"
    provider_id: str | None = None
    capability: str = "branch_replay"
    status: MetricStatus = "synthetic"
    reused_prefix_tokens: int = 0
    invalidated_token_count: int = 0
    recomputed_from_token: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GraphNode:
    id: str
    label: str
    node_type: NodeType
    status: GraphStatus
    metrics: list[MetricValue] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    plain_explanation: str = ""
    expert_explanation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GraphEdge:
    source: str
    target: str
    edge_type: EdgeType
    weight: MetricValue | None = None
    evidence_ids: list[str] = field(default_factory=list)
    plain_explanation: str = ""
    expert_explanation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_evidence(
    source_kind: SourceKind,
    source_name: str,
    source_field: str | None = None,
    formula: str | None = None,
    caveat: str | None = None,
    capability: str | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id(),
        source_kind=source_kind,
        source_name=source_name,
        source_field=source_field,
        formula=formula,
        caveat=caveat,
        capability=capability,
    )


def make_unavailable_metric(
    *,
    name: str,
    display_label: str,
    evidence: EvidenceRecord,
    capability: str,
    explanation: str,
) -> MetricValue:
    return MetricValue(
        name=name,
        value=None,
        unit=None,
        status="unavailable",
        evidence_id=evidence.evidence_id,
        display_label=display_label,
        plain_explanation=explanation,
        expert_explanation=evidence.caveat,
        source_name=evidence.source_name,
        capability=capability,
        caveat=evidence.caveat,
    )


def metric_dict(metric: MetricValue | None) -> dict[str, Any] | None:
    return metric.to_dict() if metric else None
