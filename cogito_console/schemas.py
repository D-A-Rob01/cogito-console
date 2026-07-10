from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SourceKind = Literal["provider", "runtime", "derived", "synthetic", "operator", "system"]
MetricStatus = Literal["real", "derived", "synthetic"]
GraphStatus = Literal["real", "derived", "synthetic", "operator", "system"]
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
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id(),
        source_kind=source_kind,
        source_name=source_name,
        source_field=source_field,
        formula=formula,
        caveat=caveat,
    )


def metric_dict(metric: MetricValue | None) -> dict[str, Any] | None:
    return metric.to_dict() if metric else None
