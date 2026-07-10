from __future__ import annotations

from typing import Any

from .schemas import (
    EvidenceRecord,
    GraphEdge,
    GraphNode,
    MetricValue,
    RouterValidationEvent,
    TokenEvent,
    make_evidence,
)


def router_validation_events(
    sub_context: dict[str, dict[str, Any]],
) -> tuple[list[RouterValidationEvent], list[EvidenceRecord]]:
    events: list[RouterValidationEvent] = []
    evidence: list[EvidenceRecord] = []
    for key, item in sub_context.items():
        record = make_evidence(
            "synthetic",
            "ConfluenceRouter",
            f"sub_context.{key}.confidence",
            caveat="Local heuristic confidence from prompt micro-validation; not provider telemetry.",
        )
        evidence.append(record)
        confidence = MetricValue(
            name=f"{key}_confidence",
            value=item.get("confidence"),
            unit="score",
            status="synthetic",
            evidence_id=record.evidence_id,
            display_label="Check confidence",
            plain_explanation="How strongly this prompt check matched its local heuristic.",
            expert_explanation="Computed by the local ConfluenceRouter micro-validation routine.",
        )
        events.append(
            RouterValidationEvent(
                name=item.get("name", key),
                verdict=item.get("verdict", ""),
                confidence=confidence,
                signals=list(item.get("signals") or []),
                status="synthetic",
            )
        )
    return events, evidence


def build_router_graph(
    *,
    session_id: str,
    prompt: str,
    router_events: list[RouterValidationEvent],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    prompt_node = GraphNode(
        id=f"prompt:{session_id}",
        label="Prompt",
        node_type="prompt",
        status="operator",
        plain_explanation="The text entered by the operator.",
        expert_explanation=f"Prompt length: {len(prompt)} characters.",
    )
    nodes = [prompt_node]
    edges: list[GraphEdge] = []
    for event in router_events:
        node_id = f"router:{session_id}:{event.name.lower().replace(' ', '_')}"
        nodes.append(
            GraphNode(
                id=node_id,
                label=event.name,
                node_type="router",
                status=event.status,
                metrics=[event.confidence],
                evidence_ids=[event.confidence.evidence_id],
                plain_explanation=event.verdict,
                expert_explanation=f"Signals: {', '.join(event.signals) if event.signals else 'none'}",
            )
        )
        edges.append(
            GraphEdge(
                source=prompt_node.id,
                target=node_id,
                edge_type="routed_by",
                weight=event.confidence,
                evidence_ids=[event.confidence.evidence_id],
                plain_explanation="This prompt check was run before generation.",
                expert_explanation="Router validations are local demo heuristics unless replaced by a real validator.",
            )
        )
    return nodes, edges


def build_context_graph(session_id: str, mesh: dict[str, Any]) -> tuple[list[GraphNode], list[GraphEdge]]:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    for item in mesh.get("nodes") or []:
        nodes.append(
            GraphNode(
                id=f"context:{session_id}:{item['id']}",
                label=item.get("label", item["id"]),
                node_type="context",
                status="system",
                plain_explanation="A temporary context mesh node held only for this session.",
                expert_explanation="The node exists in runtime memory and dissolves when the session is cleared or completes.",
            )
        )
    for item in mesh.get("edges") or []:
        edges.append(
            GraphEdge(
                source=f"context:{session_id}:{item['source']}",
                target=f"context:{session_id}:{item['target']}",
                edge_type="held_in_context",
                evidence_ids=[],
                plain_explanation=item.get("label", "Temporary context relationship."),
                expert_explanation="Context mesh weights are local orchestration hints, not provider telemetry.",
            )
        )
    return nodes, edges


def build_token_graph(token_event: TokenEvent, provider_name: str) -> tuple[list[GraphNode], list[GraphEdge]]:
    provider_id = f"provider:{token_event.session_id}:{provider_name.lower().replace(' ', '_')}"
    token_id = f"token:{token_event.run_id}:{token_event.token_id}"
    token_status = token_event.logprob.status if token_event.logprob else "real"
    token_metrics = [
        metric
        for metric in [
            token_event.probability,
            token_event.logprob,
            token_event.latency_ms,
            token_event.byte_length,
        ]
        if metric is not None
    ]
    nodes = [
        GraphNode(
            id=provider_id,
            label=provider_name,
            node_type="provider",
            status="system",
            plain_explanation="The adapter that produced the token event.",
            expert_explanation="Provider adapter status is reported separately by /api/providers.",
        ),
        GraphNode(
            id=token_id,
            label=token_event.text.strip() or "space",
            node_type="token",
            status=token_status,
            metrics=token_metrics,
            evidence_ids=[metric.evidence_id for metric in token_metrics],
            plain_explanation="A token on the accepted generation path.",
            expert_explanation="Token metrics are tagged independently; in demo mode token probabilities are synthetic.",
        ),
    ]
    edges = [
        GraphEdge(
            source=provider_id,
            target=token_id,
            edge_type="generated_by",
            weight=token_event.probability,
            evidence_ids=[token_event.probability.evidence_id] if token_event.probability else [],
            plain_explanation="The provider adapter emitted this token.",
            expert_explanation="A missing weight means the provider did not return a logprob for this token.",
        )
    ]
    for alternative in token_event.alternatives:
        alt_id = f"alternative:{token_event.run_id}:{token_event.token_id}:{alternative.rank}"
        alt_metrics = [
            metric
            for metric in [alternative.probability, alternative.logprob, alternative.margin_from_selected]
            if metric is not None
        ]
        nodes.append(
            GraphNode(
                id=alt_id,
                label=alternative.token.strip() or "space",
                node_type="alternative",
                status=alternative.status,
                metrics=alt_metrics,
                evidence_ids=[metric.evidence_id for metric in alt_metrics],
                plain_explanation=alternative.rationale or "An alternate token path.",
                expert_explanation="Clicking this node sends a steer event and rewrites history at the source token.",
            )
        )
        edges.append(
            GraphEdge(
                source=token_id,
                target=alt_id,
                edge_type="alternative_to",
                weight=alternative.probability,
                evidence_ids=[alternative.probability.evidence_id] if alternative.probability else [],
                plain_explanation="This branch was available as an alternative continuation.",
                expert_explanation="Dashed alternative edges should not be interpreted as accepted output.",
            )
        )
    return nodes, edges


def build_rewrite_graph(rewrite_event: dict[str, Any]) -> tuple[list[GraphNode], list[GraphEdge]]:
    rewrite_id = f"rewrite:{rewrite_event['new_run_id']}:{rewrite_event['token_id']}"
    old_token = f"token:{rewrite_event['old_run_id']}:{rewrite_event['token_id']}"
    new_token = f"token:{rewrite_event['new_run_id']}:{rewrite_event['token_id']}"
    node = GraphNode(
        id=rewrite_id,
        label="Path changed",
        node_type="rewrite",
        status="operator",
        plain_explanation="The operator selected an alternative token and changed the generation path.",
        expert_explanation="The server cancelled the active stream, spliced the transaction log, and started a new run.",
    )
    edges = [
        GraphEdge(
            source=old_token,
            target=rewrite_id,
            edge_type="selected_by_operator",
            plain_explanation="The previous path was interrupted at this token.",
            expert_explanation="This edge is operator provenance, not model telemetry.",
        ),
        GraphEdge(
            source=rewrite_id,
            target=new_token,
            edge_type="caused_rewrite",
            plain_explanation="The new stream resumes from the selected token.",
            expert_explanation="The replacement token is normalized before generation resumes.",
        ),
    ]
    return [node], edges
