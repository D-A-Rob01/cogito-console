from __future__ import annotations

import math
from collections.abc import Sequence

from .schemas import MetricStatus, MetricValue


class MetricEngine:
    """Computes auditable metrics and preserves synthetic status when inputs are synthetic."""

    @staticmethod
    def token_probability(logprob: MetricValue, evidence_id: str) -> MetricValue:
        status: MetricStatus = "synthetic" if logprob.status == "synthetic" else "derived"
        value = math.exp(float(logprob.value)) if logprob.value is not None else None
        return MetricValue(
            name="token_probability",
            value=value,
            unit="probability",
            status=status,
            evidence_id=evidence_id,
            display_label="Probability",
            plain_explanation="How likely this token was according to its logprob.",
            expert_explanation="Computed with probability = exp(logprob). If the source logprob is synthetic, this metric remains synthetic.",
        )

    @staticmethod
    def alternative_margin(
        selected_logprob: MetricValue,
        alternative_logprob: MetricValue,
        evidence_id: str,
    ) -> MetricValue:
        status: MetricStatus = (
            "synthetic"
            if "synthetic" in {selected_logprob.status, alternative_logprob.status}
            else "derived"
        )
        selected = float(selected_logprob.value) if selected_logprob.value is not None else 0.0
        alternative = float(alternative_logprob.value) if alternative_logprob.value is not None else 0.0
        return MetricValue(
            name="alternative_margin",
            value=selected - alternative,
            unit="logprob",
            status=status,
            evidence_id=evidence_id,
            display_label="Gap from chosen token",
            plain_explanation="How far behind this alternative was from the selected token.",
            expert_explanation="Computed with selected_logprob - alternative_logprob.",
        )

    @staticmethod
    def entropy_from_top_logprobs(logprobs: Sequence[MetricValue], evidence_id: str) -> MetricValue:
        if not logprobs:
            entropy = 0.0
        else:
            probabilities = [math.exp(float(metric.value)) for metric in logprobs if metric.value is not None]
            total = sum(probabilities)
            normalized = [probability / total for probability in probabilities] if total else []
            entropy = -sum(p * math.log(p, 2) for p in normalized if p > 0)
        status: MetricStatus = "synthetic" if any(metric.status == "synthetic" for metric in logprobs) else "derived"
        return MetricValue(
            name="entropy_from_top_logprobs",
            value=entropy,
            unit="bits",
            status=status,
            evidence_id=evidence_id,
            display_label="Choice spread",
            plain_explanation="Higher entropy means several next-token options were plausible.",
            expert_explanation="Top logprobs are exponentiated, normalized, then scored with Shannon entropy.",
        )

    @staticmethod
    def confidence_gap(logprobs: Sequence[MetricValue], evidence_id: str) -> MetricValue:
        probabilities = sorted(
            [math.exp(float(metric.value)) for metric in logprobs if metric.value is not None],
            reverse=True,
        )
        gap = probabilities[0] - probabilities[1] if len(probabilities) >= 2 else None
        status: MetricStatus = "synthetic" if any(metric.status == "synthetic" for metric in logprobs) else "derived"
        return MetricValue(
            name="confidence_gap",
            value=gap,
            unit="probability",
            status=status,
            evidence_id=evidence_id,
            display_label="Confidence gap",
            plain_explanation="Large gap means the model strongly preferred one continuation.",
            expert_explanation="Computed with probability(top_1) - probability(top_2).",
        )

    @staticmethod
    def tokens_per_second(token_count: int, duration_ms: float, evidence_id: str) -> MetricValue:
        value = token_count / (duration_ms / 1000) if duration_ms > 0 else 0.0
        return MetricValue(
            name="tokens_per_second",
            value=value,
            unit="tokens/s",
            status="real",
            evidence_id=evidence_id,
            display_label="Speed",
            plain_explanation="How many tokens arrived per second during this run.",
            expert_explanation="Measured from runtime timestamps around token streaming.",
        )

    @staticmethod
    def token_latency_ms(duration_ms: float, evidence_id: str) -> MetricValue:
        return MetricValue(
            name="token_latency_ms",
            value=duration_ms,
            unit="ms",
            status="real",
            evidence_id=evidence_id,
            display_label="Token latency",
            plain_explanation="How long the runtime waited before this token packet arrived.",
            expert_explanation="Measured with time.perf_counter around the provider stream.",
        )

    @staticmethod
    def rewrite_count(count: int, evidence_id: str) -> MetricValue:
        return MetricValue(
            name="rewrite_count",
            value=count,
            unit="rewrites",
            status="real",
            evidence_id=evidence_id,
            display_label="Path changes",
            plain_explanation="How many times the operator changed the generation path.",
            expert_explanation="Counted from received steer events accepted by the server.",
        )

    @staticmethod
    def branch_depth(depth: int, evidence_id: str) -> MetricValue:
        return MetricValue(
            name="branch_depth",
            value=depth,
            unit="branches",
            status="derived",
            evidence_id=evidence_id,
            display_label="Branch depth",
            plain_explanation="How many rewrite steps sit behind the current path.",
            expert_explanation="Derived from rewrite graph traversal depth.",
        )

    @staticmethod
    def graph_density(actual_edges: int, possible_edges: int, evidence_id: str) -> MetricValue:
        value = actual_edges / possible_edges if possible_edges > 0 else 0.0
        return MetricValue(
            name="graph_density",
            value=value,
            unit="ratio",
            status="derived",
            evidence_id=evidence_id,
            display_label="Graph density",
            plain_explanation="How connected the visible graph is.",
            expert_explanation="Computed with actual_edges / possible_edges.",
        )

    @staticmethod
    def session_duration_ms(duration_ms: float, evidence_id: str) -> MetricValue:
        return MetricValue(
            name="session_duration_ms",
            value=duration_ms,
            unit="ms",
            status="real",
            evidence_id=evidence_id,
            display_label="Session duration",
            plain_explanation="How long the temporary session has been alive.",
            expert_explanation="Measured from runtime session start and current runtime timestamp.",
        )
