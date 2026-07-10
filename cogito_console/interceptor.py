from __future__ import annotations

import asyncio
import hashlib
import math
import re
import uuid
from dataclasses import dataclass, asdict
from typing import Any, AsyncIterator


@dataclass(slots=True)
class AlternativeToken:
    token: str
    logprob: float
    rank: int
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\S+\s*", text)


def rewrite_history_at(history: list[str], token_id: int, alternative_token: str) -> list[str]:
    if token_id < 0 or token_id >= len(history):
        raise IndexError(f"token_id {token_id} is outside history length {len(history)}")
    return [*history[:token_id], alternative_token]


class LatentBeamInterceptor:
    """Streams token packets with exposed alternatives and steerable history."""

    def __init__(self, token_delay: float = 0.14) -> None:
        self.token_delay = token_delay

    async def stream_with_steering_hooks(
        self,
        prompt: str,
        system_context: str,
        committed_history: list[str] | None = None,
        run_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        run_id = run_id or uuid.uuid4().hex
        history = committed_history or []
        plan = self._compose_plan(prompt, system_context, history)
        planned_tokens = tokenize(plan)
        start_index = min(len(history), len(planned_tokens))

        for index in range(start_index, len(planned_tokens)):
            if self.token_delay:
                await asyncio.sleep(self.token_delay)

            token = planned_tokens[index]
            hidden_state = self._hidden_state(prompt, token, index, history)
            alternatives = [
                alt.to_dict() for alt in self._alternatives_for(prompt, token, index, history, hidden_state)
            ]
            packet_history = [*history, *planned_tokens[start_index : index + 1]]
            yield {
                "source": "stream_with_steering_hooks",
                "run_id": run_id,
                "token_id": index,
                "token": token,
                "logprob": self._logprob(token, index, hidden_state),
                "alternatives": alternatives,
                "hidden_state": hidden_state,
                "text": "".join(packet_history),
                "committed_history": packet_history,
            }

    def _compose_plan(self, prompt: str, system_context: str, history: list[str]) -> str:
        branch = self._branch_from_history(history)
        normalized_prompt = " ".join(prompt.strip().split())
        clipped_prompt = normalized_prompt[:130] + ("..." if len(normalized_prompt) > 130 else "")

        openings = {
            "default": "Confluence Router validates the prompt across semantic, relational, constraint, and architecture nerves. ",
            "skeptical": "Skeptical steering tightens every validation gate before any confident continuation. ",
            "creative": "Creative steering opens a wider beam while keeping the operator in control of each branch. ",
            "precise": "Precise steering narrows the beam to auditable contracts, token packets, and replayable state. ",
            "fast": "Fast steering compresses the route into low-latency checks and immediate token-level feedback. ",
        }
        body = (
            "Latent Beam emits each token with top alternatives, logprobs, and hidden-state probes; "
            "a clicked alternative rewrites the history at that token and restarts generation from the new prefix. "
        )
        memory = (
            "Epigenetic Mesh holds temporary context edges only for this task, then evaporates after completion "
            "so no dormant graph keeps taxing the runtime. "
        )
        context_note = (
            f"Current prompt imprint: {clipped_prompt}. "
            "Steering hooks remain active until the stream settles."
        )
        if "constraint" in system_context.lower():
            body += "The websocket protocol treats steering as a first-class event rather than a UI afterthought. "
        return openings[branch] + body + memory + context_note

    @staticmethod
    def _branch_from_history(history: list[str]) -> str:
        joined = "".join(history).lower()
        if "skeptical" in joined:
            return "skeptical"
        if "creative" in joined:
            return "creative"
        if "precise" in joined or "auditable" in joined:
            return "precise"
        if "fast" in joined or "low-latency" in joined:
            return "fast"
        return "default"

    def _alternatives_for(
        self,
        prompt: str,
        token: str,
        index: int,
        history: list[str],
        hidden_state: dict[str, float],
    ) -> list[AlternativeToken]:
        base = [
            ("precise ", "Force a stricter architecture branch."),
            ("creative ", "Widen the inference beam."),
            ("skeptical ", "Increase validation pressure."),
            ("fast ", "Prefer low-latency continuation."),
            ("auditable ", "Favor replayable state transitions."),
        ]
        offset = self._stable_int(f"{prompt}|{token}|{index}") % len(base)
        rotated = base[offset:] + base[:offset]
        entropy = hidden_state["entropy"]
        return [
            AlternativeToken(
                token=alt_token,
                logprob=round(-0.35 - rank * 0.42 - entropy * 0.2, 3),
                rank=rank + 1,
                rationale=rationale,
            )
            for rank, (alt_token, rationale) in enumerate(rotated[:5])
        ]

    def _hidden_state(self, prompt: str, token: str, index: int, history: list[str]) -> dict[str, float]:
        seed = f"{prompt}|{token}|{index}|{''.join(history[-8:])}"
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        values = [byte / 255 for byte in digest[:6]]
        entropy = -sum(v * math.log(max(v, 0.0001)) for v in values[:4]) / 4
        return {
            "semantic": round(values[0], 3),
            "constraint": round(values[1], 3),
            "operator_pull": round(values[2], 3),
            "memory_load": round(values[3], 3),
            "steering_tension": round(values[4], 3),
            "entropy": round(min(1.0, entropy), 3),
        }

    @staticmethod
    def _logprob(token: str, index: int, hidden_state: dict[str, float]) -> float:
        base = -0.08 - (index % 7) * 0.07
        pressure = hidden_state["steering_tension"] * 0.25
        return round(base - pressure, 3)

    @staticmethod
    def _stable_int(text: str) -> int:
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)

