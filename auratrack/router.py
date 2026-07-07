from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(slots=True)
class MicroValidation:
    name: str
    verdict: str
    confidence: float
    signals: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfluenceRouter:
    """Parallel sub-agent layer that validates intent before generation starts."""

    async def analyze_sub_perceptions(self, raw_prompt: str) -> dict[str, dict[str, Any]]:
        tasks = {
            "semantic_vibe": self._semantic_vibe(raw_prompt),
            "relationship_context": self._relationship_context(raw_prompt),
            "constraint_bounds": self._constraint_bounds(raw_prompt),
            "architecture_pressure": self._architecture_pressure(raw_prompt),
        }
        results = await asyncio.gather(*tasks.values())
        return {name: result.to_dict() for name, result in zip(tasks.keys(), results)}

    async def _semantic_vibe(self, prompt: str) -> MicroValidation:
        await asyncio.sleep(0.08)
        lower = prompt.lower()
        signals = self._keywords(lower, ["introspective", "sub-perceptual", "steering", "latent", "synaptic"])
        return MicroValidation(
            name="Semantic Vibe",
            verdict="Build as an instrumented cognition console, not a plain chat endpoint.",
            confidence=0.92 if signals else 0.72,
            signals=signals or ["systems-language", "runtime-introspection"],
        )

    async def _relationship_context(self, prompt: str) -> MicroValidation:
        await asyncio.sleep(0.05)
        signals = self._keywords(prompt.lower(), ["human", "ui", "click", "mid-stream", "real-time"])
        return MicroValidation(
            name="Relationship Context",
            verdict="Keep the human in the inference loop with reversible, visible steering.",
            confidence=0.88 if signals else 0.67,
            signals=signals or ["operator-in-the-loop"],
        )

    async def _constraint_bounds(self, prompt: str) -> MicroValidation:
        await asyncio.sleep(0.07)
        signals = self._keywords(prompt.lower(), ["fastapi", "websocket", "frontend", "logprob", "history"])
        return MicroValidation(
            name="Constraint Bounds",
            verdict="Expose a websocket protocol with token packets and explicit rewrite commands.",
            confidence=0.95 if len(signals) >= 3 else 0.78,
            signals=signals or ["streaming-contract", "history-rewrite"],
        )

    async def _architecture_pressure(self, prompt: str) -> MicroValidation:
        await asyncio.sleep(0.06)
        component_hits = len(re.findall(r"router|interceptor|memory|controller|orchestrator", prompt, re.I))
        return MicroValidation(
            name="Architecture Pressure",
            verdict="Separate routing, latent streaming, and temporary memory as swappable middleware layers.",
            confidence=min(0.98, 0.7 + component_hits * 0.05),
            signals=["layered-routing", "adapter-boundary", "temporary-context-graph"],
        )

    @staticmethod
    def _keywords(text: str, words: list[str]) -> list[str]:
        return [word for word in words if word in text]

