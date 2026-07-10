from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


SEMANTIC_BRACKETS = {
    "(": ")",
    "[": "]",
    "{": "}",
}


@dataclass(slots=True)
class TokenTransaction:
    token_id: int
    token: str
    run_id: str
    source: str
    logprob: float | None = None
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    hidden_state: dict[str, Any] = field(default_factory=dict)
    created_at_ns: int = field(default_factory=time.perf_counter_ns)

    def to_packet(self, history: list[str]) -> dict[str, Any]:
        return {
            "source": self.source,
            "run_id": self.run_id,
            "token_id": self.token_id,
            "token": self.token,
            "logprob": self.logprob,
            "alternatives": self.alternatives,
            "hidden_state": self.hidden_state,
            "text": "".join(history),
            "committed_history": history,
        }


@dataclass(slots=True)
class RewriteTransaction:
    token_id: int
    replacement: str
    history: list[str]
    text: str
    entry: TokenTransaction


class TokenTransactionLog:
    """Session-local token ledger with deterministic splice/rewrite semantics."""

    def __init__(self) -> None:
        self._entries: list[TokenTransaction] = []

    def reset(self) -> None:
        self._entries.clear()

    def append_packet(self, packet: dict[str, Any]) -> TokenTransaction:
        token_id = int(packet["token_id"])
        expected_id = len(self._entries)
        if token_id != expected_id:
            raise ValueError(f"cannot append token_id {token_id}; expected {expected_id}")

        entry = TokenTransaction(
            token_id=token_id,
            token=packet["token"],
            run_id=packet["run_id"],
            source=packet.get("source", "stream_with_steering_hooks"),
            logprob=packet.get("logprob"),
            alternatives=packet.get("alternatives") or [],
            hidden_state=packet.get("hidden_state") or {},
        )
        self._entries.append(entry)
        return entry

    def rewrite_at(self, token_id: int, alternative: dict[str, Any], run_id: str) -> RewriteTransaction:
        if token_id < 0 or token_id >= len(self._entries):
            raise IndexError(f"token_id {token_id} is outside history length {len(self._entries)}")

        prefix_entries = self._entries[:token_id]
        prefix_text = "".join(entry.token for entry in prefix_entries)
        replacement = normalize_splice_token(prefix_text, alternative["token"])
        entry = TokenTransaction(
            token_id=token_id,
            token=replacement,
            run_id=run_id,
            source="steer_transaction",
            logprob=alternative.get("logprob"),
            alternatives=[],
            hidden_state={
                "rewrite_rank": alternative.get("rank"),
                "rewrite_rationale": alternative.get("rationale", "operator selected alternative token"),
            },
        )
        self._entries = [*prefix_entries, entry]
        history = self.history()
        return RewriteTransaction(
            token_id=token_id,
            replacement=replacement,
            history=history,
            text="".join(history),
            entry=entry,
        )

    def history(self) -> list[str]:
        return [entry.token for entry in self._entries]

    def text(self) -> str:
        return "".join(self.history())

    def packets(self) -> dict[int, dict[str, Any]]:
        history: list[str] = []
        packets: dict[int, dict[str, Any]] = {}
        for entry in self._entries:
            history.append(entry.token)
            packets[entry.token_id] = entry.to_packet(history.copy())
        return packets

    def __len__(self) -> int:
        return len(self._entries)


def normalize_splice_token(prefix_text: str, replacement_token: str) -> str:
    token = replacement_token.replace("\r\n", "\n").replace("\r", "\n")
    if not prefix_text:
        token = token.lstrip()
    elif prefix_text[-1].isspace():
        token = token.lstrip()
    else:
        token = re.sub(r"^\s+", " ", token)

    trailing_match = re.search(r"(\s+)$", token)
    trailing = trailing_match.group(1) if trailing_match else ""
    core = token.rstrip()

    if not core:
        return " "

    core = _close_open_semantic_markers(core)
    suffix = _normalize_trailing_whitespace(core, trailing)
    return core + suffix


def _normalize_trailing_whitespace(core: str, trailing: str) -> str:
    if "\n" in trailing:
        return "\n\n" if trailing.count("\n") > 1 else "\n"
    if trailing:
        return " "
    if core[-1].isalnum() or core[-1] in {")", "]", "}"}:
        return " "
    return ""


def _close_open_semantic_markers(core: str) -> str:
    core = _close_terminal_tag(core)

    stack: list[str] = []
    for character in core:
        if character in SEMANTIC_BRACKETS:
            stack.append(character)
        elif stack and character == SEMANTIC_BRACKETS[stack[-1]]:
            stack.pop()

    if not stack:
        return core

    return core + "".join(SEMANTIC_BRACKETS[character] for character in reversed(stack))


def _close_terminal_tag(core: str) -> str:
    match = re.search(r"<([A-Za-z][\w:-]*)>$", core)
    if not match:
        return core
    tag_name = match.group(1)
    if f"</{tag_name}>" in core:
        return core
    return f"{core}</{tag_name}>"
