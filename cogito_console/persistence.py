from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class PersistenceStore:
    def __init__(self, path: str | Path = "./data/cogito.sqlite", *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

    @classmethod
    def from_env(cls) -> "PersistenceStore":
        enabled = os.getenv("COGITO_PERSIST", "true").strip().lower() != "false"
        path = os.getenv("COGITO_DB_PATH", "./data/cogito.sqlite")
        return cls(path, enabled=enabled)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    prompt TEXT,
                    provider_name TEXT,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS token_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    token_id INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS alternative_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    token_id INTEGER NOT NULL,
                    rank INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rewrite_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    old_run_id TEXT NOT NULL,
                    new_run_id TEXT NOT NULL,
                    token_id INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS graph_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS graph_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evidence_records (
                    evidence_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS metric_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    event_ref TEXT,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    payload TEXT NOT NULL
                );
                """
            )

    def upsert_session(
        self,
        session_id: str,
        *,
        prompt: str,
        provider_name: str,
        status: str = "active",
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        created_at = time.time()
        body = payload or {}
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions(session_id, prompt, provider_name, created_at, status, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    prompt=excluded.prompt,
                    provider_name=excluded.provider_name,
                    status=excluded.status,
                    payload=excluded.payload
                """,
                (session_id, prompt, provider_name, created_at, status, _json(body)),
            )

    def upsert_run(
        self,
        run_id: str,
        *,
        session_id: str,
        status: str = "active",
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        created_at = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(run_id, session_id, created_at, status, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET status=excluded.status, payload=excluded.payload
                """,
                (run_id, session_id, created_at, status, _json(payload or {})),
            )

    def save_token_packet(self, session_id: str, packet: dict[str, Any]) -> None:
        if not self.enabled:
            return
        token_event = packet.get("token_event") or {}
        run_id = str(packet.get("run_id") or token_event.get("run_id"))
        token_id = int(packet.get("token_id") or token_event.get("token_id") or 0)
        created_at = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO token_events(session_id, run_id, token_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, run_id, token_id, created_at, _json(packet)),
            )
            for alternative in packet.get("alternatives") or []:
                connection.execute(
                    """
                    INSERT INTO alternative_events(session_id, run_id, token_id, rank, created_at, payload)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        run_id,
                        token_id,
                        int(alternative.get("rank") or 0),
                        created_at,
                        _json(alternative),
                    ),
                )
            for record in packet.get("evidence") or []:
                self._save_evidence(connection, session_id, record)
            for metric in _packet_metrics(packet):
                self._save_metric(connection, session_id, f"token:{run_id}:{token_id}", metric)

    def save_rewrite_event(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rewrite_events(session_id, old_run_id, new_run_id, token_id, created_at, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event["session_id"],
                    event["old_run_id"],
                    event["new_run_id"],
                    int(event["token_id"]),
                    time.time(),
                    _json(event),
                ),
            )

    def save_evidence_records(self, session_id: str, records: list[Any]) -> None:
        if not self.enabled:
            return
        with self._connect() as connection:
            for record in records:
                self._save_evidence(connection, session_id, _payload(record))

    def save_metrics(self, session_id: str, event_ref: str, metrics: list[Any]) -> None:
        if not self.enabled:
            return
        with self._connect() as connection:
            for metric in metrics:
                self._save_metric(connection, session_id, event_ref, _payload(metric))

    def save_graph(self, session_id: str, nodes: list[Any], edges: list[Any]) -> None:
        if not self.enabled:
            return
        created_at = time.time()
        with self._connect() as connection:
            for node in nodes:
                payload = _payload(node)
                connection.execute(
                    """
                    INSERT INTO graph_nodes(session_id, node_id, node_type, status, created_at, payload)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        payload["id"],
                        payload["node_type"],
                        payload["status"],
                        created_at,
                        _json(payload),
                    ),
                )
            for edge in edges:
                payload = _payload(edge)
                connection.execute(
                    """
                    INSERT INTO graph_edges(session_id, source, target, edge_type, created_at, payload)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        payload["source"],
                        payload["target"],
                        payload["edge_type"],
                        created_at,
                        _json(payload),
                    ),
                )

    def list_sessions(self) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        return [_row(row) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return _row(row) if row else None

    def get_events(self, session_id: str) -> dict[str, list[dict[str, Any]]]:
        if not self.enabled:
            return {"tokens": [], "alternatives": [], "rewrites": []}
        with self._connect() as connection:
            token_rows = connection.execute(
                "SELECT * FROM token_events WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            alternative_rows = connection.execute(
                "SELECT * FROM alternative_events WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            rewrite_rows = connection.execute(
                "SELECT * FROM rewrite_events WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return {
            "tokens": [_row(row) for row in token_rows],
            "alternatives": [_row(row) for row in alternative_rows],
            "rewrites": [_row(row) for row in rewrite_rows],
        }

    def get_graph(self, session_id: str) -> dict[str, list[dict[str, Any]]]:
        if not self.enabled:
            return {"nodes": [], "edges": []}
        with self._connect() as connection:
            node_rows = connection.execute(
                "SELECT * FROM graph_nodes WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            edge_rows = connection.execute(
                "SELECT * FROM graph_edges WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return {"nodes": [_row(row) for row in node_rows], "edges": [_row(row) for row in edge_rows]}

    def get_metrics(self, session_id: str) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM metric_values WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [_row(row) for row in rows]

    def get_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_records WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()
        return _row(row) if row else None

    def _save_evidence(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        record: dict[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT OR REPLACE INTO evidence_records(evidence_id, session_id, source_kind, created_at, payload)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record["evidence_id"],
                session_id,
                record["source_kind"],
                float(record.get("created_at") or time.time()),
                _json(record),
            ),
        )

    def _save_metric(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        event_ref: str,
        metric: dict[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO metric_values(session_id, event_ref, name, status, evidence_id, created_at, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                event_ref,
                metric["name"],
                metric["status"],
                metric["evidence_id"],
                time.time(),
                _json(metric),
            ),
        )


def _packet_metrics(packet: dict[str, Any]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    metrics.extend((packet.get("metrics") or {}).values())
    metrics.extend((packet.get("hidden_state_metrics") or {}).values())
    for alternative in packet.get("alternatives") or []:
        metrics.extend((alternative.get("metrics") or {}).values())
    return [metric for metric in metrics if metric]


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    return dict(value)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _row(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    if "payload" in payload:
        payload["payload"] = json.loads(payload["payload"])
    return payload
