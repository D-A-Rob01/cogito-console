from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(slots=True)
class ContextNode:
    id: str
    label: str
    layer: str
    weight: float
    status: str = "active"


@dataclass(slots=True)
class ContextEdge:
    source: str
    target: str
    label: str
    weight: float


@dataclass(slots=True)
class TemporaryMesh:
    session_id: str
    created_at: float
    parameters: dict[str, Any]
    nodes: list[ContextNode] = field(default_factory=list)
    edges: list[ContextEdge] = field(default_factory=list)
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["age_ms"] = int((time.time() - self.created_at) * 1000)
        return payload


class EpigeneticMemoryController:
    """Manifests temporary context graphs and removes them after completion."""

    def __init__(self) -> None:
        self.active_meshes: dict[str, TemporaryMesh] = {}

    def manifest_temporary_mesh(self, session_id: str, dense_metadata: dict[str, Any]) -> dict[str, Any]:
        nodes = [
            ContextNode("prompt", "Prompt Ingest", "ingest", 1.0),
            ContextNode("semantic_vibe", "Semantic Vibe", "router", 0.92),
            ContextNode("relationship_context", "Relationship Context", "router", 0.86),
            ContextNode("constraint_bounds", "Constraint Bounds", "router", 0.95),
            ContextNode("architecture_pressure", "Architecture Pressure", "router", 0.9),
            ContextNode("latent_beam", "Latent Beam", "interceptor", 0.97),
            ContextNode("epigenetic_mesh", "Epigenetic Mesh", "memory", 0.89),
        ]
        edges = [
            ContextEdge("prompt", "semantic_vibe", "parallel validation", 0.9),
            ContextEdge("prompt", "relationship_context", "parallel validation", 0.82),
            ContextEdge("prompt", "constraint_bounds", "parallel validation", 0.95),
            ContextEdge("prompt", "architecture_pressure", "parallel validation", 0.88),
            ContextEdge("semantic_vibe", "epigenetic_mesh", "parameter imprint", 0.84),
            ContextEdge("relationship_context", "epigenetic_mesh", "operator intent", 0.8),
            ContextEdge("constraint_bounds", "latent_beam", "steering limits", 0.92),
            ContextEdge("epigenetic_mesh", "latent_beam", "temporary context", 0.94),
        ]
        mesh = TemporaryMesh(
            session_id=session_id,
            created_at=time.time(),
            parameters=dense_metadata,
            nodes=nodes,
            edges=edges,
        )
        self.active_meshes[session_id] = mesh
        return mesh.to_dict()

    def get_mesh(self, session_id: str) -> dict[str, Any] | None:
        mesh = self.active_meshes.get(session_id)
        return mesh.to_dict() if mesh else None

    def evaporate_mesh(self, session_id: str) -> dict[str, Any]:
        mesh = self.active_meshes.pop(session_id, None)
        if mesh is None:
            return {"session_id": session_id, "status": "already_evaporated", "released_nodes": 0}
        return {
            "session_id": session_id,
            "status": "evaporated",
            "released_nodes": len(mesh.nodes),
            "released_edges": len(mesh.edges),
            "age_ms": int((time.time() - mesh.created_at) * 1000),
        }

