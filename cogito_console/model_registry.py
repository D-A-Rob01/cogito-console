from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ReferenceModelSpec:
    key: str
    model_id: str
    revision: str
    license: str
    parameter_count: int
    max_context_tokens: int
    default_max_input_tokens: int
    default_max_new_tokens: int
    torch_dtype: str = "float32"
    trust_remote_code: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REFERENCE_MODELS: dict[str, ReferenceModelSpec] = {
    "smollm2-360m-instruct": ReferenceModelSpec(
        key="smollm2-360m-instruct",
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        revision="main",
        license="apache-2.0",
        parameter_count=360_000_000,
        max_context_tokens=8192,
        default_max_input_tokens=512,
        default_max_new_tokens=32,
    ),
}

DEFAULT_REFERENCE_MODEL = "smollm2-360m-instruct"


def get_reference_model(key: str | None = None) -> ReferenceModelSpec:
    selected = key or os.getenv("COGITO_LOCAL_MODEL", DEFAULT_REFERENCE_MODEL)
    try:
        spec = REFERENCE_MODELS[selected]
    except KeyError as exc:
        allowed = ", ".join(sorted(REFERENCE_MODELS))
        raise ValueError(
            f"Unknown COGITO_LOCAL_MODEL {selected!r}; registry keys: {allowed}"
        ) from exc
    revision = os.getenv("COGITO_LOCAL_MODEL_REVISION")
    if not revision:
        return spec
    return ReferenceModelSpec(
        key=spec.key,
        model_id=spec.model_id,
        revision=revision,
        license=spec.license,
        parameter_count=spec.parameter_count,
        max_context_tokens=spec.max_context_tokens,
        default_max_input_tokens=spec.default_max_input_tokens,
        default_max_new_tokens=spec.default_max_new_tokens,
        torch_dtype=spec.torch_dtype,
        trust_remote_code=spec.trust_remote_code,
    )


def list_reference_models() -> list[dict[str, Any]]:
    return [spec.to_dict() for spec in REFERENCE_MODELS.values()]
