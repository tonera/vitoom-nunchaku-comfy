"""ArchSpec registry."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class QuantConfig:
    def __init__(self, raw: dict, precision: str, rank: int, swiglu_order: str | None):
        self.raw = raw
        self.precision = precision
        self.rank = rank
        self.swiglu_order = swiglu_order


class ArchSpec(Protocol):
    model_class: str
    supported_precisions: tuple[str, ...]
    min_nunchaku: str

    def comfy_model_config(self, diffusers_config: dict):
        """Build a ComfyUI supported_models instance from metadata config."""

    def validate(self, diffusers_config: dict, qcfg: QuantConfig) -> None:
        """Check precision, rank, and architecture-specific structural constraints."""

    def patch_model(self, diffusion_model: nn.Module, qcfg: QuantConfig, dtype: torch.dtype) -> None:
        """Replace quantized Linear layers with SVDQ and install the rewritten forward."""

    def convert_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert diffusers key names to ComfyUI key names."""


_REGISTRY: dict[str, ArchSpec] = {}


def register_spec(spec: ArchSpec) -> None:
    _REGISTRY[spec.model_class] = spec


def get_spec(model_class: str) -> ArchSpec:
    try:
        return _REGISTRY[model_class]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "(empty)"
        raise KeyError(f"No ArchSpec for {model_class!r}. Registered: {known}") from exc


def registered_model_classes() -> frozenset[str]:
    return frozenset(_REGISTRY)


def _infer_svd_rank(tensors: dict | None) -> int | None:
    """FLUX.1 checkpoints omit quantization_config.rank. The low-rank factor stores it."""
    if not tensors:
        return None
    for key, meta in tensors.items():
        if not key.endswith((".lora_down", ".proj_down")):
            continue
        shape = meta.get("shape") if isinstance(meta, dict) else None
        if shape:
            return int(shape[-1])
    return None


def parse_quant_config(raw: dict, spec: ArchSpec, tensors: dict | None = None) -> QuantConfig:
    from nunchaku.utils import get_precision_from_quantization_config

    precision = get_precision_from_quantization_config(raw)
    if precision not in spec.supported_precisions:
        raise ValueError(
            f"{spec.model_class} supports {spec.supported_precisions}; checkpoint precision is {precision!r}."
        )
    declared = raw.get("precision")
    # Official Z-Image exports write nvfp4 as precision="fp4".
    aliases = {"fp4": "nvfp4"}
    if declared is not None and aliases.get(declared, declared) != precision:
        raise ValueError(
            f"quantization_config.precision is {declared!r}, but the weight dtype maps to {precision!r}."
        )
    rank = raw.get("rank")
    if rank is None:
        rank = _infer_svd_rank(tensors)
    min_rank = int(getattr(spec, "min_rank", 1))
    if rank is None or int(rank) < min_rank:
        if min_rank <= 0:
            raise ValueError(f"Checkpoint requires rank >= {min_rank}, got {rank!r}.")
        raise ValueError(f"Checkpoint requires rank > 0, got {rank!r}.")
    return QuantConfig(raw=raw, precision=precision, rank=int(rank), swiglu_order=raw.get("swiglu_order"))
