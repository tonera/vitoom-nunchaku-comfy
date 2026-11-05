"""Read only the safetensors header and cache it by mtime."""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field


@dataclass
class CheckpointInfo:
    path: str
    model_class: str
    config: dict
    quantization_config: dict
    comfy_config: dict
    precision: str
    rank: int
    tensors: dict[str, dict] = field(default_factory=dict)

    @property
    def label(self) -> str:
        name = os.path.basename(self.path)
        if self.rank:
            return f"{name} [{self.precision} r{self.rank}]"
        return f"{name} [{self.precision}]"


_CACHE: dict[str, tuple[int, int, CheckpointInfo]] = {}


def precision_from_config(quantization_config: dict) -> str:
    """Does not import nunchaku. Used when the UI lists files."""
    weight = quantization_config.get("weight") or {}
    dtype = weight.get("dtype")
    if dtype == "int8":
        return "int8"
    if dtype == "fp4_e2m1_all":
        return "nvfp4"
    if dtype == "int4":
        return "int4"
    declared = quantization_config.get("precision")
    return str(declared) if declared else "unknown"


def read_header(path: str) -> tuple[dict, dict]:
    """Return (metadata, tensor_index). tensor_index values are the dtype/shape from the safetensors header."""
    with open(path, "rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path} is not a safetensors file")
        header_len = struct.unpack("<Q", raw_len)[0]
        raw = handle.read(header_len)
    if len(raw) != header_len:
        raise ValueError(f"The safetensors header of {path} is truncated")
    header = json.loads(raw)
    metadata = header.pop("__metadata__", {}) or {}
    return metadata, header


def read_checkpoint(path: str) -> CheckpointInfo:
    stat = os.stat(path)
    key = os.path.abspath(path)
    cached = _CACHE.get(key)
    if cached is not None and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]

    metadata, tensors = read_header(path)
    config_raw = metadata.get("config") or "{}"
    quant_raw = metadata.get("quantization_config") or "{}"
    comfy_raw = metadata.get("comfy_config") or "{}"
    config = json.loads(config_raw) if isinstance(config_raw, str) else dict(config_raw)
    quant = json.loads(quant_raw) if isinstance(quant_raw, str) else dict(quant_raw)
    try:
        comfy_config = json.loads(comfy_raw) if isinstance(comfy_raw, str) else dict(comfy_raw)
    except json.JSONDecodeError:
        comfy_config = {}
    rank_raw = quant.get("rank")
    if rank_raw in (None, ""):
        from vitoom_nunchaku.archs.base import _infer_svd_rank

        rank_raw = _infer_svd_rank(tensors) or 0
    info = CheckpointInfo(
        path=key,
        model_class=str(metadata.get("model_class") or ""),
        config=config,
        quantization_config=quant,
        comfy_config=comfy_config if isinstance(comfy_config, dict) else {},
        precision=precision_from_config(quant),
        rank=int(rank_raw),
        tensors=tensors,
    )
    _CACHE[key] = (stat.st_mtime_ns, stat.st_size, info)
    return info


def clear_cache() -> None:
    _CACHE.clear()


def iter_diffusion_checkpoints():
    """Scan ComfyUI diffusion_models and keep only nunchaku single files the registry knows."""
    import folder_paths

    from vitoom_nunchaku.archs import registered_model_classes

    known = registered_model_classes()
    names = folder_paths.get_filename_list("diffusion_models")
    found = []
    for name in names:
        if not str(name).endswith(".safetensors"):
            continue
        path = folder_paths.get_full_path("diffusion_models", name)
        if not path or not os.path.isfile(path):
            continue
        try:
            info = read_checkpoint(path)
        except Exception:
            continue
        if info.model_class in known:
            found.append((name, info))
    return found


def choice_label(filename: str, info: CheckpointInfo) -> str:
    if info.rank:
        return f"{filename} [{info.precision} r{info.rank}]"
    return f"{filename} [{info.precision}]"


def filename_from_choice(choice: str) -> str:
    marker = " ["
    if choice.endswith("]") and marker in choice:
        return choice.rsplit(marker, 1)[0]
    return choice
