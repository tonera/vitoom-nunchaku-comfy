"""Load a nunchaku single-file checkpoint as a ComfyUI MODEL."""

from __future__ import annotations

import contextvars
import logging
import os
from dataclasses import dataclass

import torch

from vitoom_nunchaku.archs import get_spec
from vitoom_nunchaku.archs.base import parse_quant_config
from vitoom_nunchaku.runtime.checkpoint import read_checkpoint
from vitoom_nunchaku.runtime.env import ensure_nunchaku, require_nunchaku
from vitoom_nunchaku.runtime.patcher import assert_hardware, patcher_class

log = logging.getLogger("vitoom_nunchaku")

_ATTENTION_NOTE = False


@dataclass
class LoadState:
    """Per-load config. Specs read this instead of fields on the shared ArchSpec singleton."""

    path: str
    config: dict | None = None


_LOAD_STATE: contextvars.ContextVar[LoadState | None] = contextvars.ContextVar("vitoom_nunchaku_load_state", default=None)


def current_load_state() -> LoadState | None:
    return _LOAD_STATE.get()


def resolve_device(name: str) -> torch.device:
    import comfy.model_management as model_management

    if name in (None, "", "default"):
        return model_management.get_torch_device()
    return torch.device(name)


def resolve_dtype(choice: str, device: torch.device) -> torch.dtype:
    if choice == "bfloat16":
        return torch.bfloat16
    if choice == "float16":
        return torch.float16
    if device.type != "cuda":
        raise RuntimeError("auto dtype requires CUDA. This plugin does not support CPU or MPS.")
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) == (7, 5):
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _note_attention(attention: str) -> None:
    global _ATTENTION_NOTE
    if _ATTENTION_NOTE:
        return
    _ATTENTION_NOTE = True
    if attention == "nunchaku_fp16":
        log.warning("attention=nunchaku_fp16 is not implemented yet. This stage uses ComfyUI native attention.")
    else:
        log.info("This stage uses ComfyUI native attention. Fused attention_fp16 is enabled in a later milestone.")


def _materialize_wtscale(module: torch.nn.Module) -> None:
    """nvfp4 stores wtscale as a tensor attribute. The GEMM expects a Python float, and .to() does not move it."""
    for child in module.modules():
        scale = getattr(child, "wtscale", None)
        if torch.is_tensor(scale):
            child.wtscale = float(scale.detach().float().reshape(-1)[0].item())


def _strict_load(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    incompatible = module.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if not missing and not unexpected:
        return
    lines = ["Quantized weights do not match the ComfyUI model keys."]
    if missing:
        lines.append("Missing keys (up to 20):\n" + "\n".join(missing[:20]))
    if unexpected:
        lines.append("Unexpected keys (up to 20):\n" + "\n".join(unexpected[:20]))
    raise RuntimeError("\n".join(lines))


def load_nunchaku_diffusion_model(path: str, options: dict | None = None, disable_dynamic: bool = False):
    """Called by nodes and ``cached_patcher_init``. ``disable_dynamic=True`` overrides the same option."""
    state = LoadState(path=os.path.abspath(path))
    token = _LOAD_STATE.set(state)
    try:
        return _load_nunchaku_diffusion_model(path, options, disable_dynamic, state)
    finally:
        _LOAD_STATE.reset(token)


def _load_nunchaku_diffusion_model(path: str, options: dict | None, disable_dynamic: bool, state: LoadState):
    import comfy.model_management as model_management

    options = dict(options or {})
    if disable_dynamic:
        options["disable_dynamic_vram"] = True

    ensure_nunchaku()
    from nunchaku.models.transformers.utils import convert_fp16, patch_scale_key
    from nunchaku.utils import load_state_dict_in_safetensors

    info = read_checkpoint(path)
    state.config = info.config
    spec = get_spec(info.model_class)
    require_nunchaku(getattr(spec, "min_nunchaku", "1.4.1"))
    qcfg = parse_quant_config(info.quantization_config, spec, info.tensors)
    spec.validate(info.config, qcfg)

    device = resolve_device(options.get("device", "default"))
    if device.type != "cuda":
        raise RuntimeError(f"Load device must be CUDA, got {device}.")
    assert_hardware(info.quantization_config, device)

    dtype = resolve_dtype(options.get("dtype", "auto"), device)
    attention = options.get("attention", "auto")
    _note_attention(attention)

    bind = getattr(spec, "bind_checkpoint", None)
    if bind is not None:
        bind(path)
    model_config = spec.comfy_model_config(info.config)
    model_config.set_inference_dtype(dtype, None, device=device)
    # Empty weights stay on meta until quantized layers are replaced, so a full bf16 DiT is not allocated first.
    model = model_config.get_model({}, "", device=torch.device("meta"))
    spec.patch_model(model.diffusion_model, qcfg, dtype)
    model.diffusion_model.requires_grad_(False)
    model.diffusion_model.to_empty(device="cpu")
    restore_buffers = getattr(spec, "restore_runtime_buffers", None)
    if restore_buffers is not None:
        restore_buffers(model.diffusion_model)
    model.device = model_management.unet_offload_device()

    state_dict = load_state_dict_in_safetensors(path, device="cpu")
    state_dict = spec.convert_state_dict(state_dict)
    supplement = getattr(spec, "supplement_state_dict", None)
    if supplement is not None:
        state_dict = supplement(state_dict, path)
    patch_scale_key(model.diffusion_model, state_dict)
    _materialize_wtscale(model.diffusion_model)
    if dtype == torch.float16:
        convert_fp16(model.diffusion_model, state_dict)
    _strict_load(model.diffusion_model, state_dict)
    model_management.archive_model_dtypes(model.diffusion_model)

    offload_device = model_management.unet_offload_device()
    load_device = device
    patcher_cls = patcher_class(bool(options.get("disable_dynamic_vram")))
    patcher = patcher_cls(model, load_device=load_device, offload_device=offload_device)
    if not model_management.is_device_cpu(offload_device):
        model.to(offload_device)
    patcher.model_options["transformer_options"]["vitoom_nunchaku_attention"] = attention
    from vitoom_nunchaku.runtime.lora import ensure_diffusion_wrapper

    ensure_diffusion_wrapper(patcher)
    patcher.cached_patcher_init = (load_nunchaku_diffusion_model, (path, options))
    log.info(
        "Loaded %s (%s r%s, %s, attention=%s)",
        path,
        qcfg.precision,
        qcfg.rank,
        dtype,
        attention,
    )
    return patcher
