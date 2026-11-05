"""Fuse LoRAs into SVDQ low-rank branches and float or AWQ residuals.

ComfyUI's weight patches edit ``.weight``. SVDQ modules have no such parameter, so each
LoRA down/up is concatenated onto ``proj_down`` / ``proj_up``. AdaLN is AWQ W4A16, so
that LoRA is added in the forward instead. The shared diffusion module is reset whenever
the active stack changes, which lets several cloned patchers take turns.
"""

from __future__ import annotations

import logging
import os

import torch
from torch import nn

log = logging.getLogger("vitoom_nunchaku")

_STACK_KEY = "vitoom_nunchaku_loras"
_WRAPPER_KEY = "vitoom_nunchaku_lora"
_PAIR_CACHE: dict[tuple[str, int, int], dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}


def ensure_diffusion_wrapper(patcher) -> None:
    from comfy.patcher_extension import CallbacksMP, WrappersMP

    for wrapper in patcher.get_all_wrappers(WrappersMP.DIFFUSION_MODEL):
        if getattr(wrapper, "_vitoom_lora", False):
            break
    else:
        patcher.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, _WRAPPER_KEY, diffusion_lora_wrapper)
    for callback in patcher.get_all_callbacks(CallbacksMP.ON_LOAD):
        if getattr(callback, "_vitoom_lora", False):
            return
    patcher.add_callback_with_key(CallbacksMP.ON_LOAD, _WRAPPER_KEY, _invalidate_lora_on_load)


def _invalidate_lora_on_load(patcher, *_args, **_kwargs) -> None:
    """Dynamic VRAM restores parameters from the pre-LoRA backup. Forget the fused signature so the next forward applies it again."""
    model = getattr(patcher, "model", None)
    diffusion = getattr(model, "diffusion_model", None)
    if diffusion is None or not getattr(diffusion, "_vitoom_lora_stamp", ()):
        return
    diffusion._vitoom_lora_signature = None


_invalidate_lora_on_load._vitoom_lora = True


def diffusion_lora_wrapper(executor, *args, **kwargs):
    options = kwargs.get("transformer_options")
    if not isinstance(options, dict):
        options = next((arg for arg in args if isinstance(arg, dict) and _STACK_KEY in arg), None)
    stack = () if not isinstance(options, dict) else tuple(options.get(_STACK_KEY) or ())
    sync_lora_stack(executor.class_obj, stack)
    return executor(*args, **kwargs)


diffusion_lora_wrapper._vitoom_lora = True


def append_lora(model, path: str, strength: float):
    """Clone the patcher and append one LoRA. Weights are fused on the next forward."""
    cloned = model.clone()
    options = cloned.model_options.setdefault("transformer_options", {})
    stack = list(options.get(_STACK_KEY) or ())
    stack.append((os.path.abspath(path), float(strength)))
    options[_STACK_KEY] = stack
    ensure_diffusion_wrapper(cloned)
    return cloned


def sync_lora_stack(diffusion: nn.Module, stack: tuple) -> None:
    signature = tuple((path, float(strength)) for path, strength in stack if float(strength) != 0.0)
    stamp = getattr(diffusion, "_vitoom_lora_stamp", ())
    if getattr(diffusion, "_vitoom_lora_signature", None) == signature and _lowrank_intact(diffusion, stamp):
        return
    _reset(diffusion)
    if signature:
        _apply(diffusion, signature)
    diffusion._vitoom_lora_signature = signature
    diffusion._vitoom_lora_stamp = _lowrank_stamp(diffusion) if signature else ()
    if signature:
        log.info("Fused %d LoRA(s) into the quantized DiT.", len(signature))
    else:
        log.info("Restored the quantized DiT to its original low-rank branches.")


def _lowrank_stamp(diffusion: nn.Module) -> tuple:
    base = getattr(diffusion, "_vitoom_lora_base", None) or {}
    return tuple((name, id(_resolve(diffusion, name).proj_down)) for name in base)


def _lowrank_intact(diffusion: nn.Module, stamp) -> bool:
    if not stamp:
        return True
    for name, ident in stamp:
        try:
            module = _resolve(diffusion, name)
        except (AttributeError, IndexError, TypeError):
            return False
        if id(getattr(module, "proj_down", None)) != ident:
            return False
    return True


def flux_lora_pairs(path: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Map one FLUX.1 LoRA file onto ComfyUI Flux module paths. Down is ``[rank, in]``."""
    stat = os.stat(path)
    key = (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)
    cached = _PAIR_CACHE.get(key)
    if cached is not None:
        return cached
    from nunchaku.lora.flux.diffusers_converter import to_diffusers

    converted = to_diffusers(path)
    pairs = _pairs_from_diffusers(converted)
    if not pairs:
        raise ValueError(f"{path} produced no FLUX.1 LoRA layers.")
    _PAIR_CACHE[key] = pairs
    return pairs


def _pairs_from_diffusers(state: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    grouped: dict[str, list[torch.Tensor | None]] = {}
    unused = []
    for key, tensor in state.items():
        if key.endswith(".lora_A.weight"):
            base = key[: -len(".lora_A.weight")]
            slot = grouped.setdefault(base, [None, None])
            slot[0] = tensor
        elif key.endswith(".lora_B.weight"):
            base = key[: -len(".lora_B.weight")]
            slot = grouped.setdefault(base, [None, None])
            slot[1] = tensor
        else:
            unused.append(key)
    if unused:
        raise ValueError("Unsupported LoRA tensors: " + ", ".join(unused[:8]))

    direct: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    qkv: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for base, pair in grouped.items():
        down, up = pair
        if down is None or up is None:
            raise ValueError(f"LoRA {base} is missing lora_A or lora_B.")
        target, role = _target(base)
        if role is None:
            direct[target] = (down, up)
        else:
            qkv.setdefault(target, {})[role] = (down, up)

    for target, parts in qkv.items():
        missing = [name for name in ("q", "k", "v") if name not in parts]
        if missing:
            raise ValueError(f"LoRA QKV {target} is missing {missing}.")
        direct[target] = _fuse_qkv(parts["q"], parts["k"], parts["v"])

    expanded: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for target, (down, up) in direct.items():
        if target.endswith(".proj_out_fused"):
            module = target[: -len(".proj_out_fused")]
            hidden = up.shape[0]
            if down.shape[1] <= hidden:
                raise ValueError(
                    f"Single-block proj_out LoRA for {module} has input {down.shape[1]}, expected more than {hidden}."
                )
            expanded[f"{module}.out_proj"] = (down[:, :hidden].contiguous(), up)
            expanded[f"{module}.mlp_fc2"] = (down[:, hidden:].contiguous(), up)
            continue
        expanded[target] = _reorder_adanorm(target, down, up)
    return expanded


def _target(base: str) -> tuple[str, str | None]:
    double = "transformer.transformer_blocks."
    single = "transformer.single_transformer_blocks."
    if base.startswith(double):
        index, _, rest = base[len(double):].partition(".")
        prefix = f"double_blocks.{index}"
        mapping = {
            "attn.to_out.0": (f"{prefix}.img_attn.proj", None),
            "attn.to_add_out": (f"{prefix}.txt_attn.proj", None),
            "ff.net.0.proj": (f"{prefix}.img_mlp.0", None),
            "ff.net.2": (f"{prefix}.img_mlp.2", None),
            "ff_context.net.0.proj": (f"{prefix}.txt_mlp.0", None),
            "ff_context.net.2": (f"{prefix}.txt_mlp.2", None),
            "norm1.linear": (f"{prefix}.img_mod.lin", None),
            "norm1_context.linear": (f"{prefix}.txt_mod.lin", None),
            "attn.to_q": (f"{prefix}.img_attn.qkv", "q"),
            "attn.to_k": (f"{prefix}.img_attn.qkv", "k"),
            "attn.to_v": (f"{prefix}.img_attn.qkv", "v"),
            "attn.add_q_proj": (f"{prefix}.txt_attn.qkv", "q"),
            "attn.add_k_proj": (f"{prefix}.txt_attn.qkv", "k"),
            "attn.add_v_proj": (f"{prefix}.txt_attn.qkv", "v"),
        }
    elif base.startswith(single):
        index, _, rest = base[len(single):].partition(".")
        prefix = f"single_blocks.{index}"
        mapping = {
            "proj_mlp": (f"{prefix}.mlp_fc1", None),
            "proj_out": (f"{prefix}.proj_out_fused", None),
            "norm.linear": (f"{prefix}.modulation.lin", None),
            "attn.to_q": (f"{prefix}.qkv", "q"),
            "attn.to_k": (f"{prefix}.qkv", "k"),
            "attn.to_v": (f"{prefix}.qkv", "v"),
        }
    else:
        raise ValueError(f"Unsupported FLUX.1 LoRA key {base}.")
    found = mapping.get(rest)
    if found is None:
        raise ValueError(f"Unsupported FLUX.1 LoRA key {base}.")
    return found


def _fuse_qkv(q, k, v) -> tuple[torch.Tensor, torch.Tensor]:
    parts = []
    max_rank = max(q[0].shape[0], k[0].shape[0], v[0].shape[0])
    for down, up in (q, k, v):
        if down.shape[0] < max_rank:
            down = torch.nn.functional.pad(down, (0, 0, 0, max_rank - down.shape[0]))
            up = torch.nn.functional.pad(up, (0, max_rank - up.shape[1]))
        parts.append((down, up))
    if all(torch.equal(parts[0][0], item[0]) for item in parts[1:]):
        return parts[0][0].contiguous(), torch.cat([item[1] for item in parts], dim=0).contiguous()
    down = torch.cat([item[0] for item in parts], dim=0).contiguous()
    up = torch.zeros(
        sum(item[1].shape[0] for item in parts),
        max_rank * len(parts),
        dtype=parts[0][1].dtype,
    )
    row = col = 0
    for _, part in parts:
        up[row : row + part.shape[0], col : col + max_rank] = part
        row += part.shape[0]
        col += max_rank
    return down, up.contiguous()


def _reorder_adanorm(target: str, down: torch.Tensor, up: torch.Tensor):
    if not (target.endswith("mod.lin") or target.endswith("modulation.lin")):
        return down, up
    from nunchaku.lora.flux.nunchaku_converter import reorder_adanorm_lora_up

    hidden = down.shape[1]
    if hidden == 0 or up.shape[0] % hidden != 0:
        raise ValueError(f"AdaLN LoRA {target} cannot be reordered from shape {tuple(up.shape)}.")
    splits = up.shape[0] // hidden
    if splits not in (3, 6):
        raise ValueError(f"AdaLN LoRA {target} has {splits} chunks; FLUX.1 expects 3 or 6.")
    return down, reorder_adanorm_lora_up(up, splits)


def _resolve(root: nn.Module, path: str) -> nn.Module:
    module: nn.Module | nn.ModuleList = root
    for part in path.split("."):
        if part.isdigit() and isinstance(module, (nn.ModuleList, nn.Sequential)):
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _family(diffusion: nn.Module) -> str:
    if hasattr(diffusion, "txt_mode_blocks") and hasattr(diffusion, "content_mode_blocks"):
        return "flux3"
    blocks = getattr(diffusion, "transformer_blocks", None)
    if blocks is not None and len(blocks) and hasattr(blocks[0], "img_mlp"):
        return "qwen_image21"
    if hasattr(diffusion, "layers") and hasattr(diffusion, "noise_refiner"):
        return "zimage"
    singles = getattr(diffusion, "single_blocks", None)
    if singles is not None and len(singles) and hasattr(singles[0], "qkv_proj"):
        return "flux2"
    if hasattr(diffusion, "input_blocks") and hasattr(diffusion, "label_emb"):
        return "sdxl"
    if hasattr(diffusion, "double_blocks"):
        return "flux1"
    raise RuntimeError("LoRA fusion supports FLUX.1, FLUX.2 Klein, FLUX.3, Qwen-Image 2.1, Z-Image, and SDXL.")


def _conv_delta(module: nn.Conv2d, x: torch.Tensor, down: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    down = down.to(device=x.device, dtype=x.dtype)
    up = up.to(device=x.device, dtype=x.dtype)
    if down.ndim == 2:
        down = down.reshape(down.shape[0], down.shape[1], 1, 1)
    if up.ndim == 2:
        up = up.reshape(up.shape[0], up.shape[1], 1, 1)
    hidden = torch.nn.functional.conv2d(
        x,
        down,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
    )
    return torch.nn.functional.conv2d(hidden, up)


def _install_residual(module: nn.Module) -> None:
    if getattr(module, "_vitoom_lora_wrapped", False):
        return
    original = module.forward

    def forward(x, *args, **kwargs):
        result = original(x, *args, **kwargs)
        extra = getattr(module, "_vitoom_lora", None)
        if extra is None:
            return result
        down, up = extra
        if isinstance(module, nn.Conv2d):
            delta = _conv_delta(module, x, down, up)
        else:
            flat = x.reshape(-1, x.shape[-1])
            down = down.to(device=flat.device, dtype=flat.dtype)
            up = up.to(device=flat.device, dtype=flat.dtype)
            delta = torch.nn.functional.linear(torch.nn.functional.linear(flat, down), up)
            delta = delta.reshape(result.shape)
        return result + delta

    module.forward = forward
    module._vitoom_lora_wrapped = True


def _apply(diffusion: nn.Module, stack: tuple) -> None:
    from nunchaku.lora.common.mixin import _fuse_lowrank_pair
    from nunchaku.models.linear import AWQW4A16Linear, SVDQW4A4Linear

    from vitoom_nunchaku.runtime.lora_models import flux3_pairs, pairs_for, sdxl_pairs

    family = _family(diffusion)
    merged: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for path, strength in stack:
        if family == "flux1":
            mapped = flux_lora_pairs(path)
        elif family == "sdxl":
            mapped = sdxl_pairs(diffusion, path)
        elif family == "flux3":
            mapped = flux3_pairs(path)
        else:
            mapped = pairs_for(family, path)
        for name, (down, up) in mapped.items():
            merged.setdefault(name, []).append((down * strength, up))

    base = diffusion.__dict__.setdefault("_vitoom_lora_base", {})
    svd = residual = 0
    for name, parts in merged.items():
        module = _resolve(diffusion, name)
        down = torch.cat([item[0] for item in parts], dim=0).contiguous()
        up = torch.cat([item[1] for item in parts], dim=1).contiguous()
        if isinstance(module, SVDQW4A4Linear):
            if name not in base:
                base[name] = (module.proj_down.detach().clone(), module.proj_up.detach().clone(), module.rank)
            packed_down, packed_up, new_rank = _fuse_lowrank_pair(base[name][0], base[name][1], down, up)
            module.proj_down = nn.Parameter(packed_down, requires_grad=False)
            module.proj_up = nn.Parameter(packed_up, requires_grad=False)
            module.rank = new_rank
            svd += 1
        elif isinstance(module, AWQW4A16Linear):
            module._vitoom_lora = (
                down.to(device=module.qweight.device, dtype=torch.bfloat16),
                up.to(device=module.qweight.device, dtype=torch.bfloat16),
            )
            residual += 1
        elif isinstance(module, nn.Conv2d):
            if module.groups != 1:
                raise RuntimeError(f"LoRA target {name} is a grouped convolution, which this loader does not support.")
            _install_residual(module)
            module._vitoom_lora = (down, up)
            residual += 1
        elif isinstance(module, nn.Linear):
            _install_residual(module)
            module._vitoom_lora = (down, up)
            residual += 1
        else:
            raise RuntimeError(f"LoRA target {name} is {type(module).__name__}, which cannot take a LoRA.")
    log.info("LoRA touched %d SVDQ layers and %d residual layers.", svd, residual)


def _reset(diffusion: nn.Module) -> None:
    from nunchaku.models.linear import SVDQW4A4Linear

    base = getattr(diffusion, "_vitoom_lora_base", {})
    for name, (down, up, rank) in base.items():
        module = _resolve(diffusion, name)
        if not isinstance(module, SVDQW4A4Linear):
            continue
        module.proj_down = nn.Parameter(down.detach().clone(), requires_grad=False)
        module.proj_up = nn.Parameter(up.detach().clone(), requires_grad=False)
        module.rank = rank
    for module in diffusion.modules():
        if getattr(module, "_vitoom_lora", None) is not None:
            module._vitoom_lora = None
