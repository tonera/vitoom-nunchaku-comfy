"""Map FLUX.2, FLUX.3, Qwen-Image 2.1, Z-Image, and SDXL LoRAs onto this plugin's module paths.

Down is ``[rank, in]`` and up is ``[out, rank]``, the same layout FLUX.1 uses.
SDXL convolutions keep the kohya 4-D layout: down is ``[rank, in, kH, kW]`` and up is ``[out, rank, 1, 1]``.
"""

from __future__ import annotations

import os

import torch
from torch import nn

_CACHE: dict[tuple[str, int, int, str], dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}


def pairs_for(family: str, path: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    stat = os.stat(path)
    key = (os.path.abspath(path), stat.st_mtime_ns, stat.st_size, family)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    state = _normalized(path)
    if family == "flux2":
        pairs = _flux2(state)
    elif family == "qwen_image21":
        pairs = _qwen(state)
    elif family == "zimage":
        pairs = _zimage(state)
    else:
        raise ValueError(f"No LoRA mapper for {family}.")
    if not pairs:
        raise ValueError(f"{path} produced no {family} LoRA layers.")
    _CACHE[key] = pairs
    return pairs


def _normalized(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    from nunchaku.lora.common.mixin import _normalize_lora_format

    return _normalize_lora_format(load_file(path, device="cpu"))


def _grouped(state: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    grouped: dict[str, list[torch.Tensor | None]] = {}
    unused = []
    for key, tensor in state.items():
        if key.endswith(".lora_A.weight"):
            slot = grouped.setdefault(key[: -len(".lora_A.weight")], [None, None])
            slot[0] = tensor
        elif key.endswith(".lora_B.weight"):
            slot = grouped.setdefault(key[: -len(".lora_B.weight")], [None, None])
            slot[1] = tensor
        else:
            unused.append(key)
    if unused:
        raise ValueError("Unsupported LoRA tensors: " + ", ".join(unused[:8]))
    pairs = {}
    for base, pair in grouped.items():
        down, up = pair
        if down is None or up is None:
            raise ValueError(f"LoRA {base} is missing lora_A or lora_B.")
        pairs[base] = (down, up)
    return pairs


def _fuse_qkv(q, k, v) -> tuple[torch.Tensor, torch.Tensor]:
    from vitoom_nunchaku.runtime.lora import _fuse_qkv as fuse

    return fuse(q, k, v)


def _cat_up_gate(up_down, up_up, gate_down, gate_up) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse separate up and gate LoRAs into one ``[up | gate]`` projection."""
    if torch.equal(up_down, gate_down):
        return up_down.contiguous(), torch.cat([up_up, gate_up], dim=0).contiguous()
    down = torch.cat([up_down, gate_down], dim=0).contiguous()
    up = torch.zeros(
        up_up.shape[0] + gate_up.shape[0],
        up_up.shape[1] + gate_up.shape[1],
        dtype=up_up.dtype,
    )
    up[: up_up.shape[0], : up_up.shape[1]] = up_up
    up[up_up.shape[0] :, up_up.shape[1] :] = gate_up
    return down, up.contiguous()


def _swap_gate_up(down: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ComfyUI ``gate_up`` is ``[gate | up]``. This runtime stores ``[up | gate]``."""
    gate, value = up.chunk(2, dim=0)
    return down.contiguous(), torch.cat([value, gate], dim=0).contiguous()


def _split_output(down: torch.Tensor, up: torch.Tensor, first: int):
    if up.shape[0] <= first:
        raise ValueError(f"Cannot split an output of {up.shape[0]} at {first}.")
    return (down.contiguous(), up[:first].contiguous()), (down.contiguous(), up[first:].contiguous())


def _split_input(down: torch.Tensor, up: torch.Tensor, first: int):
    if down.shape[1] <= first:
        raise ValueError(f"Cannot split an input of {down.shape[1]} at {first}.")
    return (down[:, :first].contiguous(), up.contiguous()), (down[:, first:].contiguous(), up.contiguous())


def _flux2(state: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    pairs = _grouped(state)
    if any(key.startswith(("double_blocks.", "single_blocks.")) for key in pairs):
        return _flux2_comfy(pairs)
    if any(key.startswith(("transformer_blocks.", "single_transformer_blocks.")) for key in pairs):
        return _flux2_diffusers(pairs)
    raise ValueError("This file is not a FLUX.2 LoRA.")


def _flux2_comfy(pairs: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    out = {}
    hidden = _flux2_hidden(pairs, "double_blocks.", "img_attn.qkv", "single_blocks.", "linear2")
    direct = {
        "img_attn.qkv": "img_attn.qkv",
        "img_attn.proj": "img_attn.proj",
        "txt_attn.qkv": "txt_attn.qkv",
        "txt_attn.proj": "txt_attn.proj",
        "img_mlp.0": "img_mlp.0",
        "img_mlp.2": "img_mlp.2",
        "txt_mlp.0": "txt_mlp.0",
        "txt_mlp.2": "txt_mlp.2",
    }
    for base, pair in pairs.items():
        if base.startswith("double_blocks."):
            index, _, rest = base[len("double_blocks.") :].partition(".")
            mapped = direct.get(rest)
            if mapped is None:
                raise ValueError(f"Unsupported FLUX.2 LoRA key {base}.")
            out[f"double_blocks.{index}.{mapped}"] = pair
        elif base.startswith("single_blocks."):
            index, _, rest = base[len("single_blocks.") :].partition(".")
            prefix = f"single_blocks.{index}"
            down, up = pair
            if rest == "linear1":
                qkv, mlp = _split_output(down, up, hidden * 3)
                out[f"{prefix}.qkv_proj"] = qkv
                out[f"{prefix}.mlp_fc1"] = mlp
            elif rest == "linear2":
                proj, mlp = _split_input(down, up, hidden)
                out[f"{prefix}.out_proj"] = proj
                out[f"{prefix}.mlp_fc2"] = mlp
            else:
                raise ValueError(f"Unsupported FLUX.2 LoRA key {base}.")
        else:
            raise ValueError(f"Unsupported FLUX.2 LoRA key {base}.")
    return out


def _flux2_diffusers(pairs: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    out = {}
    pending: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    hidden = None
    for base, pair in pairs.items():
        if base.startswith("single_transformer_blocks.") and base.endswith(".attn.to_out"):
            hidden = pair[1].shape[0]
            break
        if base.startswith("transformer_blocks.") and base.endswith(".attn.to_q"):
            hidden = pair[1].shape[0]
            break
    if hidden is None:
        raise ValueError("Cannot read the FLUX.2 hidden size from this LoRA.")
    for base, pair in pairs.items():
        if base.startswith("single_transformer_blocks."):
            index, _, rest = base[len("single_transformer_blocks.") :].partition(".")
            prefix = f"single_blocks.{index}"
            down, up = pair
            if rest == "attn.to_qkv_mlp_proj":
                qkv, mlp = _split_output(down, up, hidden * 3)
                out[f"{prefix}.qkv_proj"] = qkv
                out[f"{prefix}.mlp_fc1"] = mlp
            elif rest == "attn.to_out":
                proj, mlp = _split_input(down, up, hidden)
                out[f"{prefix}.out_proj"] = proj
                out[f"{prefix}.mlp_fc2"] = mlp
            else:
                raise ValueError(f"Unsupported FLUX.2 LoRA key {base}.")
            continue
        if not base.startswith("transformer_blocks."):
            raise ValueError(f"Unsupported FLUX.2 LoRA key {base}.")
        index, _, rest = base[len("transformer_blocks.") :].partition(".")
        prefix = f"double_blocks.{index}"
        mapping = {
            "attn.to_out.0": f"{prefix}.img_attn.proj",
            "attn.to_add_out": f"{prefix}.txt_attn.proj",
            "ff.linear_in": f"{prefix}.img_mlp.0",
            "ff.linear_out": f"{prefix}.img_mlp.2",
            "ff_context.linear_in": f"{prefix}.txt_mlp.0",
            "ff_context.linear_out": f"{prefix}.txt_mlp.2",
        }
        qkv = {
            "attn.to_q": (f"{prefix}.img_attn.qkv", "q"),
            "attn.to_k": (f"{prefix}.img_attn.qkv", "k"),
            "attn.to_v": (f"{prefix}.img_attn.qkv", "v"),
            "attn.add_q_proj": (f"{prefix}.txt_attn.qkv", "q"),
            "attn.add_k_proj": (f"{prefix}.txt_attn.qkv", "k"),
            "attn.add_v_proj": (f"{prefix}.txt_attn.qkv", "v"),
        }
        if rest in mapping:
            out[mapping[rest]] = pair
        elif rest in qkv:
            target, role = qkv[rest]
            pending.setdefault(target, {})[role] = pair
        else:
            raise ValueError(f"Unsupported FLUX.2 LoRA key {base}.")
    for target, parts in pending.items():
        missing = [name for name in ("q", "k", "v") if name not in parts]
        if missing:
            raise ValueError(f"LoRA QKV {target} is missing {missing}.")
        out[target] = _fuse_qkv(parts["q"], parts["k"], parts["v"])
    return out


def _flux2_hidden(pairs, double_prefix, qkv_suffix, single_prefix, linear2_suffix) -> int:
    for base, pair in pairs.items():
        if base.startswith(double_prefix) and base.endswith(qkv_suffix):
            return pair[1].shape[0] // 3
        if base.startswith(single_prefix) and base.endswith(linear2_suffix):
            return pair[1].shape[0]
    raise ValueError("Cannot read the FLUX.2 hidden size from this LoRA.")


def _qwen(state: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    pairs = _grouped(state)
    out = {}
    pending: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    mlp: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for base, pair in pairs.items():
        if base.startswith("transformer_blocks."):
            index, _, rest = base[len("transformer_blocks.") :].partition(".")
            prefix = f"transformer_blocks.{index}"
            if rest == "img_mlp.gate_up":
                out[f"{prefix}.img_mlp.net.0.proj"] = _swap_gate_up(*pair)
            elif rest == "img_mlp.out":
                out[f"{prefix}.img_mlp.net.2"] = pair
            elif rest == "attn.to_out.0":
                out[f"{prefix}.attn.to_out.0"] = pair
            elif rest in ("attn.to_q", "attn.to_k", "attn.to_v"):
                pending.setdefault(f"{prefix}.attn.to_qkv", {})[rest[-1]] = pair
            elif rest in ("img_mlp.proj", "img_mlp.gate_layer"):
                mlp.setdefault(prefix, {})[rest] = pair
            else:
                raise ValueError(f"Unsupported Qwen-Image 2.1 LoRA key {base}.")
            continue
        if base in (
            "modulation.1",
            "time_text_embed.timestep_embedder.linear_1",
            "time_text_embed.timestep_embedder.linear_2",
        ):
            out[base] = pair
            continue
        raise ValueError(f"Unsupported Qwen-Image 2.1 LoRA key {base}.")
    for target, parts in pending.items():
        missing = [name for name in ("q", "k", "v") if name not in parts]
        if missing:
            raise ValueError(f"LoRA QKV {target} is missing {missing}.")
        out[target] = _fuse_qkv(parts["q"], parts["k"], parts["v"])
    for prefix, parts in mlp.items():
        if "img_mlp.proj" not in parts or "img_mlp.gate_layer" not in parts:
            raise ValueError(f"LoRA {prefix} is missing img_mlp.proj or img_mlp.gate_layer.")
        up_down, up_up = parts["img_mlp.proj"]
        gate_down, gate_up = parts["img_mlp.gate_layer"]
        out[f"{prefix}.img_mlp.net.0.proj"] = _cat_up_gate(up_down, up_up, gate_down, gate_up)
    return out


def _zimage(state: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    pairs = _grouped(state)
    out = {}
    pending: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    ffn: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for base, pair in pairs.items():
        head, _, rest = base.partition(".")
        if head not in ("layers", "noise_refiner", "context_refiner"):
            raise ValueError(f"Unsupported Z-Image LoRA key {base}.")
        index, _, rest = rest.partition(".")
        prefix = f"{head}.{index}"
        if rest == "attention.to_out.0":
            out[f"{prefix}.attention.out"] = pair
        elif rest == "adaLN_modulation.0":
            out[f"{prefix}.adaLN_modulation.0"] = pair
        elif rest in ("attention.to_q", "attention.to_k", "attention.to_v"):
            pending.setdefault(f"{prefix}.attention.qkv", {})[rest[-1]] = pair
        elif rest in ("feed_forward.w1", "feed_forward.w2", "feed_forward.w3"):
            ffn.setdefault(prefix, {})[rest] = pair
        else:
            raise ValueError(f"Unsupported Z-Image LoRA key {base}.")
    for target, parts in pending.items():
        missing = [name for name in ("q", "k", "v") if name not in parts]
        if missing:
            raise ValueError(f"LoRA QKV {target} is missing {missing}.")
        out[target] = _fuse_qkv(parts["q"], parts["k"], parts["v"])
    for prefix, parts in ffn.items():
        has_up = "feed_forward.w3" in parts
        has_gate = "feed_forward.w1" in parts
        if has_up != has_gate:
            raise ValueError(f"LoRA {prefix} must include both feed_forward.w3 (up) and feed_forward.w1 (gate).")
        if "feed_forward.w2" not in parts and not has_up:
            raise ValueError(f"LoRA {prefix} has no feed-forward weights.")
        if "feed_forward.w2" in parts:
            out[f"{prefix}.feed_forward.down"] = parts["feed_forward.w2"]
        if has_up and has_gate:
            up_down, up_up = parts["feed_forward.w3"]
            gate_down, gate_up = parts["feed_forward.w1"]
            out[f"{prefix}.feed_forward.proj"] = _cat_up_gate(up_down, up_up, gate_down, gate_up)
    return out


_SDXL_TEXT_PREFIXES = ("lora_te", "text_encoder", "te1.", "te2.")
_SDXL_UNET_PREFIXES = ("unet.", "base_model.model.", "diffusion_model.")
_SDXL_CACHE: dict[tuple, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}


def sdxl_pairs(diffusion: nn.Module, path: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Map one kohya, diffusers, or Comfy SDXL LoRA onto this UNet. Text-encoder keys are dropped."""
    import logging

    stat = os.stat(path)
    signature = (
        len(diffusion.input_blocks),
        len(diffusion.output_blocks),
        tuple(int(v) for v in diffusion.num_res_blocks),
        tuple(int(v) for v in diffusion.channel_mult),
        tuple(_transformer_count(block) for block in list(diffusion.input_blocks) + list(diffusion.output_blocks)),
    )
    key = (os.path.abspath(path), stat.st_mtime_ns, stat.st_size, signature)
    cached = _SDXL_CACHE.get(key)
    if cached is not None:
        return cached

    from safetensors.torch import load_file
    from nunchaku.lora.common.mixin import _normalize_lora_format

    raw = load_file(path, device="cpu")
    if any("hada_" in name or "lokr_" in name or ".lora_mid." in name for name in raw):
        raise ValueError("LyCORIS LoHA/LoKr weights are not supported. Use a standard LoRA with down and up weights.")
    kept = {}
    text = 0
    for name, tensor in raw.items():
        if name.startswith(_SDXL_TEXT_PREFIXES):
            text += 1
            continue
        for prefix in _SDXL_UNET_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        name = name.replace(".lora.down.weight", ".lora_A.weight").replace(".lora.up.weight", ".lora_B.weight")
        kept[name] = tensor
    kept = _normalize_lora_format(kept)
    names = _sdxl_name_map(diffusion)
    rewritten = {}
    missed = []
    for name, tensor in kept.items():
        mapped = _rename_sdxl_key(name, names)
        if mapped is None:
            missed.append(name)
        else:
            rewritten[mapped] = tensor
    pairs = _grouped(rewritten)
    pairs = _fuse_sdxl_qkv(diffusion, pairs)
    _check_sdxl_shapes(diffusion, pairs)
    if not pairs:
        sample = ", ".join(missed[:6])
        raise ValueError(f"No LoRA key matches this SDXL UNet. Unmatched keys start with: {sample}")
    log = logging.getLogger("vitoom_nunchaku")
    if text:
        log.info("Ignored %d SDXL text-encoder LoRA tensors.", text)
    if missed:
        log.warning("SDXL LoRA left %d keys unmatched, including %s.", len(missed), ", ".join(missed[:6]))
    _SDXL_CACHE[key] = pairs
    return pairs


def _transformer_count(block: nn.Module) -> int:
    for child in block.children():
        blocks = getattr(child, "transformer_blocks", None)
        if blocks is not None:
            return len(blocks)
    return 0


def _sdxl_unet_config(diffusion: nn.Module) -> dict:
    channel_mult = [int(value) for value in diffusion.channel_mult]
    num_res_blocks = [int(value) for value in diffusion.num_res_blocks]
    if len(num_res_blocks) != len(channel_mult):
        raise RuntimeError("SDXL UNet num_res_blocks does not match channel_mult.")
    transformer_depth = []
    index = 1
    for level in range(len(channel_mult)):
        for _ in range(num_res_blocks[level]):
            transformer_depth.append(_transformer_count(diffusion.input_blocks[index]))
            index += 1
        if level != len(channel_mult) - 1:
            index += 1
    if index != len(diffusion.input_blocks):
        raise RuntimeError(
            f"SDXL input blocks do not match the resblock count: walked {index}, found {len(diffusion.input_blocks)}."
        )
    deep_first = [_transformer_count(block) for block in diffusion.output_blocks]
    expected = sum(count + 1 for count in num_res_blocks)
    if len(deep_first) != expected:
        raise RuntimeError(f"SDXL output blocks are {len(deep_first)}, expected {expected}.")
    return {
        "num_res_blocks": num_res_blocks,
        "channel_mult": channel_mult,
        "transformer_depth": transformer_depth,
        "transformer_depth_output": list(reversed(deep_first)),
        "transformer_depth_middle": _transformer_count(diffusion.middle_block),
    }


def _sdxl_name_map(diffusion: nn.Module) -> dict[str, str]:
    import comfy.utils
    from nunchaku.models.linear import SVDQW4A4Linear

    mapping = comfy.utils.unet_to_diffusers(_sdxl_unet_config(diffusion))
    comfy_to_diffusers = {}
    for diffusers_key, comfy_key in mapping.items():
        comfy_to_diffusers[comfy_key.rsplit(".", 1)[0]] = diffusers_key.rsplit(".", 1)[0]
    names: dict[str, str] = {}
    for name, module in diffusion.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, SVDQW4A4Linear)):
            _register_sdxl_name(names, name, comfy_to_diffusers)
        if name.endswith(".attn1") and hasattr(module, "to_qkv"):
            for part in ("to_q", "to_k", "to_v"):
                _register_sdxl_name(names, f"{name}.{part}", comfy_to_diffusers)
    return names


def _register_sdxl_name(names: dict[str, str], comfy_path: str, comfy_to_diffusers: dict[str, str]) -> None:
    names[comfy_path.replace(".", "_")] = comfy_path
    diffusers = comfy_to_diffusers.get(comfy_path)
    if diffusers is not None:
        names[diffusers.replace(".", "_")] = comfy_path


def _rename_sdxl_key(key: str, names: dict[str, str]) -> str | None:
    for marker in (".lora_A.", ".lora_B."):
        if marker not in key:
            continue
        base, _, suffix = key.partition(marker)
        if base.startswith("lora_unet_"):
            base = base[len("lora_unet_") :]
        path = names.get(base.replace(".", "_"))
        if path is None:
            return None
        return f"{path}{marker}{suffix}"
    return None


def _fuse_sdxl_qkv(diffusion: nn.Module, pairs: dict[str, tuple[torch.Tensor, torch.Tensor]]):
    pending: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    out = {}
    for base, pair in pairs.items():
        parent, _, leaf = base.rpartition(".")
        if leaf in ("to_q", "to_k", "to_v") and parent.endswith(".attn1") and _has_qkv(diffusion, parent):
            pending.setdefault(parent, {})[leaf[-1]] = pair
        else:
            out[base] = pair
    for parent, parts in pending.items():
        missing = [name for name in ("q", "k", "v") if name not in parts]
        if missing:
            raise ValueError(f"LoRA QKV {parent} is missing {missing}.")
        out[f"{parent}.to_qkv"] = _fuse_qkv(parts["q"], parts["k"], parts["v"])
    return out


def _has_qkv(diffusion: nn.Module, path: str) -> bool:
    module: nn.Module | nn.ModuleList = diffusion
    for part in path.split("."):
        if part.isdigit() and isinstance(module, (nn.ModuleList, nn.Sequential)):
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return hasattr(module, "to_qkv")


def _check_sdxl_shapes(diffusion: nn.Module, pairs: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
    for path, (down, up) in pairs.items():
        module = diffusion
        for part in path.split("."):
            if part.isdigit() and isinstance(module, (nn.ModuleList, nn.Sequential)):
                module = module[int(part)]
            else:
                module = getattr(module, part)
        if isinstance(module, nn.Conv2d):
            kernel = tuple(module.kernel_size)
            if down.ndim == 4:
                ok = down.shape[1] == module.in_channels and tuple(down.shape[2:]) == kernel and up.shape[0] == module.out_channels
            else:
                ok = down.ndim == 2 and down.shape[1] == module.in_channels and kernel == (1, 1) and up.shape[0] == module.out_channels
        else:
            in_features = int(module.in_features)
            out_features = int(module.out_features)
            ok = down.reshape(down.shape[0], -1).shape[1] == in_features and up.shape[0] == out_features
        if not ok:
            raise ValueError(
                f"SDXL LoRA {path} has down {tuple(down.shape)} and up {tuple(up.shape)}, "
                f"which does not match {type(module).__name__}."
            )


def flux3_pairs(path: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Map a diffusers or ComfyUI FLUX.3 LoRA onto this plugin's block paths.

    Quantized blocks store ``mlp_in`` as ``[up | gate]``. Published LoRAs store the
    diffusers order ``[gate | up]``, so those outputs are swapped. ``image_cond`` stays
    a native block and keeps ``[gate | up]``.
    """
    stat = os.stat(path)
    key = (os.path.abspath(path), stat.st_mtime_ns, stat.st_size, "flux3")
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    pairs = _grouped(_normalized(path))
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    pending: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    missed = []
    for base, pair in pairs.items():
        located = _flux3_locate(base)
        if located is None:
            missed.append(base)
            continue
        prefix, inner = located
        role = _FLUX3_QKV.get(inner)
        if role is not None:
            pending.setdefault(f"{prefix}.qkv", {})[role] = pair
            continue
        target = _FLUX3_DIRECT.get(inner)
        if target is None:
            missed.append(base)
            continue
        if target == "mlp_in" and _flux3_quantized(prefix):
            pair = _swap_gate_up(*pair)
        out[f"{prefix}.{target}"] = pair
    for target, parts in pending.items():
        missing = [name for name in ("q", "k", "v") if name not in parts]
        if missing:
            raise ValueError(f"LoRA QKV {target} is missing {missing}.")
        out[target] = _fuse_qkv(parts["q"], parts["k"], parts["v"])
    if not out:
        sample = ", ".join(missed[:6])
        raise ValueError(f"No LoRA key matches this FLUX.3 DiT. Unmatched keys start with: {sample}")
    if missed:
        import logging

        logging.getLogger("vitoom_nunchaku").warning(
            "FLUX.3 LoRA left %d keys unmatched, including %s.", len(missed), ", ".join(missed[:6])
        )
    _CACHE[key] = out
    return out


_FLUX3_QKV = {
    "q_proj": "q",
    "k_proj": "k",
    "v_proj": "v",
    "attn.to_q": "q",
    "attn.to_k": "k",
    "attn.to_v": "v",
}
_FLUX3_DIRECT = {
    "qkv": "qkv",
    "attn.to_qkv": "qkv",
    "attn_out": "attn_out",
    "attn.to_out.0": "attn_out",
    "mlp_in": "mlp_in",
    "ff.linear_in": "mlp_in",
    "mlp_out": "mlp_out",
    "ff.linear_out": "mlp_out",
}


def _flux3_locate(base: str) -> tuple[str, str] | None:
    if base.startswith("mode_blocks."):
        _, mode, index, inner = base.split(".", 3)
        return f"content_mode_blocks.{mode}.{index}", inner
    if base.startswith("single_transformer_blocks."):
        _, index, inner = base.split(".", 2)
        return f"single_blocks.{index}", inner
    for head in ("content_mode_blocks", "txt_mode_blocks", "single_blocks"):
        if not base.startswith(head + "."):
            continue
        rest = base[len(head) + 1 :]
        if head == "content_mode_blocks":
            mode, index, inner = rest.split(".", 2)
            return f"{head}.{mode}.{index}", inner
        index, _, inner = rest.partition(".")
        return f"{head}.{index}", inner
    return None


def _flux3_quantized(prefix: str) -> bool:
    """``image_cond`` keeps the native ``[gate | up]`` MLP. The other blocks are swapped on load."""
    if prefix.startswith("content_mode_blocks.image_cond."):
        return False
    return prefix.startswith(("txt_mode_blocks.", "single_blocks.", "content_mode_blocks.image."))
