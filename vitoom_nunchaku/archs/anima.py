"""Anima: native ComfyUI Cosmos/Anima network with quantized linear layers replaced by W8A8.

The nunchaku single file contains only the DiT. The LLM adapter lives in text_conditioner
beside it, and its keys are already the ComfyUI ``llm_adapter`` keys without that prefix.
Attention and the MLP still use ComfyUI's forward. W8A8 packs itself on the first CUDA forward.
"""

from __future__ import annotations

import logging
import os
import re

import torch
from torch import nn

from vitoom_nunchaku.archs.base import QuantConfig, register_spec
from vitoom_nunchaku.runtime.checkpoint import read_header
from vitoom_nunchaku.runtime.layers import make_w8a8

log = logging.getLogger("vitoom_nunchaku")

_NUNCHAKU_STEM = re.compile(r"^svdq-[a-z0-9]+_r\d+-(.+)-nunchaku$", re.IGNORECASE)
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^patch_embed\.proj\."), "x_embedder.proj.1."),
    (re.compile(r"^time_embed\.t_embedder\."), "t_embedder.1."),
    (re.compile(r"^time_embed\.norm\."), "t_embedding_norm."),
    (re.compile(r"^proj_out\."), "final_layer.linear."),
    (re.compile(r"^norm_out\.linear_1\."), "final_layer.adaln_modulation.1."),
    (re.compile(r"^norm_out\.linear_2\."), "final_layer.adaln_modulation.2."),
    (re.compile(r"^transformer_blocks\.(\d+)\.attn1\.to_q\."), r"blocks.\1.self_attn.q_proj."),
    (re.compile(r"^transformer_blocks\.(\d+)\.attn1\.to_k\."), r"blocks.\1.self_attn.k_proj."),
    (re.compile(r"^transformer_blocks\.(\d+)\.attn1\.to_v\."), r"blocks.\1.self_attn.v_proj."),
    (re.compile(r"^transformer_blocks\.(\d+)\.attn1\.to_out\.0\."), r"blocks.\1.self_attn.output_proj."),
    (re.compile(r"^transformer_blocks\.(\d+)\.attn1\.norm_q\."), r"blocks.\1.self_attn.q_norm."),
    (re.compile(r"^transformer_blocks\.(\d+)\.attn1\.norm_k\."), r"blocks.\1.self_attn.k_norm."),
    (re.compile(r"^transformer_blocks\.(\d+)\.attn2\.to_q\."), r"blocks.\1.cross_attn.q_proj."),
    (re.compile(r"^transformer_blocks\.(\d+)\.attn2\.to_k\."), r"blocks.\1.cross_attn.k_proj."),
    (re.compile(r"^transformer_blocks\.(\d+)\.attn2\.to_v\."), r"blocks.\1.cross_attn.v_proj."),
    (re.compile(r"^transformer_blocks\.(\d+)\.attn2\.to_out\.0\."), r"blocks.\1.cross_attn.output_proj."),
    (re.compile(r"^transformer_blocks\.(\d+)\.attn2\.norm_q\."), r"blocks.\1.cross_attn.q_norm."),
    (re.compile(r"^transformer_blocks\.(\d+)\.attn2\.norm_k\."), r"blocks.\1.cross_attn.k_norm."),
    (re.compile(r"^transformer_blocks\.(\d+)\.ff\.net\.0\.proj\."), r"blocks.\1.mlp.layer1."),
    (re.compile(r"^transformer_blocks\.(\d+)\.ff\.net\.2\."), r"blocks.\1.mlp.layer2."),
    (re.compile(r"^transformer_blocks\.(\d+)\.norm1\.linear_1\."), r"blocks.\1.adaln_modulation_self_attn.1."),
    (re.compile(r"^transformer_blocks\.(\d+)\.norm1\.linear_2\."), r"blocks.\1.adaln_modulation_self_attn.2."),
    (re.compile(r"^transformer_blocks\.(\d+)\.norm2\.linear_1\."), r"blocks.\1.adaln_modulation_cross_attn.1."),
    (re.compile(r"^transformer_blocks\.(\d+)\.norm2\.linear_2\."), r"blocks.\1.adaln_modulation_cross_attn.2."),
    (re.compile(r"^transformer_blocks\.(\d+)\.norm3\.linear_1\."), r"blocks.\1.adaln_modulation_mlp.1."),
    (re.compile(r"^transformer_blocks\.(\d+)\.norm3\.linear_2\."), r"blocks.\1.adaln_modulation_mlp.2."),
)


def _convert_key(key: str) -> str:
    if "to_qkv" in key or "joint_qkv" in key:
        raise KeyError(f"Anima only accepts separate to_q/to_k/to_v and cannot map {key}")
    for pattern, repl in _RULES:
        converted, count = pattern.subn(repl, key, count=1)
        if count:
            return converted
    raise KeyError(f"Cannot map Anima weight key: {key}")


def _core_name(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = _NUNCHAKU_STEM.match(stem)
    name = match.group(1) if match else stem
    return name.lower().replace("_", "-")


def _adapter_pairs(keys: list[str]) -> list[tuple[str, str]]:
    if any("llm_adapter." in key for key in keys):
        return [(key, key[key.find("llm_adapter.") :]) for key in keys if "llm_adapter." in key]
    if "embed.weight" in keys and any(key.startswith("blocks.0.self_attn.") for key in keys):
        return [(key, "llm_adapter." + key) for key in keys]
    raise RuntimeError("This file does not contain an Anima LLM adapter.")


def _header_adapter_pairs(path: str) -> list[tuple[str, str]] | None:
    try:
        _metadata, tensors = read_header(path)
    except Exception:
        return None
    try:
        return _adapter_pairs(list(tensors))
    except RuntimeError:
        return None


def _search_dirs(near: str) -> list[str]:
    dirs = [
        near,
        os.path.expanduser("~/models/Anima/split_files/diffusion_models"),
    ]
    try:
        import folder_paths

        dirs.extend(folder_paths.get_folder_paths("diffusion_models"))
    except Exception:
        pass
    seen = set()
    ordered = []
    for directory in dirs:
        absolute = os.path.abspath(directory)
        if absolute in seen or not os.path.isdir(absolute):
            continue
        seen.add(absolute)
        ordered.append(absolute)
    return ordered


def find_llm_adapter_path(nunchaku_path: str) -> str:
    """The quantized DiT does not include the LLM adapter. Prefer text_conditioner in the same directory."""
    real = os.path.realpath(nunchaku_path)
    sibling = os.path.join(os.path.dirname(real), "text_conditioner", "diffusion_pytorch_model.safetensors")
    if os.path.isfile(sibling) and _header_adapter_pairs(sibling):
        return sibling
    core = _core_name(real)
    for directory in _search_dirs(os.path.dirname(real)):
        candidate = os.path.join(directory, core + ".safetensors")
        if os.path.isfile(candidate) and _header_adapter_pairs(candidate):
            return candidate
    raise RuntimeError(
        f"{os.path.basename(nunchaku_path)} does not contain an LLM adapter. "
        "Keep text_conditioner/diffusion_pytorch_model.safetensors in the nunchaku directory, "
        f"or place the full weights in diffusion_models with the same name as {core}."
    )


def _linear_io(linear: nn.Module) -> tuple[int, int]:
    in_features = getattr(linear, "in_features", None)
    out_features = getattr(linear, "out_features", None)
    if in_features is not None and out_features is not None:
        return int(in_features), int(out_features)
    weight = getattr(linear, "weight", None)
    if isinstance(weight, torch.Tensor):
        return int(weight.shape[1]), int(weight.shape[0])
    raise RuntimeError("Cannot read input and output sizes from the linear layer.")


def _module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    return torch.device("meta")


def _replace_linear(parent: nn.Module, name: str, rank: int, dtype: torch.dtype, device: torch.device) -> None:
    in_features, out_features = _linear_io(getattr(parent, name))
    setattr(parent, name, make_w8a8(in_features, out_features, rank, dtype, device))


class AnimaSpec:
    model_class = "NunchakuAnimaTransformer3DModel"
    supported_precisions = ("int8",)
    min_nunchaku = "1.4.1"
    min_rank = 0

    def validate(self, diffusers_config: dict, qcfg: QuantConfig) -> None:
        class_name = str(diffusers_config.get("_class_name", ""))
        if "Cosmos" not in class_name and "Anima" not in class_name:
            raise ValueError(f"config._class_name is not Cosmos/Anima: {class_name!r}")
        if qcfg.raw.get("joint_qkv"):
            raise ValueError("Anima quantized files must use separate to_q/to_k/to_v. This checkpoint has joint_qkv=true.")
        if int(diffusers_config.get("attention_head_dim", 0)) != 128:
            raise ValueError("Anima Q/K RMSNorm and RoPE are implemented for head_dim=128.")
        if [int(size) for size in diffusers_config.get("patch_size", ())] != [1, 2, 2]:
            raise ValueError(f"Anima patch_size must be [1, 2, 2], got {diffusers_config.get('patch_size')!r}.")
        if int(diffusers_config.get("in_channels", 0)) != 16 or int(diffusers_config.get("out_channels", 0)) != 16:
            raise ValueError("Anima input and output channels must be 16.")
        if diffusers_config.get("concat_padding_mask", False) is not True:
            raise ValueError("Anima patch embed requires concat_padding_mask.")
        if diffusers_config.get("use_crossattn_projection", False):
            raise ValueError("This checkpoint includes a cross-attention projection. This loader has no branch for that.")
        if diffusers_config.get("extra_pos_embed_type") not in (None, "null"):
            raise ValueError("This checkpoint includes extra positional embeddings. This loader has no branch for that.")
        if float(diffusers_config.get("mlp_ratio", 0)) != 4.0:
            raise ValueError(f"Anima mlp_ratio must be 4, got {diffusers_config.get('mlp_ratio')!r}.")
        if int(diffusers_config.get("adaln_lora_dim", 0)) != 256:
            raise ValueError(f"Anima adaln_lora_dim must be 256, got {diffusers_config.get('adaln_lora_dim')!r}.")
        if int(diffusers_config.get("text_embed_dim", 0)) != 1024:
            raise ValueError(f"Anima text dimension must be 1024, got {diffusers_config.get('text_embed_dim')!r}.")
        if int(diffusers_config.get("num_layers", 0)) < 1 or int(diffusers_config.get("num_attention_heads", 0)) < 1:
            raise ValueError("Checkpoint is missing num_layers or num_attention_heads.")

    def comfy_model_config(self, diffusers_config: dict):
        import comfy.supported_models as supported_models

        heads = int(diffusers_config["num_attention_heads"])
        head_dim = int(diffusers_config["attention_head_dim"])
        patch = [int(size) for size in diffusers_config["patch_size"]]
        max_size = [int(size) for size in diffusers_config.get("max_size", (128, 240, 240))]
        rope = [float(scale) for scale in diffusers_config.get("rope_scale", (1.0, 4.0, 4.0))]
        unet_config = {
            "image_model": "anima",
            "max_img_h": max_size[1],
            "max_img_w": max_size[2],
            "max_frames": max_size[0],
            "in_channels": int(diffusers_config["in_channels"]),
            "out_channels": int(diffusers_config["out_channels"]),
            "patch_spatial": patch[1],
            "patch_temporal": patch[0],
            "model_channels": heads * head_dim,
            "concat_padding_mask": True,
            "crossattn_emb_channels": int(diffusers_config["text_embed_dim"]),
            "pos_emb_cls": "rope3d",
            "pos_emb_learnable": True,
            "pos_emb_interpolation": "crop",
            "min_fps": 1,
            "max_fps": 30,
            "use_adaln_lora": True,
            "adaln_lora_dim": int(diffusers_config["adaln_lora_dim"]),
            "num_blocks": int(diffusers_config["num_layers"]),
            "num_heads": heads,
            "mlp_ratio": float(diffusers_config["mlp_ratio"]),
            "extra_per_block_abs_pos_emb": False,
            "rope_h_extrapolation_ratio": rope[1],
            "rope_w_extrapolation_ratio": rope[2],
            "rope_t_extrapolation_ratio": rope[0],
            "extra_h_extrapolation_ratio": 1.0,
            "extra_w_extrapolation_ratio": 1.0,
            "extra_t_extrapolation_ratio": 1.0,
            "rope_enable_fps_modulation": False,
        }
        return supported_models.Anima(unet_config)

    def restore_runtime_buffers(self, diffusion_model: nn.Module) -> None:
        """Building on meta and then calling to_empty drops non-persistent buffers computed in __init__."""
        pos = diffusion_model.pos_embedder
        head_dim = int(diffusion_model.model_channels // diffusion_model.num_heads)
        dim_h = head_dim // 6 * 2
        dim_t = head_dim - 2 * dim_h
        spatial = torch.arange(0, dim_h, 2, device=pos.dim_spatial_range.device, dtype=torch.float32)
        temporal = torch.arange(0, dim_t, 2, device=pos.dim_temporal_range.device, dtype=torch.float32)
        pos.dim_spatial_range.copy_(spatial[: dim_h // 2] / dim_h)
        pos.dim_temporal_range.copy_(temporal[: dim_t // 2] / dim_t)

        rotary = diffusion_model.llm_adapter.rotary_emb
        head = int(rotary.inv_freq.shape[0] * 2)
        positions = torch.arange(0, head, 2, device=rotary.inv_freq.device, dtype=torch.float32)
        inv_freq = 1.0 / (rotary.rope_theta ** (positions / head))
        rotary.inv_freq.copy_(inv_freq)

    def patch_model(self, diffusion_model: nn.Module, qcfg: QuantConfig, dtype: torch.dtype) -> None:
        if dtype != torch.bfloat16:
            raise ValueError(f"Anima W8A8 only accepts bfloat16, got {dtype}. Set dtype to auto or bfloat16.")
        device = _module_device(diffusion_model)
        rank = qcfg.rank
        for block in diffusion_model.blocks:
            for parent, names in (
                (block.self_attn, ("q_proj", "k_proj", "v_proj", "output_proj")),
                (block.cross_attn, ("q_proj", "k_proj", "v_proj", "output_proj")),
                (block.mlp, ("layer1", "layer2")),
            ):
                for name in names:
                    _replace_linear(parent, name, rank, dtype, device)

    def convert_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {_convert_key(key): tensor for key, tensor in state_dict.items()}

    def supplement_state_dict(self, state_dict: dict[str, torch.Tensor], path: str) -> dict[str, torch.Tensor]:
        if any(key.startswith("llm_adapter.") for key in state_dict):
            return state_dict
        source = find_llm_adapter_path(path)
        from safetensors import safe_open

        with safe_open(source, framework="pt", device="cpu") as handle:
            pairs = _adapter_pairs(list(handle.keys()))
            for source_key, target_key in pairs:
                state_dict[target_key] = handle.get_tensor(source_key)
        log.info("LLM adapter loaded from %s (%s tensors)", source, len(pairs))
        return state_dict


register_spec(AnimaSpec())
