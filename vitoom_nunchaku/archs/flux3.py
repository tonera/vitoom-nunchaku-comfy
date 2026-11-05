"""FLUX.3 Image Klein: replace quantized linear layers inside the native ComfyUI network.

Checked against diffusers ``Flux3Transformer2DModel`` in the local nunchaku environment:

- Stream modulation linear layers output ``[shift | scale | gate]``, matching ComfyUI ``Modulation``, so the halves are not swapped.
- ``norm_out.linear`` is ``[scale | shift]``. ComfyUI ``LastLayer`` is ``[shift | scale]``, so the two halves are swapped on load.
- Quantized blocks store ``ff.linear_in`` with ``swiglu_order=up_gate``, and forward uses ``fused_swiglu_mlp``.
- ``image_cond`` is not quantized. It stays diffusers / ComfyUI ``[gate | up]`` and keeps the native Block.
  Its Q/K/V are three separate weights. The network is built with ``fused_qkv=True``, and q, k, v are concatenated into ``qkv.weight`` on load.
"""

from __future__ import annotations

import re

import torch
from torch import nn

from vitoom_nunchaku.archs.base import QuantConfig, register_spec
from vitoom_nunchaku.runtime.layers import fused_swiglu, make_w4a4, restore_fp32_qk_norms

_EXACT = {
    "context_embedder.weight": "txt_in.weight",
    "context_embedder.bias": "txt_in.bias",
    "x_embedders.image.weight": "emb_in.image.weight",
    "x_embedders.image.bias": "emb_in.image.bias",
    "x_embedders.image_cond.weight": "emb_in.image_cond.weight",
    "x_embedders.image_cond.bias": "emb_in.image_cond.bias",
    "timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
    "timestep_embedder.linear_1.bias": "time_in.in_layer.bias",
    "timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
    "timestep_embedder.linear_2.bias": "time_in.out_layer.bias",
    "target_timestep_embedder.linear_1.weight": "ref_time_in.in_layer.weight",
    "target_timestep_embedder.linear_1.bias": "ref_time_in.in_layer.bias",
    "target_timestep_embedder.linear_2.weight": "ref_time_in.out_layer.weight",
    "target_timestep_embedder.linear_2.bias": "ref_time_in.out_layer.bias",
    "proj_out.weight": "final_layer.image.linear.weight",
    "proj_out.bias": "final_layer.image.linear.bias",
}

# Longer prefixes come first so attn.to_q does not consume attn.to_qkv.
_INNER = (
    ("attn.to_out.0", "attn_out"),
    ("attn.to_qkv", "qkv"),
    ("attn.norm_q", "norm.query_norm"),
    ("attn.norm_k", "norm.key_norm"),
    ("ff.linear_in", "mlp_in"),
    ("ff.linear_out", "mlp_out"),
    ("attn.to_q", "q_proj"),
    ("attn.to_k", "k_proj"),
    ("attn.to_v", "v_proj"),
)

_MOD_RE = re.compile(
    r"^(early_stream_modulations|single_stream_modulations)\.([^.]+)\.linear\.(weight|bias)$"
)
_REQUIRED_BLOCK_ATTRS = ("pre_norm", "norm", "mlp_in", "mlp_out", "attn_out")


def swap_scale_shift(weight: torch.Tensor) -> torch.Tensor:
    """Turn ``[scale | shift]`` into ComfyUI ``LastLayer`` ``[shift | scale]``."""
    scale, shift = weight.chunk(2, dim=0)
    return torch.cat((shift, scale), dim=0)


class NunchakuFlux3Block(nn.Module):
    """Replacement forward for a quantized Block. Semantics match ``comfy.ldm.flux3.model.Block``."""

    def __init__(self, source: nn.Module, qkv, attn_out, mlp_in, mlp_out):
        super().__init__()
        self.comfy_attention = source.comfy_attention
        self.num_heads = source.num_heads
        self.fused_qkv = True
        self.norm = source.norm
        self.pre_norm = source.pre_norm
        self.qkv = qkv
        self.attn_out = attn_out
        self.mlp_in = mlp_in
        self.mlp_out = mlp_out

    def forward(self, x, spans, pe, transformer_options={}):
        import comfy.ldm.modules.attention as attention
        from comfy.ldm.flux.layers import modulated_norm

        batch, length, _ = x.shape
        first = spans[0][2]
        x_mod = modulated_norm(x, self.pre_norm, first.scale, first.shift)
        for start, end, mod in spans[1:]:
            x_mod[:, start:end] = modulated_norm(x[:, start:end], self.pre_norm, mod.scale, mod.shift)

        query, key, value = self.qkv(x_mod).chunk(3, dim=-1)
        mlp = fused_swiglu(x_mod, self.mlp_in, self.mlp_out)
        del x_mod
        query, key = self.norm(
            query.view(batch, length, self.num_heads, -1),
            key.view(batch, length, self.num_heads, -1),
            pe,
        )
        query = attention.AttentionTensorContainer(query.flatten(2))
        key = attention.AttentionTensorContainer(key.flatten(2))
        value = attention.AttentionTensorContainer(value)
        attn = attention.optimized_attention(
            query,
            key,
            value,
            self.num_heads,
            transformer_options=transformer_options,
            preferred_attention=self.comfy_attention,
        )
        # SVDQ modules have no linear.weight, so they cannot go through linear_input_act.
        out = self.attn_out(attn) + mlp
        for start, end, mod in spans:
            x[:, start:end].addcmul_(out[:, start:end], mod.gate)
        return x


def _require_block(block: nn.Module, where: str) -> None:
    missing = [name for name in _REQUIRED_BLOCK_ATTRS if not hasattr(block, name)]
    if missing or not hasattr(block, "qkv"):
        if not hasattr(block, "qkv"):
            missing.append("qkv")
        raise RuntimeError(
            f"ComfyUI Flux3 {where} is missing {missing}. "
            "This plugin is written against ComfyUI 0.44.0 comfy.ldm.flux3.model.Block."
        )


def _replace_quantized_block(block: nn.Module, hidden: int, mlp_hidden: int, qcfg: QuantConfig, dtype: torch.dtype):
    _require_block(block, "quantized Block")
    device = next(block.parameters()).device
    rank = qcfg.rank
    precision = qcfg.precision
    return NunchakuFlux3Block(
        block,
        qkv=make_w4a4(hidden, 3 * hidden, rank, precision, dtype, device),
        attn_out=make_w4a4(hidden, hidden, rank, precision, dtype, device),
        mlp_in=make_w4a4(hidden, 2 * mlp_hidden, rank, precision, dtype, device),
        mlp_out=make_w4a4(mlp_hidden, hidden, rank, precision, dtype, device),
    )


def _map_inner(inner: str) -> tuple[str, str]:
    for src, dst in _INNER:
        if inner == src:
            return dst, ""
        prefix = src + "."
        if inner.startswith(prefix):
            return dst, inner[len(src):]
    raise KeyError(inner)


def _block_location(kind: str, rest: str) -> tuple[str, str]:
    if kind == "txt_mode_blocks":
        index, inner = rest.split(".", 1)
        return f"txt_mode_blocks.{index}", inner
    if kind == "single_transformer_blocks":
        index, inner = rest.split(".", 1)
        return f"single_blocks.{index}", inner
    if kind == "mode_blocks":
        mode, index, inner = rest.split(".", 2)
        return f"content_mode_blocks.{mode}.{index}", inner
    raise KeyError(kind)


class Flux3Spec:
    model_class = "NunchakuFlux3Transformer2DModel"
    supported_precisions = ("nvfp4", "int4")
    min_nunchaku = "1.4.1"

    def validate(self, diffusers_config: dict, qcfg: QuantConfig) -> None:
        class_name = str(diffusers_config.get("_class_name", ""))
        if "Flux3" not in class_name:
            raise ValueError(f"config._class_name is not Flux3: {class_name!r}")
        head_dim = int(diffusers_config.get("attention_head_dim", 0))
        if head_dim != 128:
            raise ValueError(f"The fused kernel only supports attention_head_dim=128, checkpoint has {head_dim}.")
        mlp_ratio = float(diffusers_config.get("mlp_ratio", 3.0))
        if mlp_ratio != 3.0:
            raise ValueError(f"ComfyUI Flux3 Block fixes mlp_ratio at 3.0, checkpoint has {mlp_ratio}.")
        if int(diffusers_config.get("in_channels", 128)) != 128:
            raise ValueError("ComfyUI Flux3 fixes in_channels at 128.")
        if int(diffusers_config.get("joint_attention_dim", 20480)) != 20480:
            raise ValueError("ComfyUI Flux3 fixes the text width at 20480.")
        if qcfg.swiglu_order != "up_gate":
            raise ValueError(
                f"FLUX.3 quantized-block ff.linear_in must be [up | gate] (swiglu_order='up_gate'), got {qcfg.swiglu_order!r}."
            )
        weight = qcfg.raw.get("weight") or {}
        expected_group = 16 if qcfg.precision == "nvfp4" else 64
        group_size = int(weight.get("group_size", expected_group))
        if group_size != expected_group:
            raise ValueError(f"FLUX.3 {qcfg.precision} group_size must be {expected_group}, got {group_size}.")

    def comfy_model_config(self, diffusers_config: dict):
        import comfy.supported_models as supported_models

        heads = int(diffusers_config["num_attention_heads"])
        head_dim = int(diffusers_config["attention_head_dim"])
        unet_config = {
            "image_model": "flux3",
            "hidden_size": heads * head_dim,
            "num_heads": heads,
            "fused_qkv": True,
            "flow_map": bool(diffusers_config.get("flow_map", False)),
            "depth": int(diffusers_config["num_layers"]),
            "depth_single_blocks": int(diffusers_config["num_single_layers"]),
        }
        return supported_models.Flux3(unet_config)

    def patch_model(self, diffusion_model: nn.Module, qcfg: QuantConfig, dtype: torch.dtype) -> None:
        block0 = diffusion_model.txt_mode_blocks[0]
        # Head dimension comes from RMSNorm, not from Linear.weight, which lazy loading may leave empty.
        hidden = int(block0.num_heads * block0.norm.query_norm.weight.shape[0])
        mlp_hidden = int(hidden * 3.0)
        groups = (
            diffusion_model.txt_mode_blocks,
            diffusion_model.content_mode_blocks["image"],
            diffusion_model.single_blocks,
        )
        for blocks in groups:
            for index, block in enumerate(blocks):
                blocks[index] = _replace_quantized_block(block, hidden, mlp_hidden, qcfg, dtype)
        restore_fp32_qk_norms(diffusion_model)
        if not hasattr(diffusion_model.content_mode_blocks["image_cond"][0], "qkv"):
            raise RuntimeError("The image_cond branch should stay a native ComfyUI Block with fused_qkv=True.")

    def convert_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        converted: dict[str, torch.Tensor] = {}
        qkv_parts: dict[str, dict[str, torch.Tensor]] = {}
        for key, tensor in state_dict.items():
            if key in _EXACT:
                converted[_EXACT[key]] = tensor
                continue
            mod = _MOD_RE.match(key)
            if mod:
                converted[f"{mod.group(1)}.{mod.group(2)}.lin.{mod.group(3)}"] = tensor
                continue
            if key in ("norm_out.linear.weight", "norm_out.linear.bias"):
                suffix = key.rsplit(".", 1)[1]
                converted[f"final_layer.image.adaLN_modulation.1.{suffix}"] = swap_scale_shift(tensor)
                continue
            if key.startswith("txt_mode_blocks.") or key.startswith("single_transformer_blocks.") or key.startswith("mode_blocks."):
                kind, rest = key.split(".", 1)
                prefix, inner = _block_location(kind, rest)
                try:
                    dst, suffix = _map_inner(inner)
                except KeyError as exc:
                    raise KeyError(f"Cannot map {key}") from exc
                if dst in ("q_proj", "k_proj", "v_proj"):
                    if suffix != ".weight":
                        raise KeyError(f"{key} must be a standalone Q/K/V weight and cannot carry {suffix}")
                    qkv_parts.setdefault(f"{prefix}.qkv", {})[dst[0]] = tensor
                    continue
                converted[f"{prefix}.{dst}{suffix}"] = tensor
                continue
            raise KeyError(f"Checkpoint key has no matching ComfyUI module: {key}")

        for prefix, parts in qkv_parts.items():
            if set(parts) != {"q", "k", "v"}:
                raise KeyError(f"{prefix} needs to_q, to_k, and to_v together, got {sorted(parts)}")
            converted[prefix + ".weight"] = torch.cat((parts["q"], parts["k"], parts["v"]), dim=0)
        return converted


register_spec(Flux3Spec())
