"""Qwen-Image 2.1: replace quantized linear layers inside the native ComfyUI network.

Checkpoint key names already match ComfyUI. Only two structures differ from the native network:

- Attention is one fused ``attn.to_qkv``. Native ComfyUI uses separate ``to_q/to_k/to_v``.
- The MLP is diffusers SwiGLU: ``img_mlp.net.0.proj`` is ``[up | gate]`` and ``img_mlp.net.2`` is the down projection.
  Native ComfyUI ``gate_up`` is ``[gate | up]``, so the two weight halves cannot be swapped and dropped in.

FunControl is a branch attached at sampling time. Its weights are not in this DiT checkpoint, so it is left unchanged.
Prefix KV cache still goes through ComfyUI's attention callback. Quantized layers only compute Q/K/V.
"""

from __future__ import annotations

import torch
from torch import nn

from vitoom_nunchaku.archs.base import QuantConfig, register_spec
from vitoom_nunchaku.runtime.layers import fused_swiglu, make_w4a4

_TOP_LEVEL = (
    "img_in.",
    "modulation.",
    "norm_out.",
    "proj_out.",
    "time_text_embed.",
    "txt_in.",
)
_BLOCK_MODULES = (
    "attn.norm_q.",
    "attn.norm_k.",
    "attn.to_out.0.",
    "attn.to_qkv.",
    "img_mlp.net.0.proj.",
    "img_mlp.net.2.",
)


class NunchakuQwenImage21Attention(nn.Module):
    """Forward of ComfyUI ``Attention``, with Q/K/V sliced from one fused linear layer."""

    def __init__(self, source: nn.Module, qkv: nn.Module, attn_out: nn.Module):
        super().__init__()
        self.comfy_attention = source.comfy_attention
        self.heads = source.heads
        self.to_qkv = qkv
        self.to_out = nn.ModuleList([attn_out])
        self.norm_q = source.norm_q
        self.norm_k = source.norm_k

    def forward(self, x, pe, attn_fn, prefix_len, transformer_options={}):
        import comfy.model_management
        import comfy.ops
        import comfy.quant_ops
        from comfy.ldm.flux.math import apply_rope1
        from comfy.ldm.modules.attention import AttentionTensorContainer

        batch, length, _ = x.shape
        query, key, value = (part.view(batch, length, self.heads, -1) for part in self.to_qkv(x).chunk(3, dim=-1))
        patches = transformer_options.get("patches", {}).get("attn1_patch", [])
        if comfy.model_management.in_training or patches:
            query, key = self.norm_q(query), self.norm_k(key)
            if patches:
                query, key, value, pe = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), pe.transpose(1, 2)
                extra_options = {**transformer_options, "img_slice": [prefix_len, length]}
                for patch in patches:
                    out = patch(query, key, value, pe=pe, attn_mask=None, extra_options=extra_options)
                    query, key, value, pe = out.get("q", query), out.get("k", key), out.get("v", value), out.get("pe", pe)
                query = query.transpose(1, 2).contiguous()
                key = key.transpose(1, 2).contiguous()
                value = value.transpose(1, 2).contiguous()
                pe = pe.transpose(1, 2).contiguous()
            query = apply_rope1(query, pe)
            key = apply_rope1(key, pe)
        else:
            q_scale, _, q_stream = comfy.ops.cast_bias_weight(self.norm_q, query, offloadable=True)
            k_scale, _, k_stream = comfy.ops.cast_bias_weight(self.norm_k, key, offloadable=True)
            query, key = comfy.quant_ops.ck.rms_rope(query, key, pe, q_scale, k_scale, self.norm_q.eps)
            comfy.ops.uncast_bias_weight(self.norm_q, q_scale, None, q_stream)
            comfy.ops.uncast_bias_weight(self.norm_k, k_scale, None, k_stream)
        query = AttentionTensorContainer(query)
        key = AttentionTensorContainer(key)
        value = AttentionTensorContainer(value)
        return self.to_out[0](attn_fn(query, key, value, self.heads, preferred_attention=self.comfy_attention))


class _UpProjection(nn.Module):
    def __init__(self, proj: nn.Module):
        super().__init__()
        self.proj = proj


class NunchakuQwenImage21MLP(nn.Module):
    """Key names stay ``net.0.proj`` and ``net.2``. Forward uses fused SwiGLU."""

    def __init__(self, proj: nn.Module, down: nn.Module):
        super().__init__()
        self.net = nn.ModuleList([_UpProjection(proj), nn.Identity(), down])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return fused_swiglu(hidden_states, self.net[0].proj, self.net[2])


def _linear_out_features(linear: nn.Module) -> int:
    weight = getattr(linear, "weight", None)
    if isinstance(weight, torch.Tensor):
        return int(weight.shape[0])
    out_features = getattr(linear, "out_features", None)
    if out_features is None:
        raise RuntimeError("Cannot read the output size from the linear layer.")
    return int(out_features)


def _allowed_key(key: str) -> bool:
    if key.startswith(_TOP_LEVEL):
        return True
    parts = key.split(".", 2)
    if len(parts) < 3 or parts[0] != "transformer_blocks":
        return False
    return parts[2].startswith(_BLOCK_MODULES)


class QwenImage21Spec:
    model_class = "NunchakuQwenImage21Transformer2DModel"
    supported_precisions = ("nvfp4", "int4")
    min_nunchaku = "1.4.1"

    def validate(self, diffusers_config: dict, qcfg: QuantConfig) -> None:
        class_name = str(diffusers_config.get("_class_name", ""))
        if "QwenImage21" not in class_name:
            raise ValueError(f"config._class_name is not Qwen-Image 2.1: {class_name!r}")
        head_dim = int(diffusers_config.get("attention_head_dim", 0))
        if head_dim != 128:
            raise ValueError(f"Q/K RMSNorm and RoPE are implemented for head_dim=128, checkpoint has {head_dim}.")
        axes = [int(dim) for dim in diffusers_config.get("axes_dims_rope", ())]
        if sum(axes) != head_dim:
            raise ValueError(f"axes_dims_rope must sum to head_dim, got {axes}.")
        if int(diffusers_config.get("patch_size", 1)) != 1:
            raise ValueError("ComfyUI Qwen-Image 2.1 img_in only accepts patch_size=1.")
        if diffusers_config.get("causal_condition", True) is not True:
            raise ValueError("ComfyUI modulation splits text/reference (t=0) from the target image, but checkpoint causal_condition is not true.")
        mlp_ratio = diffusers_config.get("mlp_ratio", 3)
        if int(mlp_ratio) != mlp_ratio:
            raise ValueError(f"ComfyUI mlp_ratio must be an integer, checkpoint has {mlp_ratio}.")
        hidden = int(diffusers_config["num_attention_heads"]) * head_dim
        mlp_hidden = hidden * int(mlp_ratio)
        if mlp_hidden % 128 != 0:
            raise ValueError(f"Fused SwiGLU requires the MLP hidden size to be a multiple of 128, got {mlp_hidden}.")
        # Published Qwen-Image 2.1 files omit swiglu_order. The fused MLP is [up | gate]; any other value is rejected.
        if qcfg.swiglu_order not in (None, "up_gate"):
            raise ValueError(
                f"img_mlp.net.0.proj must be [up | gate] (swiglu_order='up_gate'), got {qcfg.swiglu_order!r}."
            )
        weight = qcfg.raw.get("weight") or {}
        expected_group = 16 if qcfg.precision == "nvfp4" else 64
        group_size = int(weight.get("group_size", expected_group))
        if group_size != expected_group:
            raise ValueError(f"Qwen-Image 2.1 {qcfg.precision} group_size must be {expected_group}, got {group_size}.")

    def comfy_model_config(self, diffusers_config: dict):
        import comfy.supported_models as supported_models

        unet_config = {
            "image_model": "qwen_image21",
            "in_channels": int(diffusers_config.get("in_channels", 64)),
            "out_channels": int(diffusers_config.get("out_channels", 64)),
            "num_layers": int(diffusers_config["num_layers"]),
            "attention_head_dim": int(diffusers_config["attention_head_dim"]),
            "num_attention_heads": int(diffusers_config["num_attention_heads"]),
            "context_in_dim": int(diffusers_config.get("context_in_dim", 4096)),
            "mlp_ratio": int(diffusers_config.get("mlp_ratio", 3)),
            "axes_dims_rope": tuple(int(dim) for dim in diffusers_config.get("axes_dims_rope", (16, 56, 56))),
            "eps": float(diffusers_config.get("eps", 1e-6)),
            "fused_mlp": True,
        }
        return supported_models.QwenImage21(unet_config)

    def patch_model(self, diffusion_model: nn.Module, qcfg: QuantConfig, dtype: torch.dtype) -> None:
        block0 = diffusion_model.transformer_blocks[0]
        hidden = int(block0.attn.heads * block0.attn.norm_q.weight.shape[0])
        # ComfyUI builds with fused_mlp=True. Confirm the native MLP is still there, then switch to checkpoint module names.
        # With dynamic VRAM, Linear.weight starts as None and the output size is stored on out_features.
        if not hasattr(block0.img_mlp, "gate_up") and not hasattr(block0.img_mlp, "proj"):
            raise RuntimeError(
                "ComfyUI Qwen-Image 2.1 img_mlp is not the expected SwiGLU. "
                "This plugin is written against ComfyUI 0.44.0 comfy.ldm.qwen_image21.model."
            )
        if hasattr(block0.img_mlp, "gate_up"):
            mlp_hidden = _linear_out_features(block0.img_mlp.gate_up) // 2
        else:
            mlp_hidden = _linear_out_features(block0.img_mlp.proj)
        if mlp_hidden % 128 != 0:
            raise RuntimeError(f"Fused SwiGLU requires the MLP hidden size to be a multiple of 128, got {mlp_hidden}.")
        rank = qcfg.rank
        precision = qcfg.precision
        for block in diffusion_model.transformer_blocks:
            device = next(block.parameters()).device
            block.attn = NunchakuQwenImage21Attention(
                block.attn,
                make_w4a4(hidden, 3 * hidden, rank, precision, dtype, device),
                make_w4a4(hidden, hidden, rank, precision, dtype, device),
            )
            block.img_mlp = NunchakuQwenImage21MLP(
                make_w4a4(hidden, 2 * mlp_hidden, rank, precision, dtype, device),
                make_w4a4(mlp_hidden, hidden, rank, precision, dtype, device),
            )

    def convert_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        converted: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            if ".attn.to_q." in key or ".attn.to_k." in key or ".attn.to_v." in key:
                raise KeyError(
                    f"{key} is separate Q/K/V. This loader expects fused attn.to_qkv. "
                    "For Comfy native NVFP4 (to_q/to_k/to_v), use ComfyUI's own UNETLoader."
                )
            if any(token in key for token in (".img_mlp.gate_up.", ".img_mlp.out.", ".img_mlp.proj.", ".img_mlp.gate_layer.")):
                raise KeyError(f"{key} is a native ComfyUI MLP key. Nunchaku checkpoints use img_mlp.net.0.proj and img_mlp.net.2.")
            if not _allowed_key(key):
                raise KeyError(f"Checkpoint key has no matching ComfyUI module: {key}")
            converted[key] = tensor
        return converted


register_spec(QwenImage21Spec())
