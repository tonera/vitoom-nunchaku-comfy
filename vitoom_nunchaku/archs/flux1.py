"""FLUX.1 dev, depth, canny, fill, Kontext, and Krea: native ComfyUI Flux with nunchaku W4A4 linears.

The checkpoint is the V1 key layout (``qkv_proj``, ``mlp_fc1``, ``lora_down``). AdaLN projections
are AWQ W4A16 and already include the +1 scale, so the modulation output subtracts 1 before
ComfyUI adds it back. GELU MLPs use the fused kernel because INT4 shifts the activation.
"""

from __future__ import annotations

import torch
from torch import nn

from vitoom_nunchaku.archs.base import QuantConfig, register_spec
from vitoom_nunchaku.runtime.layers import fused_gelu, make_awq, make_w4a4

_SVD_SUFFIXES = (
    (".lora_down", ".proj_down"),
    (".lora_up", ".proj_up"),
    (".smooth_orig", ".smooth_factor_orig"),
    (".smooth", ".smooth_factor"),
)

# Longer names first so qkv_proj_context is not consumed by qkv_proj.
_DOUBLE = (
    ("qkv_proj_context", "txt_attn.qkv"),
    ("out_proj_context", "txt_attn.proj"),
    ("mlp_context_fc1", "txt_mlp.0"),
    ("mlp_context_fc2", "txt_mlp.2"),
    ("qkv_proj", "img_attn.qkv"),
    ("out_proj", "img_attn.proj"),
    ("mlp_fc1", "img_mlp.0"),
    ("mlp_fc2", "img_mlp.2"),
    ("norm1_context.linear", "txt_mod.lin"),
    ("norm1.linear", "img_mod.lin"),
    ("norm_added_q", "txt_attn.norm.query_norm"),
    ("norm_added_k", "txt_attn.norm.key_norm"),
    ("norm_q", "img_attn.norm.query_norm"),
    ("norm_k", "img_attn.norm.key_norm"),
)

_SINGLE = (
    ("qkv_proj", "qkv"),
    ("mlp_fc1", "mlp_fc1"),
    ("mlp_fc2", "mlp_fc2"),
    ("out_proj", "out_proj"),
    ("norm.linear", "modulation.lin"),
    ("norm_q", "norm.query_norm"),
    ("norm_k", "norm.key_norm"),
)

_BASIC = {
    "x_embedder.weight": "img_in.weight",
    "x_embedder.bias": "img_in.bias",
    "context_embedder.weight": "txt_in.weight",
    "context_embedder.bias": "txt_in.bias",
    "proj_out.weight": "final_layer.linear.weight",
    "proj_out.bias": "final_layer.linear.bias",
    "time_text_embed.timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
    "time_text_embed.timestep_embedder.linear_1.bias": "time_in.in_layer.bias",
    "time_text_embed.timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
    "time_text_embed.timestep_embedder.linear_2.bias": "time_in.out_layer.bias",
    "time_text_embed.text_embedder.linear_1.weight": "vector_in.in_layer.weight",
    "time_text_embed.text_embedder.linear_1.bias": "vector_in.in_layer.bias",
    "time_text_embed.text_embedder.linear_2.weight": "vector_in.out_layer.weight",
    "time_text_embed.text_embedder.linear_2.bias": "vector_in.out_layer.bias",
    "time_text_embed.guidance_embedder.linear_1.weight": "guidance_in.in_layer.weight",
    "time_text_embed.guidance_embedder.linear_1.bias": "guidance_in.in_layer.bias",
    "time_text_embed.guidance_embedder.linear_2.weight": "guidance_in.out_layer.weight",
    "time_text_embed.guidance_embedder.linear_2.bias": "guidance_in.out_layer.bias",
}


def _swap_halves(weight: torch.Tensor) -> torch.Tensor:
    """Diffusers ``norm_out`` is ``[scale | shift]``. ComfyUI ``LastLayer`` is ``[shift | scale]``."""
    first, second = weight.chunk(2, dim=0)
    return torch.cat((second, first), dim=0)


def _linear_bias(linear: nn.Module) -> bool:
    # Aimdo leaves bias as None and records the flag instead.
    flag = getattr(linear, "comfy_need_lazy_init_bias", None)
    if flag is not None and getattr(linear, "weight", None) is None:
        return bool(flag)
    return linear.bias is not None


def _svdq(linear: nn.Module, qcfg: QuantConfig, dtype: torch.dtype, unsigned: bool = False) -> nn.Module:
    module = make_w4a4(
        linear.in_features,
        linear.out_features,
        qcfg.rank,
        qcfg.precision,
        dtype,
        device=torch.device("meta"),
        bias=_linear_bias(linear),
    )
    if unsigned and qcfg.precision != "nvfp4":
        module.act_unsigned = True
    return module


class QuantGeluMLP(nn.Module):
    """Names stay ``0`` and ``2`` so the state dict matches ComfyUI's ``Sequential`` MLP."""

    def __init__(self, fc1: nn.Module, fc2: nn.Module):
        super().__init__()
        self.add_module("0", fc1)
        self.add_module("2", fc2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return fused_gelu(hidden_states, self._modules["0"], self._modules["2"])


class QuantModulation(nn.Module):
    """AWQ AdaLN. The packed layout is per-channel ``[shift, scale, gate, ...]``, and scale already includes +1."""

    def __init__(self, linear: nn.Module, double: bool):
        super().__init__()
        self.lin = linear
        self.is_double = double
        self.multiplier = 6 if double else 3

    def forward(self, vec: torch.Tensor):
        from comfy.ldm.flux.layers import ModulationOut

        if vec.ndim == 2:
            vec = vec[:, None, :]
        flat = torch.nn.functional.silu(vec.reshape(-1, vec.shape[-1]))
        raw = self.lin(flat)
        # AWQ has no low-rank branch. A FLUX.1 LoRA on this projection is a residual in the same layout.
        extra = getattr(self.lin, "_vitoom_lora", None)
        if extra is not None:
            down, up = extra
            raw = raw + torch.nn.functional.linear(torch.nn.functional.linear(flat, down), up)
        raw = raw.view(vec.shape[0], vec.shape[1], -1, self.multiplier)
        parts = [raw[..., index] for index in range(self.multiplier)]
        parts[1] = parts[1] - 1
        if self.multiplier == 6:
            parts[4] = parts[4] - 1
        first = ModulationOut(parts[0], parts[1], parts[2])
        second = ModulationOut(parts[3], parts[4], parts[5]) if self.multiplier == 6 else None
        return first, second


class QuantSingleStreamBlock(nn.Module):
    """ComfyUI fuses QKV and MLP into ``linear1``/``linear2``. This checkpoint keeps four GEMMs."""

    def __init__(self, source: nn.Module, qkv, mlp_fc1, out_proj, mlp_fc2):
        super().__init__()
        self.comfy_attention = source.comfy_attention
        self.hidden_size = source.hidden_size
        self.num_heads = source.num_heads
        self.norm = source.norm
        self.pre_norm = source.pre_norm
        self.modulation = source.modulation
        self.qkv = qkv
        self.mlp_fc1 = mlp_fc1
        self.out_proj = out_proj
        self.mlp_fc2 = mlp_fc2

    def forward(self, x, vec, pe, attn_mask=None, modulation_dims=None, transformer_options={}):
        from comfy.ldm.flux.layers import apply_mod, modulated_norm
        from comfy.ldm.flux.math import attention
        from comfy.ldm.modules.attention import AttentionTensorContainer

        if self.modulation:
            mod, _ = self.modulation(vec)
        else:
            mod = vec

        transformer_patches = transformer_options.get("patches", {})
        extra_options = transformer_options.copy()
        modulated = modulated_norm(x, self.pre_norm, mod.scale, mod.shift, modulation_dims)
        qkv = self.qkv(modulated)
        mlp = fused_gelu(modulated, self.mlp_fc1, self.mlp_fc2)
        del modulated

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del qkv
        q, k = self.norm(q, k, v)

        if "attn1_patch" in transformer_patches:
            for patch in transformer_patches["attn1_patch"]:
                out = patch(q, k, v, pe=pe, attn_mask=attn_mask, extra_options=extra_options)
                q, k, v, pe, attn_mask = out.get("q", q), out.get("k", k), out.get("v", v), out.get("pe", pe), out.get("attn_mask", attn_mask)

        q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
        attn = attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options, preferred_attention=self.comfy_attention)
        del q, k, v

        if "attn1_output_patch" in transformer_patches:
            for patch in transformer_patches["attn1_output_patch"]:
                attn = patch(attn, extra_options)

        output = self.out_proj(attn) + mlp
        x = x + apply_mod(output, mod.gate, None, modulation_dims)
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x


def _rename_suffix(key: str) -> str:
    for source, target in _SVD_SUFFIXES:
        if key.endswith(source):
            return key[: -len(source)] + target
    return key


def _rename_block(inner: str, pairs: tuple[tuple[str, str], ...]) -> str | None:
    for source, target in pairs:
        if inner == source or inner.startswith(source + "."):
            return target + inner[len(source):]
    return None


class Flux1Spec:
    model_class = "NunchakuFluxTransformer2dModel"
    supported_precisions = ("nvfp4", "int4")
    min_nunchaku = "1.4.1"

    def validate(self, diffusers_config: dict, qcfg: QuantConfig) -> None:
        class_name = str(diffusers_config.get("_class_name", ""))
        if class_name != "FluxTransformer2DModel":
            raise ValueError(f"config._class_name is not FluxTransformer2DModel: {class_name!r}")
        head_dim = int(diffusers_config.get("attention_head_dim", 0))
        heads = int(diffusers_config.get("num_attention_heads", 0))
        if head_dim != 128 or heads != 24:
            raise ValueError(f"FLUX.1 expects 24 heads of size 128, got heads={heads} head_dim={head_dim}.")
        if int(diffusers_config.get("num_layers", 0)) != 19 or int(diffusers_config.get("num_single_layers", 0)) != 38:
            raise ValueError("FLUX.1 expects 19 double blocks and 38 single blocks.")
        if int(diffusers_config.get("joint_attention_dim", 0)) != 4096:
            raise ValueError("FLUX.1 text width must be 4096.")
        if not diffusers_config.get("guidance_embeds", False):
            raise ValueError("This loader expects guidance embeddings. Schnell checkpoints are not wired up.")
        packed_in = int(diffusers_config.get("in_channels", 0))
        if packed_in % 4 != 0:
            raise ValueError(f"in_channels must be a multiple of 4 (patch 2), got {packed_in}.")
        weight = qcfg.raw.get("weight") or {}
        expected_group = 16 if qcfg.precision == "nvfp4" else 64
        group_size = int(weight.get("group_size", expected_group))
        if group_size != expected_group:
            raise ValueError(f"FLUX.1 {qcfg.precision} group_size must be {expected_group}, got {group_size}.")

    def comfy_model_config(self, diffusers_config: dict):
        import comfy.supported_models

        config = diffusers_config
        packed_in = int(config["in_channels"])
        packed_out = config.get("out_channels") or packed_in
        unet_config = {
            "image_model": "flux",
            "in_channels": packed_in // 4,
            "out_channels": int(packed_out) // 4,
            "vec_in_dim": int(config.get("pooled_projection_dim", 768)),
            "context_in_dim": int(config["joint_attention_dim"]),
            "hidden_size": int(config["num_attention_heads"]) * int(config["attention_head_dim"]),
            "mlp_ratio": 4.0,
            "num_heads": int(config["num_attention_heads"]),
            "depth": int(config["num_layers"]),
            "depth_single_blocks": int(config["num_single_layers"]),
            "axes_dim": [16, 56, 56],
            "theta": 10000,
            "patch_size": 2,
            "qkv_bias": True,
            "guidance_embed": True,
            "txt_ids_dims": [],
        }
        return comfy.supported_models.Flux(unet_config)

    def patch_model(self, diffusion_model: nn.Module, qcfg: QuantConfig, dtype: torch.dtype) -> None:
        for block in diffusion_model.double_blocks:
            block.img_mod = QuantModulation(_awq(block.img_mod.lin, dtype), double=True)
            block.txt_mod = QuantModulation(_awq(block.txt_mod.lin, dtype), double=True)
            block.img_attn.qkv = _svdq(block.img_attn.qkv, qcfg, dtype)
            block.img_attn.proj = _svdq(block.img_attn.proj, qcfg, dtype)
            block.txt_attn.qkv = _svdq(block.txt_attn.qkv, qcfg, dtype)
            block.txt_attn.proj = _svdq(block.txt_attn.proj, qcfg, dtype)
            block.img_mlp = _gelu_mlp(block.img_mlp, qcfg, dtype)
            block.txt_mlp = _gelu_mlp(block.txt_mlp, qcfg, dtype)

        replacements = []
        for block in diffusion_model.single_blocks:
            hidden = block.hidden_size
            mlp_hidden = block.mlp_hidden_dim
            qkv = make_w4a4(hidden, hidden * 3, qcfg.rank, qcfg.precision, dtype, device=torch.device("meta"), bias=True)
            mlp_fc1 = make_w4a4(hidden, mlp_hidden, qcfg.rank, qcfg.precision, dtype, device=torch.device("meta"), bias=True)
            out_proj = make_w4a4(hidden, hidden, qcfg.rank, qcfg.precision, dtype, device=torch.device("meta"), bias=True)
            mlp_fc2 = make_w4a4(mlp_hidden, hidden, qcfg.rank, qcfg.precision, dtype, device=torch.device("meta"), bias=True)
            if qcfg.precision != "nvfp4":
                mlp_fc2.act_unsigned = True
            block.modulation = QuantModulation(_awq(block.modulation.lin, dtype), double=False)
            replacements.append(QuantSingleStreamBlock(block, qkv, mlp_fc1, out_proj, mlp_fc2))
        for index, replacement in enumerate(replacements):
            diffusion_model.single_blocks[index] = replacement

    def convert_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        converted = {}
        for key, tensor in state_dict.items():
            if key in _BASIC:
                converted[_BASIC[key]] = tensor
                continue
            if key.startswith("norm_out.linear."):
                suffix = key.rsplit(".", 1)[-1]
                converted[f"final_layer.adaLN_modulation.1.{suffix}"] = _swap_halves(tensor)
                continue
            if key.startswith("transformer_blocks."):
                index = key.split(".")[1]
                inner = ".".join(key.split(".")[2:])
                renamed = _rename_block(inner, _DOUBLE)
                if renamed is None:
                    raise KeyError(f"Unmapped FLUX.1 double-block key {key}")
                converted[f"double_blocks.{index}.{_rename_suffix(renamed)}"] = tensor
                continue
            if key.startswith("single_transformer_blocks."):
                index = key.split(".")[1]
                inner = ".".join(key.split(".")[2:])
                renamed = _rename_block(inner, _SINGLE)
                if renamed is None:
                    raise KeyError(f"Unmapped FLUX.1 single-block key {key}")
                converted[f"single_blocks.{index}.{_rename_suffix(renamed)}"] = tensor
                continue
            raise KeyError(f"Unmapped FLUX.1 key {key}")
        return converted


def _awq(linear: nn.Module, dtype: torch.dtype) -> nn.Module:
    return make_awq(
        linear.in_features,
        linear.out_features,
        dtype,
        device=torch.device("meta"),
        bias=_linear_bias(linear),
        group_size=64,
    )


def _gelu_mlp(mlp: nn.Module, qcfg: QuantConfig, dtype: torch.dtype) -> QuantGeluMLP:
    return QuantGeluMLP(_svdq(mlp[0], qcfg, dtype), _svdq(mlp[2], qcfg, dtype, unsigned=True))


register_spec(Flux1Spec())
