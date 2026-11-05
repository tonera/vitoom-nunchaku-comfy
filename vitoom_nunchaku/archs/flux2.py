"""FLUX.2 Klein: replace quantized linear layers inside the native ComfyUI network.

Checked against nunchaku ``NunchakuFlux2Transformer2DModel`` and ComfyUI ``comfy.ldm.flux.model.Flux``:

- Double-stream blocks keep the native forward. Q/K/V is one ``to_qkv`` / ``to_added_qkv``, and the MLP stays ``[gate | up]``.
- Single-stream blocks are split in the checkpoint into ``qkv_proj``, ``mlp_fc1``, ``out_proj``, and ``mlp_fc2``, and cannot be packed back into ComfyUI ``linear1`` / ``linear2``.
- ``norm_out.linear`` is ``[scale | shift]``. ComfyUI ``LastLayer`` is ``[shift | scale]``, so the two halves are swapped on load.
- Klein sets ``guidance_embeds`` to false, so the network has no ``guidance_in``.
"""

from __future__ import annotations

import re

import torch
from torch import nn

from vitoom_nunchaku.archs.base import QuantConfig, register_spec
from vitoom_nunchaku.runtime.layers import make_w4a4

_EXACT = {
    "context_embedder.weight": "txt_in.weight",
    "x_embedder.weight": "img_in.weight",
    "time_guidance_embed.timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
    "time_guidance_embed.timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
    "double_stream_modulation_img.linear.weight": "double_stream_modulation_img.lin.weight",
    "double_stream_modulation_txt.linear.weight": "double_stream_modulation_txt.lin.weight",
    "single_stream_modulation.linear.weight": "single_stream_modulation.lin.weight",
    "proj_out.weight": "final_layer.linear.weight",
}

# Longer prefixes come first so attn.to_q does not consume attn.to_qkv, and ff does not consume ff_context.
_DOUBLE = (
    ("attn.to_out.0", "img_attn.proj"),
    ("attn.to_qkv", "img_attn.qkv"),
    ("attn.norm_q", "img_attn.norm.query_norm"),
    ("attn.norm_k", "img_attn.norm.key_norm"),
    ("attn.to_added_qkv", "txt_attn.qkv"),
    ("attn.to_add_out", "txt_attn.proj"),
    ("attn.norm_added_q", "txt_attn.norm.query_norm"),
    ("attn.norm_added_k", "txt_attn.norm.key_norm"),
    ("ff_context.linear_in", "txt_mlp.0"),
    ("ff_context.linear_out", "txt_mlp.2"),
    ("ff.linear_in", "img_mlp.0"),
    ("ff.linear_out", "img_mlp.2"),
)
_SINGLE = (
    ("attn.qkv_proj", "qkv_proj"),
    ("attn.mlp_fc1", "mlp_fc1"),
    ("attn.out_proj", "out_proj"),
    ("attn.mlp_fc2", "mlp_fc2"),
    ("attn.norm_q", "norm.query_norm"),
    ("attn.norm_k", "norm.key_norm"),
)
_BLOCK_RE = re.compile(r"^(transformer_blocks|single_transformer_blocks)\.(\d+)\.(.+)$")


def swap_scale_shift(weight: torch.Tensor) -> torch.Tensor:
    """Turn ``[scale | shift]`` into ComfyUI ``LastLayer`` ``[shift | scale]``."""
    scale, shift = weight.chunk(2, dim=0)
    return torch.cat((shift, scale), dim=0)


class NunchakuFlux2SingleBlock(nn.Module):
    """Forward of ComfyUI ``SingleStreamBlock``, with the two GEMMs split to match the checkpoint."""

    def __init__(self, source: nn.Module, qkv_proj, mlp_fc1, out_proj, mlp_fc2):
        super().__init__()
        self.comfy_attention = source.comfy_attention
        self.hidden_size = source.hidden_size
        self.num_heads = source.num_heads
        self.mlp_act = source.mlp_act
        self.norm = source.norm
        self.pre_norm = source.pre_norm
        self.modulation = source.modulation
        self.qkv_proj = qkv_proj
        self.mlp_fc1 = mlp_fc1
        self.out_proj = out_proj
        self.mlp_fc2 = mlp_fc2

    def forward(self, x, vec, pe, attn_mask=None, modulation_dims=None, transformer_options={}):
        from comfy.ldm.flux.layers import apply_mod, modulated_norm
        from comfy.ldm.flux.math import attention
        from comfy.ldm.modules.attention import AttentionTensorContainer

        if self.modulation is not None:
            mod, _ = self.modulation(vec)
        else:
            mod = vec

        transformer_patches = transformer_options.get("patches", {})
        extra_options = transformer_options.copy()
        x_mod = modulated_norm(x, self.pre_norm, mod.scale, mod.shift, modulation_dims)
        qkv = self.qkv_proj(x_mod)
        mlp = self.mlp_act(self.mlp_fc1(x_mod))
        del x_mod

        query, key, value = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del qkv
        query, key = self.norm(query, key, value)

        if "attn1_patch" in transformer_patches:
            for patch in transformer_patches["attn1_patch"]:
                out = patch(query, key, value, pe=pe, attn_mask=attn_mask, extra_options=extra_options)
                query = out.get("q", query)
                key = out.get("k", key)
                value = out.get("v", value)
                pe = out.get("pe", pe)
                attn_mask = out.get("attn_mask", attn_mask)

        query = AttentionTensorContainer(query)
        key = AttentionTensorContainer(key)
        value = AttentionTensorContainer(value)
        attn = attention(
            query,
            key,
            value,
            pe=pe,
            mask=attn_mask,
            transformer_options=transformer_options,
            preferred_attention=self.comfy_attention,
        )
        del query, key, value

        if "attn1_output_patch" in transformer_patches:
            for patch in transformer_patches["attn1_output_patch"]:
                attn = patch(attn, extra_options)

        output = self.out_proj(attn) + self.mlp_fc2(mlp)
        x = x + apply_mod(output, mod.gate, None, modulation_dims)
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x


def _rewrite(rest: str, table: tuple[tuple[str, str], ...]) -> str | None:
    for source, dest in table:
        if rest == source or rest.startswith(source + "."):
            return dest + rest[len(source) :]
    return None


def _reject_foreign_key(key: str) -> None:
    if "guidance_embedder" in key or ".guidance_in." in key:
        raise KeyError(f"{key} carries a guidance embed. This loader only accepts Klein (guidance_embeds is false).")
    if ".attn.to_qkv_mlp_proj" in key or key.endswith(".attn.to_out") or ".attn.to_out.weight" in key:
        raise KeyError(
            f"{key} is a fused single-stream layer from Comfy native NVFP4. "
            "Use ComfyUI's own UNETLoader, or switch to nunchaku's split keys (qkv_proj / mlp_fc1 / out_proj / mlp_fc2)."
        )
    if any(token in key for token in (".attn.to_q.", ".attn.to_k.", ".attn.to_v.")):
        raise KeyError(f"{key} is separate Q/K/V. FLUX.2 Klein nunchaku checkpoints use fused to_qkv.")


class Flux2Spec:
    model_class = "NunchakuFlux2Transformer2DModel"
    supported_precisions = ("nvfp4", "int4")
    min_nunchaku = "1.4.1"

    def validate(self, diffusers_config: dict, qcfg: QuantConfig) -> None:
        class_name = str(diffusers_config.get("_class_name", ""))
        if "Flux2" not in class_name:
            raise ValueError(f"config._class_name is not FLUX.2: {class_name!r}")
        head_dim = int(diffusers_config.get("attention_head_dim", 128))
        if head_dim != 128:
            raise ValueError(f"Q/K RMSNorm and RoPE are implemented for head_dim=128, checkpoint has {head_dim}.")
        axes = [int(dim) for dim in diffusers_config.get("axes_dims_rope", (32, 32, 32, 32))]
        if sum(axes) != head_dim:
            raise ValueError(f"axes_dims_rope must sum to head_dim, got {axes}.")
        if int(diffusers_config.get("patch_size", 1)) != 1:
            raise ValueError("ComfyUI FLUX.2 img_in only accepts patch_size=1.")
        if int(diffusers_config.get("in_channels", 0)) != 128:
            raise ValueError(f"FLUX.2 latent channels are 128, checkpoint in_channels is {diffusers_config.get('in_channels')!r}.")
        mlp_ratio = float(diffusers_config.get("mlp_ratio", 3.0))
        if mlp_ratio != 3.0:
            raise ValueError(f"FLUX.2 mlp_ratio must be 3, checkpoint has {mlp_ratio}.")
        if diffusers_config.get("guidance_embeds", False):
            raise ValueError("This loader is built for Klein, so guidance_embeds must be false.")
        if qcfg.swiglu_order is not None:
            raise ValueError(
                f"FLUX.2 SwiGLU is [gate | up], so the checkpoint must not set swiglu_order, got {qcfg.swiglu_order!r}."
            )
        weight = qcfg.raw.get("weight") or {}
        expected_group = 16 if qcfg.precision == "nvfp4" else 64
        group_size = int(weight.get("group_size", expected_group))
        if group_size != expected_group:
            raise ValueError(f"FLUX.2 {qcfg.precision} group_size must be {expected_group}, got {group_size}.")

    def comfy_model_config(self, diffusers_config: dict):
        import comfy.supported_models as supported_models

        heads = int(diffusers_config["num_attention_heads"])
        head_dim = int(diffusers_config.get("attention_head_dim", 128))
        hidden = int(diffusers_config.get("num_attention_heads")) * head_dim
        out_channels = diffusers_config.get("out_channels")
        unet_config = {
            "image_model": "flux2",
            "in_channels": int(diffusers_config.get("in_channels", 128)),
            "out_channels": int(out_channels if out_channels is not None else 128),
            "vec_in_dim": None,
            "context_in_dim": int(diffusers_config["joint_attention_dim"]),
            "hidden_size": hidden,
            "mlp_ratio": float(diffusers_config.get("mlp_ratio", 3.0)),
            "num_heads": heads,
            "depth": int(diffusers_config["num_layers"]),
            "depth_single_blocks": int(diffusers_config["num_single_layers"]),
            "axes_dim": [int(dim) for dim in diffusers_config.get("axes_dims_rope", (32, 32, 32, 32))],
            "theta": int(diffusers_config.get("rope_theta", 2000)),
            "patch_size": int(diffusers_config.get("patch_size", 1)),
            "qkv_bias": False,
            "guidance_embed": False,
            "txt_ids_dims": [3],
            "global_modulation": True,
            "mlp_silu_act": True,
            "ops_bias": False,
            "default_ref_method": "index",
            "ref_index_scale": 10.0,
            "yak_mlp": False,
            "txt_norm": False,
        }
        return supported_models.Flux2(unet_config)

    def patch_model(self, diffusion_model: nn.Module, qcfg: QuantConfig, dtype: torch.dtype) -> None:
        hidden = int(diffusion_model.hidden_size)
        mlp_hidden = int(hidden * float(diffusion_model.params.mlp_ratio))
        rank = qcfg.rank
        precision = qcfg.precision
        for block in diffusion_model.double_blocks:
            device = next(block.parameters()).device
            block.img_attn.qkv = make_w4a4(hidden, hidden * 3, rank, precision, dtype, device)
            block.img_attn.proj = make_w4a4(hidden, hidden, rank, precision, dtype, device)
            block.txt_attn.qkv = make_w4a4(hidden, hidden * 3, rank, precision, dtype, device)
            block.txt_attn.proj = make_w4a4(hidden, hidden, rank, precision, dtype, device)
            block.img_mlp[0] = make_w4a4(hidden, mlp_hidden * 2, rank, precision, dtype, device)
            block.img_mlp[2] = make_w4a4(mlp_hidden, hidden, rank, precision, dtype, device)
            block.txt_mlp[0] = make_w4a4(hidden, mlp_hidden * 2, rank, precision, dtype, device)
            block.txt_mlp[2] = make_w4a4(mlp_hidden, hidden, rank, precision, dtype, device)
        for index, block in enumerate(diffusion_model.single_blocks):
            device = next(block.parameters()).device
            diffusion_model.single_blocks[index] = NunchakuFlux2SingleBlock(
                block,
                make_w4a4(hidden, hidden * 3, rank, precision, dtype, device),
                make_w4a4(hidden, mlp_hidden * 2, rank, precision, dtype, device),
                make_w4a4(hidden, hidden, rank, precision, dtype, device),
                make_w4a4(mlp_hidden, hidden, rank, precision, dtype, device),
            )

    def convert_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        converted: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            _reject_foreign_key(key)
            if key in _EXACT:
                converted[_EXACT[key]] = tensor
                continue
            if key == "norm_out.linear.weight":
                converted["final_layer.adaLN_modulation.1.weight"] = swap_scale_shift(tensor)
                continue
            match = _BLOCK_RE.match(key)
            if match is None:
                raise KeyError(f"Checkpoint key has no matching ComfyUI module: {key}")
            kind, index, rest = match.groups()
            table = _DOUBLE if kind == "transformer_blocks" else _SINGLE
            rewritten = _rewrite(rest, table)
            if rewritten is None:
                raise KeyError(f"Checkpoint key has no matching ComfyUI module: {key}")
            dest = "double_blocks" if kind == "transformer_blocks" else "single_blocks"
            converted[f"{dest}.{index}.{rewritten}"] = tensor
        return converted


register_spec(Flux2Spec())
