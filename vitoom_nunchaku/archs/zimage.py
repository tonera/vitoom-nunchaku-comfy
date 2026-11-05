"""Z-Image Turbo: ComfyUI NextDiT with quantized linear layers replaced by nunchaku SVDQ."""

from __future__ import annotations

import re

import torch
from torch import nn

from vitoom_nunchaku.archs.base import QuantConfig, register_spec
from vitoom_nunchaku.runtime.layers import fused_swiglu, make_w4a4

_FP_OVERRIDE = re.compile(
    r"^(layers|noise_refiner|context_refiner)\.\d+\.feed_forward\.net\.(0\.proj|2)$"
)
# Same constants as nunchaku ZImageTurboFFNDownGuard. The guard is skipped when the file lists any fp_overrides.
_TURBO_FFN_GUARD_CHANNEL = 10078
_TURBO_FFN_GUARD_MAX_ABS = 4096.0
_KEY_REPLACEMENTS = (
    ("feed_forward.net.0.proj", "feed_forward.proj"),
    ("feed_forward.net.2", "feed_forward.down"),
    ("attention.to_out.0", "attention.out"),
    ("attention.to_qkv", "attention.qkv"),
    ("attention.norm_q", "attention.q_norm"),
    ("attention.norm_k", "attention.k_norm"),
)


def clamp_turbo_ffn_activation(hidden: torch.Tensor) -> torch.Tensor:
    """Clamp the activation channel that overflows W4A4 on released Z-Image Turbo checkpoints."""
    if hidden.shape[-1] <= _TURBO_FFN_GUARD_CHANNEL:
        return hidden
    channel = hidden[..., _TURBO_FFN_GUARD_CHANNEL]
    if not torch.any(channel.abs() > _TURBO_FFN_GUARD_MAX_ABS):
        return hidden
    guarded = hidden.clone()
    guarded[..., _TURBO_FFN_GUARD_CHANNEL] = channel.clamp(-_TURBO_FFN_GUARD_MAX_ABS, _TURBO_FFN_GUARD_MAX_ABS)
    return guarded


def _guard_down(prefix: str, overrides: set[str], down_is_quantized: bool) -> bool:
    """Wheel skips the guard for the whole model when any fp_overrides entry is present."""
    return down_is_quantized and not overrides and prefix == "layers.0"


class _QuantSwiGLU(nn.Module):
    """The checkpoint fuses w3 and w1 into feed_forward.net.0.proj as [up | gate]."""

    def __init__(self, proj: nn.Module, down: nn.Module, guard_down: bool = False):
        super().__init__()
        self.proj = proj
        self.down = down
        self.guard_down = guard_down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The turbo guard clamps a post-GLU channel, so this layer cannot stay inside fused_swiglu.
        if self.guard_down:
            up, gate = self.proj(x).chunk(2, dim=-1)
            hidden = up * torch.nn.functional.silu(gate)
            return self.down(clamp_turbo_ffn_activation(hidden))
        if getattr(self.proj, "qweight", None) is not None and getattr(self.down, "qweight", None) is not None:
            return fused_swiglu(x, self.proj, self.down)
        up, gate = self.proj(x).chunk(2, dim=-1)
        return self.down(up * torch.nn.functional.silu(gate))


def _group_size(raw: dict) -> int:
    weight = raw.get("weight") or {}
    top = raw.get("group_size")
    nested = weight.get("group_size")
    if top is not None and nested is not None and int(top) != int(nested):
        raise ValueError(f"quantization_config.group_size is {top}, but weight.group_size is {nested}.")
    if top is None and nested is None:
        return 16
    return int(top if top is not None else nested)


def _patch_token(state_dict: dict) -> str:
    found = set()
    for key in state_dict:
        for prefix in ("all_final_layer.", "all_x_embedder."):
            if key.startswith(prefix):
                found.add(key[len(prefix):].split(".", 1)[0])
    if len(found) != 1:
        raise KeyError(f"Z-Image patch size token is ambiguous: {sorted(found) or 'missing'}.")
    return found.pop()


def _quantize(linear: nn.Module, qcfg: QuantConfig, dtype: torch.dtype) -> nn.Module:
    return make_w4a4(
        linear.in_features,
        linear.out_features,
        qcfg.rank,
        qcfg.precision,
        dtype,
        device=torch.device("meta"),
        bias=linear.bias is not None,
    )


class ZImageSpec:
    model_class = "NunchakuZImageTransformer2DModel"
    supported_precisions = ("nvfp4", "int4")
    min_nunchaku = "1.4.1"

    def validate(self, diffusers_config: dict, qcfg: QuantConfig) -> None:
        if diffusers_config.get("_class_name") not in (None, "ZImageTransformer2DModel"):
            raise ValueError(f"Not a Z-Image DiT: {diffusers_config.get('_class_name')!r}.")
        if qcfg.raw.get("skip_refiners"):
            raise ValueError("This checkpoint leaves the refiners in floating point. This loader only accepts quantized refiners.")
        overrides = qcfg.raw.get("fp_overrides") or []
        if not isinstance(overrides, list) or any(not _FP_OVERRIDE.fullmatch(str(item)) for item in overrides):
            raise ValueError(f"Unrecognized fp_overrides: {overrides!r}.")
        expected_group = 16 if qcfg.precision == "nvfp4" else 64
        group = _group_size(qcfg.raw)
        if group != expected_group:
            raise ValueError(f"Z-Image {qcfg.precision} group_size must be {expected_group}, got {group}.")

        patches = diffusers_config.get("all_patch_size") or [2]
        f_patches = diffusers_config.get("all_f_patch_size") or [1]
        if len(patches) != 1 or len(f_patches) != 1:
            raise ValueError(f"Only a single patch size is supported, got {patches!r} / {f_patches!r}.")
        dim = int(diffusers_config["dim"])
        n_heads = int(diffusers_config["n_heads"])
        n_kv_heads = int(diffusers_config.get("n_kv_heads", n_heads))
        axes = [int(item) for item in diffusers_config["axes_dims"]]
        if n_heads != n_kv_heads:
            raise ValueError(f"Fused QKV requires n_heads == n_kv_heads, got {n_heads} / {n_kv_heads}.")
        if dim // n_heads != 128 or sum(axes) != 128:
            raise ValueError(f"Head dimension must be 128, got dim={dim} heads={n_heads} axes={axes}.")
        if not diffusers_config.get("qk_norm", True):
            raise ValueError("Z-Image requires qk_norm.")
        if int(diffusers_config.get("in_channels", 16)) != 16:
            raise ValueError(f"in_channels must be 16, got {diffusers_config.get('in_channels')!r}.")

    def comfy_model_config(self, diffusers_config: dict):
        import comfy.supported_models

        config = diffusers_config
        unet_config = {
            "image_model": "lumina2",
            "patch_size": int(config["all_patch_size"][0]),
            "in_channels": int(config["in_channels"]),
            "dim": int(config["dim"]),
            "cap_feat_dim": int(config["cap_feat_dim"]),
            "n_layers": int(config["n_layers"]),
            "n_refiner_layers": int(config["n_refiner_layers"]),
            "n_heads": int(config["n_heads"]),
            "n_kv_heads": int(config["n_kv_heads"]),
            "axes_dims": [int(item) for item in config["axes_dims"]],
            "axes_lens": [int(item) for item in config["axes_lens"]],
            "rope_theta": float(config["rope_theta"]),
            "qk_norm": bool(config.get("qk_norm", True)),
            "ffn_dim_multiplier": float(config.get("ffn_dim_multiplier", 8.0 / 3.0)),
            "z_image_modulation": True,
            "time_scale": float(config.get("t_scale", 1000.0)),
            "pad_tokens_multiple": 32,
            "norm_eps": float(config.get("norm_eps", 1e-5)),
        }
        return comfy.supported_models.ZImage(unet_config)

    def patch_model(self, diffusion_model: nn.Module, qcfg: QuantConfig, dtype: torch.dtype) -> None:
        overrides = set(qcfg.raw.get("fp_overrides") or [])
        groups = (
            ("layers", diffusion_model.layers),
            ("noise_refiner", diffusion_model.noise_refiner),
            ("context_refiner", diffusion_model.context_refiner),
        )
        for name, blocks in groups:
            for index, block in enumerate(blocks):
                self._patch_block(block, f"{name}.{index}", qcfg, dtype, overrides)

    def _patch_block(self, block: nn.Module, prefix: str, qcfg: QuantConfig, dtype: torch.dtype, overrides: set[str]) -> None:
        attention = block.attention
        attention.qkv = _quantize(attention.qkv, qcfg, dtype)
        attention.out = _quantize(attention.out, qcfg, dtype)

        feed_forward = block.feed_forward
        hidden = feed_forward.w1.out_features
        width = feed_forward.w1.in_features
        if f"{prefix}.feed_forward.net.0.proj" in overrides:
            proj = feed_forward.w1.__class__(
                width,
                hidden * 2,
                bias=False,
                device=torch.device("meta"),
                dtype=dtype,
            )
        else:
            proj = make_w4a4(
                width,
                hidden * 2,
                qcfg.rank,
                qcfg.precision,
                dtype,
                device=torch.device("meta"),
                bias=False,
            )
        if f"{prefix}.feed_forward.net.2" in overrides:
            down = feed_forward.w2
            quantized_down = False
        else:
            down = make_w4a4(
                hidden,
                width,
                qcfg.rank,
                qcfg.precision,
                dtype,
                device=torch.device("meta"),
                bias=False,
            )
            quantized_down = True
        block.feed_forward = _QuantSwiGLU(proj, down, guard_down=_guard_down(prefix, overrides, quantized_down))

    def convert_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        token = _patch_token(state_dict)
        renamed = {
            f"all_final_layer.{token}.": "final_layer.",
            f"all_x_embedder.{token}.": "x_embedder.",
        }
        converted = {}
        for key, tensor in state_dict.items():
            new_key = key
            for source, target in _KEY_REPLACEMENTS:
                if source in new_key:
                    new_key = new_key.replace(source, target, 1)
            for source, target in renamed.items():
                if source in new_key:
                    new_key = new_key.replace(source, target, 1)
            converted[new_key] = tensor
        return converted


register_spec(ZImageSpec())
