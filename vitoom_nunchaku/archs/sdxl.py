"""SDXL UNet: replace quantized linear layers inside the native ComfyUI network.

Checkpoint keys use diffusers names. Self-attention Q/K/V is fused into ``attn1.to_qkv``.
Cross-attention quantizes only ``to_q`` and the output projection; K/V stay BF16.
GEGLU matches ComfyUI as ``[hidden | gate]``, and the quantized layers drop into the original ``ff.net``.
"""

from __future__ import annotations

import json
import os

import torch
from torch import nn

from vitoom_nunchaku.archs.base import QuantConfig, register_spec
from vitoom_nunchaku.runtime.layers import fused_geglu, make_w4a4

_DOWN_TYPES = {"DownBlock2D", "CrossAttnDownBlock2D"}
_UP_TYPES = {"UpBlock2D", "CrossAttnUpBlock2D"}
_QUANT_SUFFIXES = (
    (".lora_down", ".proj_down"),
    (".lora_up", ".proj_up"),
    (".smooth_orig", ".smooth_factor_orig"),
    (".smooth", ".smooth_factor"),
)


def _as_list(value, count: int, name: str) -> list[int]:
    if isinstance(value, int):
        return [int(value)] * count
    items = [int(item) for item in value]
    if len(items) != count:
        raise ValueError(f"{name} length must be {count}, got {items!r}.")
    return items


def _unet_config(diffusers_config: dict) -> dict:
    channels = [int(size) for size in diffusers_config["block_out_channels"]]
    down_types = [str(kind) for kind in diffusers_config["down_block_types"]]
    up_types = [str(kind) for kind in diffusers_config["up_block_types"]]
    if len(channels) != len(down_types) or len(up_types) != len(down_types):
        raise ValueError("block_out_channels does not match the number of down/up blocks.")
    levels = len(channels)
    layers = _as_list(diffusers_config.get("layers_per_block", 2), levels, "layers_per_block")
    transformer_layers = _as_list(
        diffusers_config.get("transformer_layers_per_block", 1),
        levels,
        "transformer_layers_per_block",
    )
    reverse = diffusers_config.get("reverse_transformer_layers_per_block")
    up_layers = list(reversed(transformer_layers)) if reverse is None else _as_list(
        reverse, levels, "reverse_transformer_layers_per_block"
    )

    depth_in = []
    for index, kind in enumerate(down_types):
        if kind not in _DOWN_TYPES:
            raise ValueError(f"Unsupported down block: {kind}")
        depth = transformer_layers[index] if kind == "CrossAttnDownBlock2D" else 0
        depth_in.extend([depth] * layers[index])

    depth_out_deep = []
    for index, kind in enumerate(up_types):
        if kind not in _UP_TYPES:
            raise ValueError(f"Unsupported up block: {kind}")
        level = levels - 1 - index
        depth = up_layers[index] if kind == "CrossAttnUpBlock2D" else 0
        depth_out_deep.extend([depth] * (layers[level] + 1))

    model_channels = channels[0]
    return {
        "image_size": 32,
        "in_channels": int(diffusers_config["in_channels"]),
        "out_channels": int(diffusers_config["out_channels"]),
        "model_channels": model_channels,
        "num_res_blocks": layers,
        "channel_mult": [size // model_channels for size in channels],
        "transformer_depth": depth_in,
        "transformer_depth_output": list(reversed(depth_out_deep)),
        "transformer_depth_middle": transformer_layers[-1],
        "context_dim": int(diffusers_config["cross_attention_dim"]),
        "adm_in_channels": int(diffusers_config["projection_class_embeddings_input_dim"]),
        "use_linear_in_transformer": True,
        "use_spatial_transformer": True,
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
        "use_checkpoint": False,
        "legacy": False,
        "num_classes": "sequential",
        "num_head_channels": 64,
    }


def _prefix_pairs(unet_config: dict) -> list[tuple[str, str]]:
    import comfy.utils

    mapping = comfy.utils.unet_to_diffusers(unet_config)
    pairs: dict[str, str] = {}
    for diffusers_key, comfy_key in mapping.items():
        if diffusers_key.rsplit(".", 1)[-1] not in ("weight", "bias"):
            raise RuntimeError(f"Cannot take a module prefix from key: {diffusers_key}")
        source = diffusers_key.rsplit(".", 1)[0]
        target = comfy_key.rsplit(".", 1)[0]
        pairs[source] = target
        marker = ".transformer_blocks."
        if marker in diffusers_key:
            head, _, tail = diffusers_key.partition(marker)
            index = tail.split(".", 1)[0]
            comfy_head, _, comfy_tail = comfy_key.partition(marker)
            pairs[f"{head}{marker}{index}"] = f"{comfy_head}{marker}{comfy_tail.split('.', 1)[0]}"
    return sorted(pairs.items(), key=lambda item: len(item[0]), reverse=True)


def _rename_quant_key(key: str) -> str:
    if ".transformer_blocks." not in key:
        return key
    for source, target in _QUANT_SUFFIXES:
        if source in key:
            return key.replace(source, target)
    return key


def _linear_io(linear: nn.Module) -> tuple[int, int]:
    in_features = getattr(linear, "in_features", None)
    out_features = getattr(linear, "out_features", None)
    if in_features is None or out_features is None:
        raise RuntimeError("Cannot read input and output sizes from the linear layer.")
    return int(in_features), int(out_features)


class QuantSelfAttention(nn.Module):
    """Self-attention. Q/K/V is one fused ``to_qkv`` in the checkpoint."""

    def __init__(self, source: nn.Module, qkv: nn.Module, out_proj: nn.Module):
        super().__init__()
        self.heads = source.heads
        self.attn_precision = source.attn_precision
        self.comfy_attention = source.comfy_attention
        self.to_qkv = qkv
        self.to_out = nn.Sequential(out_proj, source.to_out[1])

    def forward(self, x, context=None, value=None, mask=None, transformer_options={}):
        if mask is not None:
            raise RuntimeError("Quantized SDXL self-attention does not support an attention mask.")
        import comfy.ldm.modules.attention as attention

        query, key, value = self.to_qkv(x).chunk(3, dim=-1)
        query = attention.AttentionTensorContainer(query)
        key = attention.AttentionTensorContainer(key)
        value = attention.AttentionTensorContainer(value)
        hidden = attention.optimized_attention(
            query,
            key,
            value,
            self.heads,
            attn_precision=self.attn_precision,
            preferred_attention=self.comfy_attention,
            transformer_options=transformer_options,
        )
        return self.to_out(hidden)


def _replace_attention(block: nn.Module, qcfg: QuantConfig, dtype: torch.dtype, device: torch.device) -> None:
    attn1 = block.attn1
    in_features, out_features = _linear_io(attn1.to_q)
    if in_features != out_features:
        raise RuntimeError(f"Self-attention Q input and output sizes differ: {in_features} -> {out_features}.")
    qkv = make_w4a4(in_features, out_features * 3, qcfg.rank, qcfg.precision, dtype, device, bias=False)
    out_in, out_out = _linear_io(attn1.to_out[0])
    out_proj = make_w4a4(out_in, out_out, qcfg.rank, qcfg.precision, dtype, device, bias=True)
    block.attn1 = QuantSelfAttention(attn1, qkv, out_proj)

    attn2 = block.attn2
    query_in, query_out = _linear_io(attn2.to_q)
    attn2.to_q = make_w4a4(query_in, query_out, qcfg.rank, qcfg.precision, dtype, device, bias=False)
    cross_in, cross_out = _linear_io(attn2.to_out[0])
    attn2.to_out[0] = make_w4a4(cross_in, cross_out, qcfg.rank, qcfg.precision, dtype, device, bias=True)


def _replace_feed_forward(block: nn.Module, qcfg: QuantConfig, dtype: torch.dtype, device: torch.device) -> None:
    proj = block.ff.net[0].proj
    second = block.ff.net[2]
    proj_in, proj_out = _linear_io(proj)
    second_in, second_out = _linear_io(second)
    block.ff.net[0].proj = make_w4a4(proj_in, proj_out, qcfg.rank, qcfg.precision, dtype, device, bias=True)
    block.ff.net[2] = make_w4a4(second_in, second_out, qcfg.rank, qcfg.precision, dtype, device, bias=True)
    _install_fused_geglu(block.ff)


def _install_fused_geglu(feed_forward: nn.Module) -> None:
    """Keep ff.net.0.proj / ff.net.2 key names and run the wheel's fused GEGLU."""
    if getattr(feed_forward, "_vitoom_fused_geglu", False):
        return
    original = feed_forward.forward

    def forward(hidden_states, *args, **kwargs):
        proj = feed_forward.net[0].proj
        down = feed_forward.net[2]
        dropout = feed_forward.net[1]
        quantized = getattr(proj, "qweight", None) is not None and getattr(down, "qweight", None) is not None
        if quantized and (not isinstance(dropout, nn.Dropout) or dropout.p == 0):
            return fused_geglu(hidden_states, proj, down)
        return original(hidden_states, *args, **kwargs)

    feed_forward.forward = forward
    feed_forward._vitoom_fused_geglu = True


def _read_scheduler(path: str) -> tuple[str, bool]:
    scheduler = os.path.join(os.path.dirname(os.path.realpath(path)), "scheduler", "scheduler_config.json")
    if not os.path.isfile(scheduler):
        return "epsilon", False
    with open(scheduler, encoding="utf-8") as handle:
        payload = json.load(handle)
    prediction = str(payload.get("prediction_type") or "epsilon")
    if prediction not in ("epsilon", "v_prediction"):
        raise ValueError(f"Unsupported scheduler prediction_type: {prediction!r}.")
    return prediction, bool(payload.get("rescale_betas_zero_snr"))


class _ComfySDXL:
    """Add v-prediction on top of ComfyUI's SDXL config. The quantized file has no ``v_pred`` key."""

    def __new__(cls, unet_config: dict, prediction: str, zero_snr: bool):
        import comfy.supported_models as supported_models

        class Configured(supported_models.SDXL):
            def model_type(self, state_dict, prefix=""):
                if prediction != "v_prediction":
                    return super().model_type(state_dict, prefix)
                marked = dict(state_dict)
                marked["v_pred"] = torch.empty(0)
                if zero_snr:
                    marked["ztsnr"] = torch.empty(0)
                return super().model_type(marked, prefix)

        return Configured(unet_config)


class SDXLSpec:
    model_class = "NunchakuSDXLUNet2DConditionModel"
    supported_precisions = ("nvfp4", "int4")
    min_nunchaku = "1.4.1"

    def __init__(self):
        self._prediction = "epsilon"
        self._zero_snr = False

    def bind_checkpoint(self, path: str) -> None:
        self._prediction, self._zero_snr = _read_scheduler(path)

    def validate(self, diffusers_config: dict, qcfg: QuantConfig) -> None:
        self._diffusers_config = diffusers_config
        class_name = str(diffusers_config.get("_class_name", ""))
        if "UNet2DCondition" not in class_name:
            raise ValueError(f"config._class_name is not UNet2DCondition: {class_name!r}")
        if diffusers_config.get("addition_embed_type") != "text_time":
            raise ValueError("SDXL requires addition_embed_type=text_time.")
        if diffusers_config.get("use_linear_projection") is not True:
            raise ValueError("SDXL attention input projections must be Linear.")
        if diffusers_config.get("dual_cross_attention") or diffusers_config.get("only_cross_attention"):
            raise ValueError("dual_cross_attention and only_cross_attention are not supported.")
        if int(diffusers_config.get("in_channels", 0)) != 4 or int(diffusers_config.get("out_channels", 0)) != 4:
            raise ValueError("This loader only accepts a 4-channel SDXL UNet. The inpaint variant is not wired up.")
        if str(diffusers_config.get("mid_block_type") or "UNetMidBlock2DCrossAttn") != "UNetMidBlock2DCrossAttn":
            raise ValueError(f"Unsupported mid block: {diffusers_config.get('mid_block_type')!r}.")
        if qcfg.raw.get("fp_overrides"):
            raise ValueError("This checkpoint has fp_overrides, so some blocks stay in floating point. This loader has no branch for that.")
        channels = [int(size) for size in diffusers_config.get("block_out_channels", ())]
        heads = diffusers_config.get("num_attention_heads") or diffusers_config.get("attention_head_dim")
        heads = _as_list(heads, len(channels), "attention_head_dim")
        for width, count in zip(channels, heads):
            if count < 1 or width % count != 0 or width // count != 64:
                raise ValueError(f"SDXL uses 64 dimensions per head; width {width} is not valid for {count} heads.")
        _unet_config(diffusers_config)

    def comfy_model_config(self, diffusers_config: dict):
        prediction, zero_snr = self._scheduler()
        return _ComfySDXL(_unet_config(diffusers_config), prediction, zero_snr)

    def _scheduler(self) -> tuple[str, bool]:
        from vitoom_nunchaku.runtime.load import current_load_state

        state = current_load_state()
        if state is not None:
            return _read_scheduler(state.path)
        return self._prediction, self._zero_snr

    def _active_config(self) -> dict:
        from vitoom_nunchaku.runtime.load import current_load_state

        state = current_load_state()
        if state is not None and state.config is not None:
            return state.config
        return self._diffusers_config

    def patch_model(self, diffusion_model: nn.Module, qcfg: QuantConfig, dtype: torch.dtype) -> None:
        import comfy.ldm.modules.attention as attention

        blocks = [module for module in diffusion_model.modules() if isinstance(module, attention.BasicTransformerBlock)]
        if not blocks:
            raise RuntimeError("The ComfyUI SDXL UNet has no transformer blocks.")
        for module in blocks:
            if not isinstance(module.ff.net[0], attention.GEGLU):
                raise RuntimeError("SDXL transformer ff.net.0 should be GEGLU.")
            device = next(module.parameters()).device
            _replace_attention(module, qcfg, dtype, device)
            _replace_feed_forward(module, qcfg, dtype, device)

    def convert_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pairs = _prefix_pairs(_unet_config(self._active_config()))
        converted = {}
        for key, value in state_dict.items():
            renamed = _rename_quant_key(key)
            matched = None
            for source, target in pairs:
                if renamed == source or renamed.startswith(source + "."):
                    matched = target + renamed[len(source):]
                    break
            if matched is None:
                raise KeyError(f"No SDXL key mapping for {key}")
            converted[matched] = value
        return converted


register_spec(SDXLSpec())
