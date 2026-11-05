"""Build SVDQ linear layers and the fused SwiGLU used by Flux3."""

from __future__ import annotations

import torch
from torch import nn


_W8A8_LINEAR = None


def _w8a8_linear_cls():
    """W8A8 on 5-D feature maps. ComfyUI Cosmos attention feeds B,T,H,W,C straight into the linear layer."""
    global _W8A8_LINEAR
    if _W8A8_LINEAR is not None:
        return _W8A8_LINEAR
    from nunchaku.models.linear_w8a8 import SVDQW8A8Linear

    class W8A8Linear(SVDQW8A8Linear):
        def forward(self, x, output=None):
            if x.ndim <= 3:
                return super().forward(x, output=output)
            if output is not None:
                raise ValueError("W8A8 preallocated output only supports 2-D or 3-D input.")
            lead = x.shape[:-1]
            flat = super().forward(x.reshape(-1, self.in_features))
            return flat.reshape(*lead, self.out_features)

    _W8A8_LINEAR = W8A8Linear
    return _W8A8_LINEAR


def make_w8a8(
    in_features: int,
    out_features: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device | str = "meta",
) -> nn.Module:
    if dtype != torch.bfloat16:
        raise ValueError(f"Anima W8A8 quantized layers only accept bfloat16, got {dtype}.")
    return _w8a8_linear_cls()(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        bias=False,
        torch_dtype=dtype,
        device=device,
    )


def make_w4a4(
    in_features: int,
    out_features: int,
    rank: int,
    precision: str,
    dtype: torch.dtype,
    device: torch.device | str = "meta",
    bias: bool = False,
) -> nn.Module:
    from nunchaku.models.linear import SVDQW4A4Linear

    return SVDQW4A4Linear(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        bias=bias,
        precision=precision,
        torch_dtype=dtype,
        device=device,
    )


def make_awq(
    in_features: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device | str = "meta",
    bias: bool = True,
    group_size: int = 64,
) -> nn.Module:
    """W4A16 AdaLN projection used by FLUX.1. It is not the W4A4 DiT GEMM."""
    from nunchaku.models.linear import AWQW4A16Linear

    return AWQW4A16Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        group_size=group_size,
        torch_dtype=dtype,
        device=device,
    )


def fused_gelu(x: torch.Tensor, fc1: nn.Module, fc2: nn.Module) -> torch.Tensor:
    """GELU MLP. INT4 shifts the activation so the down projection can use unsigned quantization."""
    from nunchaku.ops.fused import fused_gelu_mlp

    return fused_gelu_mlp(x, fc1, fc2)


def fused_swiglu(x: torch.Tensor, mlp_in: nn.Module, mlp_out: nn.Module) -> torch.Tensor:
    """Checkpoint linear_in is [up | gate]. The fused kernel computes up * silu(gate)."""
    from nunchaku.ops.fused import fused_swiglu_mlp

    return fused_swiglu_mlp(x, mlp_in, mlp_out)


def fused_geglu(x: torch.Tensor, proj: nn.Module, down: nn.Module) -> torch.Tensor:
    """SDXL GEGLU is [up | gate]. The fused kernel applies up * gelu(gate) and quantizes that activation."""
    from nunchaku.ops.fused import fused_geglu_mlp

    return fused_geglu_mlp(x, proj, down)


def restore_fp32_qk_norms(module: nn.Module) -> None:
    """Q/K RMSNorm weights are fp32 in the checkpoint, and RMSNorm multiplies by them."""
    for name, child in module.named_modules():
        if not (name.endswith("query_norm") or name.endswith("key_norm")):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dtype == torch.float32:
            continue
        child.weight = nn.Parameter(weight.detach().to(dtype=torch.float32), requires_grad=False)
