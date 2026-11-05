"""Hardware compatibility hints and a GPU-resident ModelPatcher."""

from __future__ import annotations

import torch

_BLACKWELL = {"120", "121"}
_INT4_ARCH = {"75", "80", "86", "89"}
_W8A8_ARCH = {"80", "86", "89", "90", "100", "120", "121"}


def _sm(device: str | torch.device) -> str:
    parsed = torch.device(device)
    index = 0 if parsed.index is None else parsed.index
    major, minor = torch.cuda.get_device_capability(index)
    return f"{major}{minor}"


def hardware_hint(quantization_config: dict, device: str | torch.device = "cuda") -> str | None:
    """Return None when compatible, otherwise a note on which file to use instead."""
    weight_dtype = (quantization_config.get("weight") or {}).get("dtype")
    try:
        sm = _sm(device)
    except Exception as exc:
        return f"Cannot read GPU compute capability ({exc}). This plugin only supports NVIDIA CUDA."

    if weight_dtype == "int8":
        if sm not in _W8A8_ARCH:
            return f"The current GPU (sm_{sm}) cannot run W8A8. W8A8 needs sm_80 or newer (Turing sm_75 is excluded)."
        return None
    if sm in _BLACKWELL and weight_dtype != "fp4_e2m1_all":
        return (
            f"RTX 50 series (sm_{sm}) can only use nvfp4. "
            "Use a checkpoint whose filename contains svdq-fp4, and do not use svdq-int4."
        )
    if sm in _INT4_ARCH and weight_dtype != "int4":
        return (
            f"The current GPU (sm_{sm}) should use int4. "
            "Use a checkpoint whose filename contains svdq-int4, and do not use svdq-fp4."
        )
    if weight_dtype in ("int4", "fp4_e2m1_all") and sm not in _BLACKWELL and sm not in _INT4_ARCH:
        return f"The current GPU (sm_{sm}) has no usable W4A4 kernel. Use a Turing, Ampere, Ada, or Blackwell GPU."
    return None


def assert_hardware(quantization_config: dict, device: str | torch.device = "cuda") -> None:
    from nunchaku.utils import check_hardware_compatibility

    hint = hardware_hint(quantization_config, device)
    try:
        check_hardware_compatibility(quantization_config, device)
    except ValueError as exc:
        message = hint or str(exc)
        if hint and str(exc) not in hint:
            message = f"{hint}\n{exc}"
        raise ValueError(message) from exc
    if hint:
        raise ValueError(hint)


_PATCHERS: dict[type, type] = {}


def patcher_class(disable_dynamic: bool):
    """Choose the base class at load time. CoreModelPatcher points at the dynamic-VRAM implementation only after ComfyUI starts."""
    import comfy.model_management
    import comfy.model_patcher as model_patcher

    base = model_patcher.ModelPatcher if disable_dynamic else model_patcher.CoreModelPatcher
    cached = _PATCHERS.get(base)
    if cached is not None:
        return cached

    class VitoomNunchakuModelPatcher(base):
        def model_size(self):
            # qweight is int8 and nvfp4 wscales are fp8, so size is the actual byte count.
            if self.size > 0:
                return self.size
            self.size = comfy.model_management.module_size(self.model)
            return self.size

    VitoomNunchakuModelPatcher.__name__ = "VitoomNunchakuModelPatcher"
    _PATCHERS[base] = VitoomNunchakuModelPatcher
    return VitoomNunchakuModelPatcher
