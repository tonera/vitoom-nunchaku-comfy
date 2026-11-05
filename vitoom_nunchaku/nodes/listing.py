"""Shared file lists for nodes. This module does not import nunchaku."""

from __future__ import annotations

_EMPTY_MODEL = "(no nunchaku checkpoint found)"
_EMPTY_LORA = "(loras directory is empty)"
_EMPTY_TEXT = "(no nunchaku text encoder found)"


def model_choices() -> list[str]:
    try:
        from vitoom_nunchaku.runtime.checkpoint import choice_label, iter_diffusion_checkpoints

        found = iter_diffusion_checkpoints()
    except Exception:
        return [_EMPTY_MODEL]
    if not found:
        return [_EMPTY_MODEL]
    return [choice_label(name, info) for name, info in found]


def lora_choices() -> list[str]:
    try:
        import folder_paths
    except Exception:
        return [_EMPTY_LORA]
    names = list(folder_paths.get_filename_list("loras"))
    return names or [_EMPTY_LORA]


def text_encoder_choices() -> list[str]:
    try:
        from vitoom_nunchaku.runtime.text_encoder import iter_text_encoder_files

        found = iter_text_encoder_files()
    except Exception:
        return [_EMPTY_TEXT]
    if not found:
        return [_EMPTY_TEXT]
    return [label for label, _path in found]


def is_placeholder(name: str) -> bool:
    return name in (_EMPTY_MODEL, _EMPTY_LORA, _EMPTY_TEXT) or str(name).startswith("(")
