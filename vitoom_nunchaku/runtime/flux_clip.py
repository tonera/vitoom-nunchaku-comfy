"""FLUX.1 CLIP: AWQ T5-XXL for the token sequence, plus a standard CLIP-L for the pooled vector."""

from __future__ import annotations

import os

import torch
from torch import nn

_T5_CLASS = "NunchakuT5EncoderModel"
_WEIGHTS_ROOT = os.path.expanduser("~/weights")
_T5_PAD = 0


def iter_t5_files() -> list[tuple[str, str]]:
    """Return (dropdown label, absolute path)."""
    found: list[str] = []
    seen: set[str] = set()
    for root in _search_roots():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if not name.startswith(".")]
            for filename in filenames:
                if not filename.endswith(".safetensors"):
                    continue
                path = os.path.realpath(os.path.join(dirpath, filename))
                if path in seen:
                    continue
                seen.add(path)
                metadata = _metadata(path)
                if metadata is None or metadata.get("model_class") != _T5_CLASS:
                    continue
                found.append(path)
    labeled = [(f"{os.path.basename(path)} [awq t5xxl]", path) for path in found]
    labeled.sort(key=lambda item: item[0])
    return labeled


def resolve_t5(choice: str) -> str:
    wanted = choice.split(" [", 1)[0]
    filename = os.path.basename(wanted)
    matches = []
    for label, path in iter_t5_files():
        if choice == label or wanted in (os.path.basename(path), path) or filename == os.path.basename(path):
            matches.append(path)
    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise RuntimeError(f"T5 encoder {choice!r} was not found. Place the Nunchaku T5 safetensors in ~/weights or models/text_encoders.")
    raise RuntimeError(f"T5 encoder {choice!r} matches more than one file: {unique}")


class VitoomFlux1Cond(nn.Module):
    def __init__(self, clip_l: nn.Module, t5: nn.Module):
        super().__init__()
        self.clip_l = clip_l
        self.t5 = t5

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_l.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        _hidden, pooled = self.clip_l.encode_token_weights(token_weight_pairs["l"])
        ids = _token_ids(token_weight_pairs["t5xxl"])
        device = next(self.t5.parameters()).device
        hidden = self.t5(input_ids=ids.to(device)).last_hidden_state
        return hidden.float(), pooled

    def load_sd(self, state_dict):
        return None


def load_flux_clip(t5_path: str, clip_l_name: str, options: dict | None = None):
    """Return a ComfyUI CLIP whose tokenizer is FluxTokenizer."""
    import comfy.hooks
    import comfy.model_management as model_management
    import comfy.model_patcher
    import comfy.sd
    import comfy.utils
    import folder_paths
    from comfy import sd1_clip
    from comfy.text_encoders.flux import FluxTokenizer

    from vitoom_nunchaku.runtime.env import ensure_nunchaku
    from vitoom_nunchaku.runtime.load import resolve_device

    options = dict(options or {})
    ensure_nunchaku()
    from nunchaku import NunchakuT5EncoderModel

    load_device = resolve_device(options.get("device") or "default")
    offload_device = model_management.text_encoder_offload_device()
    clip_l_path = folder_paths.get_full_path_or_raise("text_encoders", clip_l_name)
    clip_l = sd1_clip.SDClipModel(
        device=offload_device,
        dtype=torch.float16,
        return_projected_pooled=False,
        model_options={},
    )
    clip_l.load_sd(comfy.utils.load_torch_file(clip_l_path, safe_load=True))
    t5 = NunchakuT5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.bfloat16, device=offload_device)
    t5.eval()
    for parameter in t5.parameters():
        parameter.requires_grad_(False)

    cond = VitoomFlux1Cond(clip_l, t5)
    clip = comfy.sd.CLIP(no_init=True)
    clip.cond_stage_model = cond
    clip.tokenizer = FluxTokenizer()
    clip.patcher = comfy.model_patcher.ModelPatcher(cond, load_device=load_device, offload_device=offload_device)
    clip.patcher.is_clip = True
    clip.patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
    clip.layer_idx = None
    clip.use_clip_schedule = False
    clip.apply_hooks_to_conds = None
    clip.tokenizer_options = {}
    clip.patcher.cached_patcher_init = (load_flux_clip_patcher, (t5_path, clip_l_name, options))
    return clip


def load_flux_clip_patcher(t5_path: str, clip_l_name: str, options: dict | None = None, disable_dynamic: bool = False):
    del disable_dynamic
    return load_flux_clip(t5_path, clip_l_name, options).patcher


def _token_ids(sequences: list) -> torch.Tensor:
    rows = []
    width = 0
    for sequence in sequences:
        row = []
        for item in sequence:
            token = item[0] if isinstance(item, (tuple, list)) else item
            if isinstance(token, dict):
                raise RuntimeError("FLUX.1 CLIP only encodes text. Send reference images through ReferenceLatent or InstructPixToPixConditioning.")
            row.append(int(token))
        width = max(width, len(row))
        rows.append(row)
    if not rows:
        raise RuntimeError("FLUX.1 CLIP received empty tokens.")
    ids = torch.full((len(rows), width), _T5_PAD, dtype=torch.long)
    for index, row in enumerate(rows):
        if row:
            ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return ids


def _search_roots() -> list[str]:
    roots = []
    if os.path.isdir(_WEIGHTS_ROOT):
        roots.append(_WEIGHTS_ROOT)
    try:
        import folder_paths

        roots.extend(folder_paths.get_folder_paths("text_encoders"))
    except Exception:
        pass
    unique = []
    seen = set()
    for root in roots:
        key = os.path.abspath(root)
        if key in seen or not os.path.isdir(key):
            continue
        seen.add(key)
        unique.append(key)
    return unique


def _metadata(path: str) -> dict | None:
    try:
        from vitoom_nunchaku.runtime.checkpoint import read_header

        metadata, _tensors = read_header(path)
    except Exception:
        return None
    return metadata
