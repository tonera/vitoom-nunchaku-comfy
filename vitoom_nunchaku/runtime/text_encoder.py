"""Nunchaku Qwen text encoders. Quantized layers use W4A16 and are not subject to W4A4 hardware limits.

- Qwen3-VL, hidden size 2560: FLUX.3, layers 4 through 32, concatenated to 20480.
- Qwen3-VL, hidden size 4096: Qwen-Image 2.1, the last layer before the final RMSNorm.
- Text-only Qwen3 (``NunchakuQwenEncoderModel``): FLUX.2 Klein, layers 9, 18, and 27.
  4B concatenates to 7680 and 8B concatenates to 12288.
  The same 4B file can feed Z-Image-Turbo when the loader target is Z-Image: the second-to-last layer, width 2560.
"""

from __future__ import annotations

import json
import math
import os

import torch

FLUX3_LAYER_INDEXES = (4, 8, 12, 16, 20, 24, 28, 32)
FLUX2_LAYER_INDEXES = (9, 18, 27)
FLUX3_PAD_TOKEN = 151643
FLUX3_PAD_MULTIPLE = 80
FLUX3_MAX_SEQUENCE = 8192
QWEN_IMAGE21_IM_START = 151644
QWEN_IMAGE21_IMAGE_PAD = 151655
_VL_MODEL_CLASS = "NunchakuQwen3VLEncoderModel"
_TEXT_MODEL_CLASS = "NunchakuQwenEncoderModel"
_TEXT_MODEL_CLASSES = {_VL_MODEL_CLASS, _TEXT_MODEL_CLASS}
_WEIGHTS_ROOT = os.path.expanduser("~/weights")


def text_encoder_kind(metadata: dict) -> str:
    quant_raw = metadata.get("quantization_config") or "{}"
    quant = json.loads(quant_raw) if isinstance(quant_raw, str) else dict(quant_raw)
    text = quant.get("text") if isinstance(quant.get("text"), dict) else quant
    fmt = str(text.get("format") or "")
    if "w4x16" in fmt or "w4a16" in fmt:
        return "w4a16"
    weight = text.get("weight") or {}
    dtype = weight.get("dtype")
    if dtype == "int4":
        return "int4"
    if dtype == "fp4_e2m1_all":
        return "nvfp4"
    return "unknown"


def text_encoder_hidden_size(metadata: dict) -> int:
    config_raw = metadata.get("config") or "{}"
    config = json.loads(config_raw) if isinstance(config_raw, str) else dict(config_raw)
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    return int(text.get("hidden_size") or 0)


def text_encoder_recipe(metadata: dict) -> str:
    """Without model_class, treat the file as Qwen3-VL: 2560 maps to FLUX.3 and 4096 maps to Qwen-Image 2.1."""
    hidden = text_encoder_hidden_size(metadata)
    model_class = str(metadata.get("model_class") or _VL_MODEL_CLASS)
    if model_class == _TEXT_MODEL_CLASS:
        if hidden in (2560, 4096):
            return "flux2"
        raise ValueError(f"The FLUX.2 text encoder only supports hidden size 2560 or 4096, got {hidden}.")
    if hidden == 2560:
        return "flux3"
    if hidden == 4096:
        return "qwen_image21"
    raise ValueError(f"Cannot choose a text-encoder recipe from hidden_size={hidden}. Only 2560 (FLUX.3) and 4096 (Qwen-Image 2.1) are supported.")


def text_encoder_target(metadata: dict, requested: str) -> str:
    """``auto`` keeps the file recipe. ``Z-Image`` reads the second-to-last layer of text-only Qwen3-4B."""
    recipe = text_encoder_recipe(metadata)
    choice = str(requested or "auto").strip().lower().replace("_", "-")
    if choice in ("", "auto"):
        return recipe
    if choice in ("z-image", "zimage"):
        if recipe == "flux2" and text_encoder_hidden_size(metadata) == 2560:
            return "zimage"
        raise ValueError(
            "Z-Image-Turbo only accepts the text-only Qwen3-4B encoder "
            "(svdq-int4-Qwen3-4B-text-Nunchaku). This file belongs to another model."
        )
    raise ValueError(f"Unknown text encoder target {requested!r}. Use auto or Z-Image.")


def iter_text_encoder_files() -> list[tuple[str, str]]:
    """Return (dropdown label, absolute path). Scans ~/weights and ComfyUI text_encoders."""
    found: list[tuple[str, dict]] = []
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
                if metadata is None or metadata.get("model_class") not in _TEXT_MODEL_CLASSES:
                    continue
                try:
                    text_encoder_recipe(metadata)
                except ValueError:
                    continue
                found.append((path, metadata))

    labels = [
        f"{os.path.basename(path)} [{text_encoder_kind(metadata)} {text_encoder_recipe(metadata)}]"
        for path, metadata in found
    ]
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    labeled: list[tuple[str, str]] = []
    for label, (path, _metadata_item) in zip(labels, found):
        if counts[label] > 1:
            label = f"{os.path.basename(os.path.dirname(path))}/{label}"
        labeled.append((label, path))
    labeled.sort(key=lambda item: item[0])
    return labeled


def resolve_text_encoder(choice: str) -> str:
    wanted = choice.split(" [", 1)[0]
    filename = os.path.basename(wanted)
    matches = []
    for label, path in iter_text_encoder_files():
        basename = os.path.basename(path)
        if choice == label or wanted in (basename, path) or choice == path or filename == basename:
            matches.append(os.path.realpath(path))
    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    if not unique:
        raise RuntimeError(f"Text encoder {choice!r} was not found. Place the Nunchaku Qwen single file in ~/weights or models/text_encoders.")
    raise RuntimeError(f"Text encoder {choice!r} matches more than one file: {unique}")


def token_ids_from_weights(sequences: list) -> torch.Tensor:
    """Pack a batch of ComfyUI tokenizer (token, weight) pairs into a 2-D id tensor."""
    rows = []
    width = 0
    for sequence in sequences:
        row = [_token_id(item) for item in sequence]
        width = max(width, len(row))
        rows.append(row)
    if not rows:
        raise RuntimeError("The text encoder received empty tokens.")
    ids = torch.full((len(rows), width), FLUX3_PAD_TOKEN, dtype=torch.long)
    for index, row in enumerate(rows):
        if row:
            ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return pad_to_multiple(ids)


def pad_to_multiple(ids: torch.Tensor, multiple: int = FLUX3_PAD_MULTIPLE) -> torch.Tensor:
    length = ids.shape[1]
    if length == 0:
        return torch.full((ids.shape[0], multiple), FLUX3_PAD_TOKEN, dtype=torch.long, device=ids.device)
    target = min(math.ceil(length / multiple) * multiple, FLUX3_MAX_SEQUENCE)
    if target == length:
        return ids
    if target < length:
        return ids[:, :target]
    extra = torch.full((ids.shape[0], target - length), FLUX3_PAD_TOKEN, dtype=torch.long, device=ids.device)
    return torch.cat([ids, extra], dim=1)


def attention_mask_from_ids(ids: torch.Tensor) -> torch.Tensor:
    """Right-side padding is excluded from Qwen's causal mask. Position ids still use 1, matching the FLUX.3 recipe."""
    mask = torch.ones_like(ids)
    for index in range(ids.shape[0]):
        content = (ids[index] != FLUX3_PAD_TOKEN).nonzero()
        if content.numel() == 0:
            mask[index] = 0
            continue
        last = int(content[-1])
        mask[index, last + 1 :] = 0
    return mask


def flux3_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).view(1, -1).expand_as(attention_mask)
    return positions.masked_fill(attention_mask == 0, 1)


def stack_flux3_hidden(hidden_states, layers: tuple[int, ...] = FLUX3_LAYER_INDEXES) -> torch.Tensor:
    missing = [index for index in layers if index >= len(hidden_states) or hidden_states[index] is None]
    if missing:
        raise RuntimeError(
            "Qwen3-VL did not return the intermediate layers FLUX.3 needs: "
            f"{missing}. hidden_states length is {len(hidden_states)}."
        )
    stacked = torch.stack([hidden_states[index] for index in layers], dim=1)
    batch, channels, seq, hidden = stacked.shape
    return stacked.permute(0, 2, 1, 3).reshape(batch, seq, channels * hidden)


def trim_qwen_image21(hidden: torch.Tensor, token_ids: torch.Tensor, image_spans: list[tuple[int, int]], keep_vision: bool):
    """Drop the system turn, and drop vision tokens by default. Slots are the insert positions of each image after the drop."""
    keep = torch.ones(token_ids.shape[0], dtype=torch.bool)
    starts = (token_ids == QWEN_IMAGE21_IM_START).nonzero(as_tuple=False).flatten()
    if starts.numel() > 1:
        keep[: int(starts[1])] = False
    slots = []
    if not keep_vision:
        for start, size in image_spans:
            slots.append(int(keep[:start].sum()))
            keep[start : start + size] = False
    trimmed = hidden[:, keep.to(hidden.device)]
    return trimmed, slots


def _sequence_images(sequence: list) -> list:
    images = []
    for item in sequence:
        token = item[0] if isinstance(item, (tuple, list)) else item
        if isinstance(token, dict) and token.get("type") == "image":
            images.append(token["data"])
    return images


def _expand_qwen_image21_tokens(sequence: list, image_token_counts: list[int]) -> tuple[list[int], list[tuple[int, int]]]:
    counts = iter(image_token_counts)
    ids: list[int] = []
    spans: list[tuple[int, int]] = []
    for item in sequence:
        token = item[0] if isinstance(item, (tuple, list)) else item
        if isinstance(token, dict) and token.get("type") == "image":
            count = next(counts)
            if count <= 0:
                raise RuntimeError("The reference image produced no vision tokens.")
            spans.append((len(ids), count))
            ids.extend([QWEN_IMAGE21_IMAGE_PAD] * count)
            continue
        if isinstance(token, dict):
            raise RuntimeError("The Qwen-Image 2.1 text encoder only accepts text and reference images.")
        ids.append(int(token))
    return ids, spans


def _vision_batch(images: list, merge: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    import comfy.text_encoders.qwen_vl as qwen_vl

    pixels = []
    grids = []
    counts = []
    for image in images:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        patches, grid = qwen_vl.process_qwen2vl_images(
            image[..., :3],
            patch_size=16,
            merge_size=merge,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )
        pixels.append(patches)
        grids.append(grid)
        counts.append(int(grid.prod().item()) // (merge * merge))
    return torch.cat(pixels, dim=0), torch.cat(grids, dim=0), counts


def last_hidden_prenorm(encoder: torch.nn.Module, **forward_kwargs) -> torch.Tensor:
    """Qwen-Image 2.1 reads the last hidden state before the final RMSNorm."""
    captured: dict[str, torch.Tensor] = {}

    def _keep_input(_module, args, _output):
        captured["pre"] = args[0]
        return args[0]

    handle = encoder.language_model.norm.register_forward_hook(_keep_input)
    try:
        encoder(**forward_kwargs)
    finally:
        handle.remove()
    hidden = captured.get("pre")
    if hidden is None:
        raise RuntimeError("The text encoder did not pass through language_model.norm, so the last layer Qwen-Image 2.1 needs is unavailable.")
    return hidden


def penultimate_hidden(encoder: torch.nn.Module, **forward_kwargs) -> torch.Tensor:
    """Output of the second-to-last Qwen3 layer, before the final RMSNorm. Z-Image reads this 2560-wide state."""
    layers = encoder.model.layers
    if len(layers) < 2:
        raise RuntimeError("Z-Image needs a Qwen3 text encoder with at least two layers.")
    captured: dict[str, torch.Tensor] = {}

    def _keep(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if isinstance(hidden, torch.Tensor):
            captured["hidden"] = hidden
        return output

    handle = layers[-2].register_forward_hook(_keep)
    try:
        encoder(**forward_kwargs)
    finally:
        handle.remove()
    hidden = captured.get("hidden")
    if hidden is None:
        raise RuntimeError("The text encoder did not run its second-to-last layer, so the Z-Image embedding is unavailable.")
    return hidden


class VitoomZImageCond(torch.nn.Module):
    """Z-Image-Turbo Qwen3-4B. encode_token_weights returns (batch, seq, 2560)."""

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.dtypes = {torch.bfloat16}
        self.execution_device = None

    def set_clip_options(self, options):
        if "execution_device" in options:
            self.execution_device = options["execution_device"]

    def reset_clip_options(self):
        self.execution_device = None

    def encode_token_weights(self, token_weight_pairs):
        sequences = token_weight_pairs["qwen3_4b"]
        device = self.execution_device
        if device is None:
            device = next(self.encoder.parameters()).device
        ids = _token_ids(sequences).to(device)
        attention_mask = attention_mask_from_ids(ids)
        hidden = penultimate_hidden(
            self.encoder,
            input_ids=ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).float()
        return hidden, None, {"attention_mask": attention_mask}


class VitoomFlux2Cond(torch.nn.Module):
    """Klein Qwen3. encode_token_weights returns (batch, seq, 3 * hidden)."""

    def __init__(self, encoder: torch.nn.Module, token_key: str):
        super().__init__()
        self.encoder = encoder
        self.token_key = token_key
        self.dtypes = {torch.bfloat16}
        self.execution_device = None

    def set_clip_options(self, options):
        if "execution_device" in options:
            self.execution_device = options["execution_device"]

    def reset_clip_options(self):
        self.execution_device = None

    def encode_token_weights(self, token_weight_pairs):
        sequences = token_weight_pairs[self.token_key]
        device = self.execution_device
        if device is None:
            device = next(self.encoder.parameters()).device
        ids = _token_ids(sequences).to(device)
        attention_mask = attention_mask_from_ids(ids)
        output = self.encoder(
            input_ids=ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("The text encoder did not return hidden_states. FLUX.2 Klein needs layers 9, 18, and 27.")
        # transformers hidden_states[k] is the input of layer k, matching the state ComfyUI captures before entering that layer.
        embeds = stack_flux3_hidden(hidden_states, FLUX2_LAYER_INDEXES).float()
        return embeds, None


class VitoomFlux3Cond(torch.nn.Module):
    """Wrapper for ComfyUI CLIP. encode_token_weights returns (batch, seq, 20480)."""

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.dtypes = {torch.bfloat16}
        self.execution_device = None

    def set_clip_options(self, options):
        if "execution_device" in options:
            self.execution_device = options["execution_device"]

    def reset_clip_options(self):
        self.execution_device = None

    def encode_token_weights(self, token_weight_pairs):
        sequences = token_weight_pairs["qwen3vl_4b"]
        device = self.execution_device
        if device is None:
            device = next(self.encoder.parameters()).device
        ids = token_ids_from_weights(sequences).to(device)
        attention_mask = attention_mask_from_ids(ids)
        position_ids = flux3_position_ids(attention_mask)
        output = self.encoder(
            input_ids=ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("The text encoder did not return hidden_states. FLUX.3 needs layers 4 through 32.")
        embeds = stack_flux3_hidden(hidden_states).float()
        return embeds, None


class VitoomQwenImage21Cond(torch.nn.Module):
    """Qwen3-VL-8B. encode_token_weights returns (batch, seq, 4096) plus reference-image slots."""

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.dtypes = {torch.bfloat16}
        self.execution_device = None
        visual = getattr(getattr(encoder, "model", None), "visual", None)
        self.merge_size = int(getattr(visual, "spatial_merge_size", 2))

    def set_clip_options(self, options):
        if "execution_device" in options:
            self.execution_device = options["execution_device"]

    def reset_clip_options(self):
        self.execution_device = None

    def encode_token_weights(self, token_weight_pairs):
        sequences = token_weight_pairs["qwen3vl_8b"]
        keep_vision = bool(token_weight_pairs.get("keep_vision", False))
        device = self.execution_device
        if device is None:
            device = next(self.encoder.parameters()).device
        rows = []
        masks = []
        slots = None
        for sequence in sequences:
            images = _sequence_images(sequence)
            if images:
                pixels, grid, counts = _vision_batch(images, self.merge_size)
            else:
                pixels, grid, counts = None, None, []
            ids, spans = _expand_qwen_image21_tokens(sequence, counts)
            token_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = torch.ones_like(token_ids)
            forward_kwargs = {
                "input_ids": token_ids,
                "attention_mask": attention_mask,
                "output_hidden_states": True,
                "use_cache": False,
            }
            if pixels is not None:
                type_ids = torch.zeros_like(token_ids)
                type_ids[token_ids == QWEN_IMAGE21_IMAGE_PAD] = 1
                forward_kwargs.update(
                    pixel_values=pixels.to(device=device, dtype=torch.float32),
                    image_grid_thw=grid.to(device=device),
                    mm_token_type_ids=type_ids,
                )
            hidden = last_hidden_prenorm(self.encoder, **forward_kwargs)
            trimmed, sequence_slots = trim_qwen_image21(hidden, token_ids[0], spans, keep_vision)
            if sequence_slots:
                if len(sequences) != 1:
                    raise RuntimeError("Encode only one prompt at a time when reference images are attached.")
                slots = sequence_slots
            rows.append(trimmed[0])
            masks.append(torch.ones(trimmed.shape[1], dtype=torch.long, device=trimmed.device))
        width = max(row.shape[0] for row in rows)
        hidden_size = rows[0].shape[-1]
        embeds = rows[0].new_zeros(len(rows), width, hidden_size)
        attention = torch.zeros(len(rows), width, dtype=torch.long, device=embeds.device)
        for index, (row, mask) in enumerate(zip(rows, masks)):
            embeds[index, : row.shape[0]] = row
            attention[index, : mask.shape[0]] = mask
        extra = {}
        if not bool(attention.all()):
            extra["attention_mask"] = attention
        if slots:
            extra["image_slots"] = slots
        if extra:
            return embeds.float(), None, extra
        return embeds.float(), None


def load_nunchaku_clip(path: str, options: dict | None = None, disable_dynamic: bool = False):
    """Load a quantized Qwen3-VL and return a ComfyUI CLIP."""
    del disable_dynamic
    import comfy.hooks
    import comfy.model_management as model_management
    import comfy.model_patcher
    import comfy.sd
    from comfy.text_encoders.flux import Flux3Tokenizer

    from vitoom_nunchaku.runtime.checkpoint import read_header
    from vitoom_nunchaku.runtime.env import ensure_nunchaku
    from vitoom_nunchaku.runtime.load import resolve_device

    options = dict(options or {})
    ensure_nunchaku()

    metadata, _tensors = read_header(path)
    recipe = text_encoder_target(metadata, str(options.get("target") or "auto"))
    load_device = resolve_device(options.get("device") or "default")
    offload_device = model_management.text_encoder_offload_device()
    if recipe in ("flux2", "zimage"):
        from nunchaku import NunchakuQwenEncoderModel

        encoder = NunchakuQwenEncoderModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device=offload_device,
        )
    else:
        from nunchaku import NunchakuQwen3VLEncoderModel

        encoder = NunchakuQwen3VLEncoderModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device=offload_device,
        )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    if recipe == "zimage":
        from comfy.text_encoders.z_image import ZImageTokenizer

        cond = VitoomZImageCond(encoder)
        tokenizer = ZImageTokenizer()
    elif recipe == "qwen_image21":
        from comfy.text_encoders.qwen_image21 import QwenImage21Tokenizer

        cond = VitoomQwenImage21Cond(encoder)
        tokenizer = QwenImage21Tokenizer()
    elif recipe == "flux2":
        from comfy.text_encoders.flux import KleinTokenizer, KleinTokenizer8B

        if text_encoder_hidden_size(metadata) == 4096:
            cond = VitoomFlux2Cond(encoder, "qwen3_8b")
            tokenizer = KleinTokenizer8B()
        else:
            cond = VitoomFlux2Cond(encoder, "qwen3_4b")
            tokenizer = KleinTokenizer()
    else:
        cond = VitoomFlux3Cond(encoder)
        tokenizer = Flux3Tokenizer()
    clip = comfy.sd.CLIP(no_init=True)
    clip.cond_stage_model = cond
    clip.tokenizer = tokenizer
    clip.patcher = comfy.model_patcher.ModelPatcher(
        cond,
        load_device=load_device,
        offload_device=offload_device,
    )
    clip.patcher.is_clip = True
    clip.patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
    clip.layer_idx = None
    clip.use_clip_schedule = False
    clip.apply_hooks_to_conds = None
    clip.tokenizer_options = {}
    clip.patcher.cached_patcher_init = (load_nunchaku_text_encoder_patcher, (path, options))
    return clip


def load_nunchaku_text_encoder_patcher(path: str, options: dict | None = None, disable_dynamic: bool = False):
    return load_nunchaku_clip(path, options, disable_dynamic=disable_dynamic).patcher


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


def _token_ids(sequences: list) -> torch.Tensor:
    """Pack ids into a 2-D tensor using the length the tokenizer already padded. Do not align to FLUX.3's multiple of 80."""
    rows = []
    width = 0
    for sequence in sequences:
        row = [_token_id(item) for item in sequence]
        width = max(width, len(row))
        rows.append(row)
    if not rows:
        raise RuntimeError("The text encoder received empty tokens.")
    ids = torch.full((len(rows), width), FLUX3_PAD_TOKEN, dtype=torch.long)
    for index, row in enumerate(rows):
        if row:
            ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return ids


def _token_id(item) -> int:
    token = item[0] if isinstance(item, (tuple, list)) else item
    if isinstance(token, dict):
        raise RuntimeError("This text encoder only handles text. For FLUX.3 reference images, use ReferenceLatent instead of sending them into CLIP.")
    return int(token)
