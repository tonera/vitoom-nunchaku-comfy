"""Qwen3.5 prompt upsampler. W4 language-model weights, not a diffusion checkpoint.

The processor and ``system_prompts.json`` must sit next to the quantized safetensors.
Text-only requests use the t2i prompt. A reference image uses the i2i prompt.
"""

from __future__ import annotations

import json
import logging
import os

import torch

from vitoom_nunchaku.runtime.checkpoint import read_header

log = logging.getLogger("vitoom_nunchaku")

_MODEL_TYPE = "qwen3_5_pup"
_CACHE: dict[str, dict] = {}


def iter_pup_files() -> list[tuple[str, str]]:
    """Return ``(dropdown label, absolute path)`` for quantized PUP checkpoints."""
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for root in _search_roots():
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if not name.startswith(".")]
            for filename in filenames:
                if not filename.endswith(".safetensors") or "pup" not in filename.lower():
                    continue
                path = os.path.realpath(os.path.join(dirpath, filename))
                if path in seen:
                    continue
                seen.add(path)
                try:
                    metadata, _tensors = read_header(path)
                except Exception:
                    continue
                if str(metadata.get("model_type") or "") != _MODEL_TYPE:
                    continue
                found.append((f"{filename} [w4 pup]", path))
    found.sort(key=lambda item: item[0])
    return found


def resolve_pup(choice: str) -> str:
    filename = choice.split(" [", 1)[0]
    for label, path in iter_pup_files():
        if label == choice or os.path.basename(path) == filename:
            return path
    raise FileNotFoundError(
        f"Cannot find the Qwen3.5 prompt upsampler {choice!r}. "
        "Place the quantized safetensors next to its processor files."
    )


def upsample_prompt(
    path: str,
    prompt: str,
    *,
    images: torch.Tensor | None = None,
    max_new_tokens: int = 1024,
    width: int = 0,
    height: int = 0,
    device: str = "default",
) -> str:
    """Expand ``prompt`` into the caption FLUX.3 Klein should receive."""
    text = str(prompt or "").strip()
    if not text:
        raise ValueError("The prompt upsampler received an empty prompt.")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}.")
    if (width > 0) ^ (height > 0):
        raise ValueError("Set both canvas width and height, or leave both at 0.")
    if width > 0 and height > 0:
        text = f"TARGET CANVAS: {width}:{height}\n{text}"

    bundle = _bundle(path)
    torch_device = _resolve_device(device)
    holder = _load(path, bundle)
    model = holder["model"]
    processor = holder["processor"]
    system_prompts = holder["system_prompts"]
    pil_images = _pil_images(images)
    mode = "i2i" if pil_images else "t2i"
    messages = _messages(text, pil_images, system_prompts[mode])
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )

    import comfy.model_management as model_management

    model_management.unload_all_models()
    model_management.soft_empty_cache()
    model.to(torch_device)
    try:
        batch = inputs.to(model.device)
        input_tokens = int(batch["input_ids"].shape[1])
        with torch.inference_mode():
            output = model.generate(**batch, max_new_tokens=int(max_new_tokens), do_sample=False)
        sequences = output.sequences if hasattr(output, "sequences") else output
        caption = processor.decode(sequences[0, input_tokens:], skip_special_tokens=True).strip()
    finally:
        model.to("cpu")
        model_management.soft_empty_cache()
    if not caption:
        raise RuntimeError("The prompt upsampler returned an empty caption.")
    log.info("Qwen3.5 PUP %s caption, %d characters.", mode, len(caption))
    return caption


def _messages(prompt: str, images: list, system_prompt: str) -> list:
    if images:
        request = (
            "Reference images follow in order (the first image, the second image, ...).\n"
            f"User request: {prompt}\nOutput the caption."
        )
        content = [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": request}]
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]


def _pil_images(images: torch.Tensor | None) -> list:
    if images is None:
        return []
    from PIL import Image

    batch = images.detach().float().clamp(0, 1).cpu()
    converted = []
    for image in batch:
        array = (image.numpy() * 255).round().astype("uint8")
        converted.append(Image.fromarray(array))
    return converted


def _bundle(path: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    missing = [
        name
        for name in ("system_prompts.json", "tokenizer.json", "tokenizer_config.json")
        if not os.path.isfile(os.path.join(directory, name))
    ]
    if missing:
        raise FileNotFoundError(
            f"{os.path.basename(path)} needs {', '.join(missing)} in {directory}. "
            "Keep the processor files beside the quantized checkpoint."
        )
    return directory


def _load(path: str, bundle: str) -> dict:
    key = os.path.realpath(path)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    from nunchaku import NunchakuQwen3_5ForConditionalGeneration
    from transformers import AutoProcessor

    log.info("Loading Qwen3.5 prompt upsampler %s", key)
    model = NunchakuQwen3_5ForConditionalGeneration.from_pretrained(
        key,
        torch_dtype=torch.bfloat16,
        device="cpu",
        decode_cuda_graph=False,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(bundle)
    with open(os.path.join(bundle, "system_prompts.json"), encoding="utf-8") as handle:
        system_prompts = json.load(handle)
    for name in ("t2i", "i2i"):
        if name not in system_prompts:
            raise ValueError(f"system_prompts.json is missing the {name!r} prompt.")
    holder = {"model": model, "processor": processor, "system_prompts": system_prompts}
    _CACHE[key] = holder
    return holder


def _resolve_device(name: str) -> torch.device:
    import comfy.model_management as model_management

    if name in (None, "", "default"):
        return model_management.get_torch_device()
    return torch.device(name)


def _search_roots() -> list[str]:
    roots = [os.path.expanduser("~/models"), os.path.expanduser("~/weights")]
    try:
        import folder_paths

        roots.extend(folder_paths.get_folder_paths("text_encoders"))
    except Exception:
        pass
    unique = []
    seen = set()
    for root in roots:
        real = os.path.realpath(root)
        if real in seen:
            continue
        seen.add(real)
        unique.append(root)
    return unique
