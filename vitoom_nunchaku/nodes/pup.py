from __future__ import annotations

from comfy_api.latest import io

from vitoom_nunchaku.nodes.listing import is_placeholder
from vitoom_nunchaku.runtime.pup import iter_pup_files, resolve_pup, upsample_prompt

_EMPTY = "(no Qwen3.5 prompt upsampler found)"


def _choices() -> list[str]:
    found = [label for label, _path in iter_pup_files()]
    return found or [_EMPTY]


def _devices() -> list[str]:
    import torch

    devices = ["default"]
    if torch.cuda.is_available():
        devices.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    return devices


class VitoomNunchakuPromptUpsampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VitoomNunchakuPromptUpsampler",
            display_name="Vitoom Nunchaku Prompt Upsampler",
            category="conditioning/vitoom-nunchaku",
            description="Expand a short prompt with the quantized Qwen3.5 FLUX.3 Klein upsampler. Connect the string to CLIP Text Encode.",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=_choices(),
                    tooltip="Quantized Qwen3.5 PUP safetensors. The processor and system_prompts.json must be in the same directory.",
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip="Short request. With width and height set, a TARGET CANVAS line is added.",
                ),
                io.Int.Input("max_new_tokens", default=1024, min=16, max=3584, step=16),
                io.Int.Input(
                    "width",
                    default=0,
                    min=0,
                    max=8192,
                    step=16,
                    tooltip="Canvas width. Leave 0 to skip the canvas line. Set both width and height together.",
                ),
                io.Int.Input(
                    "height",
                    default=0,
                    min=0,
                    max=8192,
                    step=16,
                    tooltip="Canvas height. Leave 0 to skip the canvas line.",
                ),
                io.Combo.Input("device", options=_devices(), default="default"),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Reference images, in batch order. Connected images use the edit prompt.",
                ),
            ],
            outputs=[
                io.String.Output(tooltip="Expanded caption. Connect this to the positive CLIP Text Encode."),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_name: str,
        text: str,
        max_new_tokens: int,
        width: int,
        height: int,
        device: str,
        image=None,
    ) -> io.NodeOutput:
        if is_placeholder(model_name) or model_name == _EMPTY:
            raise RuntimeError(
                "No Qwen3.5 prompt upsampler found. Keep the quantized safetensors next to system_prompts.json."
            )
        caption = upsample_prompt(
            resolve_pup(model_name),
            text,
            images=image,
            max_new_tokens=max_new_tokens,
            width=width,
            height=height,
            device=device,
        )
        return io.NodeOutput(caption)
