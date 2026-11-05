from __future__ import annotations

from comfy_api.latest import io

from vitoom_nunchaku.nodes.listing import is_placeholder, text_encoder_choices
from vitoom_nunchaku.runtime.text_encoder import load_nunchaku_clip, resolve_text_encoder


class VitoomNunchakuTextEncoderLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VitoomNunchakuTextEncoderLoader",
            display_name="Vitoom Nunchaku Text Encoder",
            category="loaders/vitoom-nunchaku",
            description="Load a Nunchaku Qwen text encoder. Qwen3-VL 4B outputs FLUX.3 CLIP (8 layers concatenated to 20480). Qwen3-VL 8B Instruct outputs Qwen-Image 2.1 CLIP (last layer, 4096). Text-only Qwen3 outputs FLUX.2 Klein CLIP (layers 9, 18, and 27; 7680 for 4B).",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=text_encoder_choices(),
                    tooltip="A Nunchaku Qwen file in ~/weights or models/text_encoders. VL 4B connects to FLUX.3, VL 8B Instruct connects to Text Encode Qwen Image 2.1, and text-only Qwen3 connects to FLUX.2 Klein CLIP Text Encode.",
                ),
                io.Combo.Input(
                    "device",
                    options=_devices(),
                    default="default",
                    tooltip="Defaults to ComfyUI's current GPU. Weights are W4A16 and are not subject to the W4A4 precision limits.",
                ),
            ],
            outputs=[
                io.Clip.Output(tooltip="Connect FLUX.3 and FLUX.2 Klein to CLIP Text Encode. Connect Qwen-Image 2.1 to Text Encode Qwen Image 2.1."),
            ],
        )

    @classmethod
    def execute(cls, model_name: str, device: str) -> io.NodeOutput:
        if is_placeholder(model_name):
            raise RuntimeError(
                "No Nunchaku Qwen text encoder found. Place the safetensors file in ~/weights or ComfyUI models/text_encoders."
            )
        path = resolve_text_encoder(model_name)
        clip = load_nunchaku_clip(path, {"device": device})
        return io.NodeOutput(clip)


def _devices() -> list[str]:
    import torch

    devices = ["default"]
    if torch.cuda.is_available():
        devices.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    return devices
