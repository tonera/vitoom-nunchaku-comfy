from __future__ import annotations

import torch
from comfy_api.latest import io

from vitoom_nunchaku.nodes.listing import is_placeholder, model_choices
from vitoom_nunchaku.runtime.checkpoint import filename_from_choice
from vitoom_nunchaku.runtime.load import load_nunchaku_diffusion_model


def _devices() -> list[str]:
    devices = ["default"]
    if torch.cuda.is_available():
        devices.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    return devices


class VitoomNunchakuDiTLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VitoomNunchakuDiTLoader",
            display_name="Vitoom Nunchaku DiT Loader",
            category="loaders/vitoom-nunchaku",
            description="Load a tonera nunchaku quantized DiT single file and output a MODEL that can connect to KSampler.",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=model_choices(),
                    tooltip="A safetensors file in diffusion_models with nunchaku metadata. Brackets show precision and rank.",
                ),
                io.Combo.Input(
                    "device",
                    options=_devices(),
                    default="default",
                    tooltip="Defaults to ComfyUI's current GPU.",
                ),
                io.Combo.Input(
                    "dtype",
                    options=["auto", "bfloat16", "float16"],
                    default="auto",
                    tooltip="auto uses float16 on Turing (sm_75) and bfloat16 on other CUDA GPUs.",
                ),
                io.Combo.Input(
                    "attention",
                    options=["auto", "nunchaku_fp16", "comfy"],
                    default="auto",
                    advanced=True,
                    tooltip="Only ComfyUI native attention is active in this stage. nunchaku_fp16 is enabled in the fused-attention milestone.",
                ),
                io.Boolean.Input(
                    "disable_dynamic_vram",
                    default=False,
                    advanced=True,
                    tooltip="When enabled, use a non-dynamic ModelPatcher. Quantized layers always stay resident on the GPU as a whole.",
                ),
            ],
            outputs=[
                io.Model.Output(tooltip="Quantized DiT. Text encoders, VAE, and the sampler still use native ComfyUI nodes."),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_name: str,
        device: str,
        dtype: str,
        attention: str,
        disable_dynamic_vram: bool,
    ) -> io.NodeOutput:
        if is_placeholder(model_name):
            raise RuntimeError(
                "No nunchaku checkpoint found. Place an svdq-fp4, svdq-int4, or svdq-int8 safetensors file in "
                "ComfyUI models/diffusion_models."
            )
        import folder_paths

        filename = filename_from_choice(model_name)
        path = folder_paths.get_full_path_or_raise("diffusion_models", filename)
        model = load_nunchaku_diffusion_model(
            path,
            {
                "device": device,
                "dtype": dtype,
                "attention": attention,
                "disable_dynamic_vram": disable_dynamic_vram,
            },
            disable_dynamic=disable_dynamic_vram,
        )
        return io.NodeOutput(model)
