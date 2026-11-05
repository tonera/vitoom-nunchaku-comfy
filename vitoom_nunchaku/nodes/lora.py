from __future__ import annotations

from comfy_api.latest import io

from vitoom_nunchaku.nodes.listing import is_placeholder, lora_choices


class VitoomNunchakuLoraLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VitoomNunchakuLoraLoader",
            display_name="Vitoom Nunchaku LoRA Loader",
            category="loaders/vitoom-nunchaku",
            description="Fuse a LoRA into FLUX.1, FLUX.2 Klein, FLUX.3, Qwen-Image 2.1, Z-Image, or SDXL. Chain nodes to stack LoRAs. SDXL text-encoder weights in the file are ignored.",
            inputs=[
                io.Model.Input("model", tooltip="Output of VitoomNunchakuDiTLoader."),
                io.Combo.Input("lora_name", options=lora_choices(), tooltip="A LoRA file in models/loras."),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Strength scales the low-rank branch. Adding ranks from multiple LoRAs makes inference slower.",
                ),
            ],
            outputs=[
                io.Model.Output(tooltip="MODEL with a LoRA stack."),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name: str, strength: float) -> io.NodeOutput:
        if is_placeholder(lora_name):
            raise RuntimeError("No LoRA file found. Place a safetensors file in ComfyUI models/loras.")
        import folder_paths

        from vitoom_nunchaku.runtime.lora import append_lora

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return io.NodeOutput(append_lora(model, path, strength))
