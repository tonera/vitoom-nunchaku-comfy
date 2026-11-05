from __future__ import annotations

import json

from comfy_api.latest import io

from vitoom_nunchaku.nodes.listing import is_placeholder, model_choices
from vitoom_nunchaku.runtime.checkpoint import filename_from_choice, read_checkpoint
from vitoom_nunchaku.runtime.env import NunchakuUnavailable, ensure_nunchaku
from vitoom_nunchaku.runtime.patcher import hardware_hint


class VitoomNunchakuModelInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VitoomNunchakuModelInfo",
            display_name="Vitoom Nunchaku Model Info",
            category="loaders/vitoom-nunchaku",
            description="Read quantized checkpoint metadata and report whether it matches the current GPU.",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=model_choices(),
                    tooltip="A nunchaku checkpoint in diffusion_models.",
                ),
            ],
            outputs=[
                io.String.Output(tooltip="model_class, precision, rank, and whether the hardware is compatible."),
            ],
        )

    @classmethod
    def execute(cls, model_name: str) -> io.NodeOutput:
        if is_placeholder(model_name):
            raise RuntimeError("No nunchaku checkpoint found.")
        import folder_paths

        filename = filename_from_choice(model_name)
        path = folder_paths.get_full_path_or_raise("diffusion_models", filename)
        info = read_checkpoint(path)
        lines = [
            f"file: {filename}",
            f"model_class: {info.model_class}",
            f"precision: {info.precision}",
            f"rank: {info.rank}",
            f"flow_map: {info.config.get('flow_map')}",
            f"layers: {info.config.get('num_layers')}",
            f"single_layers: {info.config.get('num_single_layers')}",
            f"config: {json.dumps(info.config, ensure_ascii=False, sort_keys=True)}",
            f"quantization_config: {json.dumps(info.quantization_config, ensure_ascii=False, sort_keys=True)}",
        ]
        try:
            ensure_nunchaku()
            import torch

            if torch.cuda.is_available():
                hint = hardware_hint(info.quantization_config, "cuda")
                lines.append("hardware: compatible" if hint is None else f"hardware: {hint}")
            else:
                lines.append("hardware: no CUDA, cannot determine compatibility")
        except NunchakuUnavailable as exc:
            lines.append(f"nunchaku: {exc}")
        return io.NodeOutput("\n".join(lines))
