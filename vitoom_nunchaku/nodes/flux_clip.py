from __future__ import annotations

from comfy_api.latest import io

from vitoom_nunchaku.runtime.flux_clip import iter_t5_files, load_flux_clip, resolve_t5

_EMPTY_T5 = "(no Nunchaku T5)"
_EMPTY_CLIP = "(no CLIP-L)"


def _t5_choices() -> list[str]:
    found = iter_t5_files()
    if not found:
        return [_EMPTY_T5]
    return [label for label, _path in found]


def _clip_l_choices() -> list[str]:
    try:
        import folder_paths

        names = list(folder_paths.get_filename_list("text_encoders"))
    except Exception:
        names = []
    return names or [_EMPTY_CLIP]


class VitoomNunchakuFluxClipLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VitoomNunchakuFluxClipLoader",
            display_name="Vitoom Nunchaku Flux CLIP",
            category="loaders/vitoom-nunchaku",
            description="FLUX.1 text encoder. AWQ T5-XXL produces the 4096-wide sequence. CLIP-L produces the 768 pooled vector. Connect to CLIP Text Encode and FluxGuidance.",
            inputs=[
                io.Combo.Input(
                    "t5_name",
                    options=_t5_choices(),
                    tooltip="Nunchaku T5-XXL in ~/weights or models/text_encoders.",
                ),
                io.Combo.Input(
                    "clip_l_name",
                    options=_clip_l_choices(),
                    tooltip="Standard CLIP-L. FLUX.1 uses its pooled output, not the T5 sequence, for the vector embedding.",
                ),
                io.Combo.Input(
                    "device",
                    options=_devices(),
                    default="default",
                    tooltip="Defaults to ComfyUI's current GPU.",
                ),
            ],
            outputs=[
                io.Clip.Output(tooltip="FLUX.1 CLIP. Use CLIP Text Encode, then FluxGuidance."),
            ],
        )

    @classmethod
    def execute(cls, t5_name: str, clip_l_name: str, device: str) -> io.NodeOutput:
        if t5_name == _EMPTY_T5 or clip_l_name == _EMPTY_CLIP:
            raise RuntimeError(
                "FLUX.1 needs both the Nunchaku T5 file and a CLIP-L file in models/text_encoders."
            )
        clip = load_flux_clip(resolve_t5(t5_name), clip_l_name, {"device": device})
        return io.NodeOutput(clip)


def _devices() -> list[str]:
    import torch

    devices = ["default"]
    if torch.cuda.is_available():
        devices.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    return devices
