from __future__ import annotations

from comfy_api.latest import ComfyExtension, io

from vitoom_nunchaku.nodes.info import VitoomNunchakuModelInfo
from vitoom_nunchaku.nodes.loader import VitoomNunchakuDiTLoader
from vitoom_nunchaku.nodes.lora import VitoomNunchakuLoraLoader
from vitoom_nunchaku.nodes.pup import VitoomNunchakuPromptUpsampler
from vitoom_nunchaku.nodes.flux_clip import VitoomNunchakuFluxClipLoader
from vitoom_nunchaku.nodes.text_encoder import VitoomNunchakuTextEncoderLoader
from vitoom_nunchaku.runtime.env import warn_comfyui_version, warn_upstream_plugin


class VitoomNunchakuExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        warn_comfyui_version()
        warn_upstream_plugin()
        return [
            VitoomNunchakuDiTLoader,
            VitoomNunchakuTextEncoderLoader,
            VitoomNunchakuFluxClipLoader,
            VitoomNunchakuLoraLoader,
            VitoomNunchakuPromptUpsampler,
            VitoomNunchakuModelInfo,
        ]


def comfy_entrypoint():
    return VitoomNunchakuExtension()
