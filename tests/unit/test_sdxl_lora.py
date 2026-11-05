"""Map kohya SDXL LoRAs onto the patched ComfyUI UNet."""

import unittest

_SD15 = "/home/tonera/loras/more_details.safetensors"
_CONV = "/home/tonera/loras/Ass Adjust_alpha1.0_rank4_noxattn_last.safetensors"
_SGM = "/home/tonera/loras/halterneck_bikini.safetensors"
_UNET = None


def _diffusion():
    global _UNET
    if _UNET is not None:
        return _UNET
    import json

    import torch

    from vitoom_nunchaku.archs.base import QuantConfig
    from vitoom_nunchaku.archs.sdxl import SDXLSpec
    from vitoom_nunchaku.runtime.checkpoint import read_header

    path = "/home/tonera/project/ComfyUI/models/diffusion_models/svdq-fp4_r32-oneObsession_v18.safetensors"
    meta, _ = read_header(path)
    config = json.loads(meta["config"])
    spec = SDXLSpec()
    spec.validate(config, QuantConfig({}, "nvfp4", 32, None))
    model = spec.comfy_model_config(config).get_model({}, device=torch.device("meta"))
    spec.patch_model(model.diffusion_model, QuantConfig({}, "nvfp4", 32, None), torch.bfloat16)
    _UNET = model.diffusion_model
    return _UNET


class SdxlLoraMapTest(unittest.TestCase):
    def test_sd15_lora_does_not_match_sdxl(self):
        from vitoom_nunchaku.runtime.lora_models import sdxl_pairs

        with self.assertRaises(ValueError):
            sdxl_pairs(_diffusion(), _SD15)

    def test_conv_lora_keeps_kernel(self):
        from vitoom_nunchaku.runtime.lora_models import sdxl_pairs

        pairs = sdxl_pairs(_diffusion(), _CONV)
        down, up = pairs["input_blocks.1.0.in_layers.2"]
        self.assertEqual(tuple(down.shape[2:]), (3, 3))
        self.assertEqual(tuple(up.shape[2:]), (1, 1))
        self.assertIn("input_blocks.3.0.op", pairs)
        down, up = pairs["middle_block.1.transformer_blocks.0.attn1.to_qkv"]
        self.assertEqual(tuple(down.shape[1:]), (1280,))
        self.assertEqual(tuple(up.shape[:1]), (1280 * 3,))

    def test_sgm_names_match_comfy_blocks(self):
        from vitoom_nunchaku.runtime.lora_models import sdxl_pairs

        pairs = sdxl_pairs(_diffusion(), _SGM)
        self.assertIn("output_blocks.5.1.transformer_blocks.1.ff.net.2", pairs)
        self.assertIn("output_blocks.5.1.transformer_blocks.1.attn1.to_qkv", pairs)
        down, up = pairs["output_blocks.5.1.transformer_blocks.1.attn1.to_qkv"]
        self.assertEqual(up.shape[0], 3 * down.shape[1])


if __name__ == "__main__":
    unittest.main()
