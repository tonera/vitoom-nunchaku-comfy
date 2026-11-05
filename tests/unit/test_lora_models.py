"""Map FLUX.2, Qwen-Image 2.1, and Z-Image LoRAs onto ComfyUI module paths."""

import os
import unittest

_FLUX2_COMFY = "/home/tonera/loras/A2R_Klein_Standard.safetensors"
_FLUX2_DIFFUSERS = "/home/tonera/loras/flux2-klein-sakimi-v1.safetensors"
_QWEN_COMFY = "/home/tonera/loras/Qwen2.1_Anything2RealCharacters.safetensors"
_QWEN_DIFFUSERS = "/home/tonera/loras/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
_ZIMAGE = "/home/tonera/loras/NIceAsians_Zimage.safetensors"


class LoraModelMapTest(unittest.TestCase):
    def test_flux2_comfy_splits_single_blocks(self):
        self._require(_FLUX2_COMFY)
        from vitoom_nunchaku.runtime.lora_models import pairs_for

        pairs = pairs_for("flux2", _FLUX2_COMFY)
        qkv_down, qkv_up = pairs["single_blocks.0.qkv_proj"]
        mlp_down, mlp_up = pairs["single_blocks.0.mlp_fc1"]
        self.assertEqual(tuple(qkv_down.shape), (16, 4096))
        self.assertEqual(tuple(qkv_up.shape), (12288, 16))
        self.assertEqual(tuple(mlp_up.shape), (24576, 16))
        self.assertTrue(qkv_down.equal(mlp_down))
        proj_down, _proj_up = pairs["single_blocks.0.out_proj"]
        fc2_down, _fc2_up = pairs["single_blocks.0.mlp_fc2"]
        self.assertEqual(tuple(proj_down.shape), (16, 4096))
        self.assertEqual(tuple(fc2_down.shape), (16, 12288))
        self.assertEqual(tuple(pairs["double_blocks.0.img_attn.qkv"][1].shape), (12288, 16))
        self.assertEqual(tuple(pairs["double_blocks.0.img_mlp.0"][1].shape), (24576, 16))

    def test_flux2_diffusers_fuses_qkv(self):
        self._require(_FLUX2_DIFFUSERS)
        from vitoom_nunchaku.runtime.lora_models import pairs_for

        pairs = pairs_for("flux2", _FLUX2_DIFFUSERS)
        _down, up = pairs["double_blocks.0.img_attn.qkv"]
        self.assertEqual(tuple(up.shape), (12288, 64))
        self.assertEqual(tuple(pairs["double_blocks.0.txt_attn.qkv"][1].shape), (12288, 64))
        self.assertEqual(tuple(pairs["single_blocks.0.qkv_proj"][1].shape), (12288, 64))
        self.assertEqual(tuple(pairs["single_blocks.0.mlp_fc1"][1].shape), (24576, 64))
        self.assertEqual(tuple(pairs["single_blocks.0.mlp_fc2"][0].shape), (64, 12288))
        self.assertEqual(len(pairs), 8 * 8 + 24 * 4)

    def test_qwen_comfy_swaps_gate_up(self):
        self._require(_QWEN_COMFY)
        from safetensors import safe_open

        from vitoom_nunchaku.runtime.lora_models import pairs_for

        pairs = pairs_for("qwen_image21", _QWEN_COMFY)
        _down, up = pairs["transformer_blocks.0.img_mlp.net.0.proj"]
        self.assertEqual(tuple(up.shape), (24576, 64))
        with safe_open(_QWEN_COMFY, framework="pt") as handle:
            original = handle.get_tensor("diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight")
        gate, value = original.chunk(2, dim=0)
        self.assertTrue(up[:12288].equal(value))
        self.assertTrue(up[12288:].equal(gate))
        self.assertEqual(tuple(pairs["transformer_blocks.0.attn.to_qkv"][1].shape), (12288, 192))
        self.assertIn("transformer_blocks.0.img_mlp.net.2", pairs)
        self.assertEqual(len(pairs), 32 * 4)

    def test_qwen_diffusers_keeps_up_then_gate(self):
        self._require(_QWEN_DIFFUSERS)
        from vitoom_nunchaku.runtime.lora_models import pairs_for

        pairs = pairs_for("qwen_image21", _QWEN_DIFFUSERS)
        _down, up = pairs["transformer_blocks.0.img_mlp.net.0.proj"]
        self.assertEqual(tuple(up.shape), (24576, 512))
        self.assertIn("modulation.1", pairs)
        self.assertIn("time_text_embed.timestep_embedder.linear_1", pairs)
        self.assertEqual(len(pairs), 32 * 4 + 3)

    def test_zimage_fuses_up_then_gate(self):
        self._require(_ZIMAGE)
        from vitoom_nunchaku.runtime.lora_models import pairs_for

        pairs = pairs_for("zimage", _ZIMAGE)
        down, up = pairs["layers.0.feed_forward.proj"]
        self.assertEqual(tuple(down.shape), (256, 3840))
        self.assertEqual(tuple(up.shape), (20480, 256))
        self.assertEqual(tuple(pairs["layers.0.feed_forward.down"][0].shape), (128, 10240))
        self.assertEqual(tuple(pairs["layers.0.attention.qkv"][1].shape), (11520, 384))
        self.assertEqual(tuple(pairs["layers.0.adaLN_modulation.0"][1].shape), (15360, 128))
        self.assertEqual(len(pairs), 30 * 5)

    def test_zimage_rejects_a_one_sided_swiglu_lora(self):
        import torch

        from vitoom_nunchaku.runtime.lora_models import _zimage

        state = {
            "layers.0.feed_forward.w1.lora_A.weight": torch.zeros(4, 8),
            "layers.0.feed_forward.w1.lora_B.weight": torch.zeros(8, 4),
            "layers.0.feed_forward.w2.lora_A.weight": torch.zeros(4, 16),
            "layers.0.feed_forward.w2.lora_B.weight": torch.zeros(8, 4),
        }
        with self.assertRaises(ValueError):
            _zimage(state)

    def _require(self, path: str) -> None:
        if not os.path.isfile(path):
            self.skipTest(path)


if __name__ == "__main__":
    unittest.main()
