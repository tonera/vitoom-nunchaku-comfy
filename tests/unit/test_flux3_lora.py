"""Map diffusers and ComfyUI FLUX.3 LoRAs onto the quantized block paths."""

import unittest

import torch

_COMFY = "/home/tonera/loras/alphajace_klein_comfy.safetensors"
_DIFFUSERS = "/home/tonera/loras/alphajace_klein_diffusers.safetensors"


class Flux3LoraMapTest(unittest.TestCase):
    def test_comfy_fuses_qkv_and_swaps_quantized_mlp(self):
        from safetensors import safe_open

        from vitoom_nunchaku.runtime.lora_models import flux3_pairs

        pairs = flux3_pairs(_COMFY)
        self.assertEqual(len(pairs), 5 * 4 + 5 * 4 + 5 * 4 + 28 * 4)
        down, up = pairs["content_mode_blocks.image.0.qkv"]
        self.assertEqual(tuple(down.shape), (96, 3072))
        self.assertEqual(tuple(up.shape), (9216, 96))
        self.assertEqual(tuple(pairs["single_blocks.27.mlp_out"][0].shape), (32, 9216))
        with safe_open(_COMFY, framework="pt") as handle:
            original = handle.get_tensor("diffusion_model.content_mode_blocks.image.0.mlp_in.lora_B.weight")
            cond = handle.get_tensor("diffusion_model.content_mode_blocks.image_cond.0.mlp_in.lora_B.weight")
            q = handle.get_tensor("diffusion_model.content_mode_blocks.image.0.q_proj.lora_B.weight")
        gate, value = original.chunk(2, dim=0)
        self.assertTrue(torch.equal(pairs["content_mode_blocks.image.0.mlp_in"][1], torch.cat([value, gate], dim=0)))
        self.assertTrue(torch.equal(pairs["content_mode_blocks.image_cond.0.mlp_in"][1], cond))
        self.assertTrue(torch.equal(up[:3072, :32], q))

    def test_diffusers_matches_comfy_mapping(self):
        from vitoom_nunchaku.runtime.lora_models import flux3_pairs

        comfy = flux3_pairs(_COMFY)
        diffusers = flux3_pairs(_DIFFUSERS)
        self.assertEqual(set(comfy), set(diffusers))
        for name in (
            "txt_mode_blocks.0.attn_out",
            "content_mode_blocks.image.0.mlp_in",
            "content_mode_blocks.image_cond.4.qkv",
            "single_blocks.0.mlp_out",
        ):
            self.assertTrue(torch.equal(comfy[name][0], diffusers[name][0]))
            self.assertTrue(torch.equal(comfy[name][1], diffusers[name][1]))


if __name__ == "__main__":
    unittest.main()
