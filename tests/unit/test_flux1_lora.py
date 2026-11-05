"""Map the logo-design FLUX.1 LoRA onto ComfyUI module paths."""

import os
import unittest

_LORA = "/home/tonera/loras/FLUX-dev-lora-Logo-Design.safetensors"
_KONTEXT = "/home/tonera/loras/FLUX-Kontext-dev-lora-highresfix.safetensors"


class Flux1LoraMapTest(unittest.TestCase):
    def test_logo_lora_covers_quantized_flux_modules(self):
        if not os.path.isfile(_LORA):
            self.skipTest("logo LoRA is not on this machine")
        from vitoom_nunchaku.runtime.lora import flux_lora_pairs

        pairs = flux_lora_pairs(_LORA)
        self.assertEqual(len(pairs), 19 * 10 + 38 * 5)
        qkv_down, qkv_up = pairs["double_blocks.0.img_attn.qkv"]
        self.assertEqual(tuple(qkv_down.shape), (4, 3072))
        self.assertEqual(tuple(qkv_up.shape), (9216, 4))
        proj_down, proj_up = pairs["single_blocks.0.out_proj"]
        mlp_down, mlp_up = pairs["single_blocks.0.mlp_fc2"]
        self.assertEqual(tuple(proj_down.shape), (4, 3072))
        self.assertEqual(tuple(mlp_down.shape), (4, 12288))
        self.assertEqual(tuple(proj_up.shape), tuple(mlp_up.shape))
        mod_down, mod_up = pairs["double_blocks.0.img_mod.lin"]
        self.assertEqual(tuple(mod_down.shape), (4, 3072))
        self.assertEqual(tuple(mod_up.shape), (18432, 4))
        # Contiguous ComfyUI chunks become the per-channel layout used by the AWQ projection.
        self.assertFalse(torch_equal_chunks(mod_up))


    def test_kontext_highresfix_uses_the_same_modules(self):
        if not os.path.isfile(_KONTEXT):
            self.skipTest("Kontext highresfix LoRA is not on this machine")
        from vitoom_nunchaku.runtime.lora import flux_lora_pairs

        pairs = flux_lora_pairs(_KONTEXT)
        self.assertEqual(len(pairs), 19 * 10 + 38 * 5)
        down, up = pairs["double_blocks.0.img_attn.qkv"]
        self.assertEqual(tuple(down.shape), (32, 3072))
        self.assertEqual(tuple(up.shape), (9216, 32))
        down, up = pairs["single_blocks.0.mlp_fc2"]
        self.assertEqual(tuple(down.shape), (32, 12288))


def torch_equal_chunks(up):
    import torch

    chunks = up.view(6, up.shape[0] // 6, up.shape[1]).transpose(0, 1).reshape_as(up)
    return torch.equal(chunks, up)


if __name__ == "__main__":
    unittest.main()
