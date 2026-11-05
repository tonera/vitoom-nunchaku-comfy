"""LoRA fusion must be redone after ComfyUI replaces the low-rank parameters."""

import unittest

import torch
from torch import nn

from vitoom_nunchaku.runtime.lora import _invalidate_lora_on_load, _lowrank_intact


class _Proj(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_down = nn.Parameter(torch.zeros(2, 2))


class _Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _Proj()


class LoraReloadTest(unittest.TestCase):
    def test_replaced_parameter_is_not_intact(self):
        root = _Root()
        stamp = (("block", id(root.block.proj_down)),)
        self.assertTrue(_lowrank_intact(root, stamp))
        self.assertTrue(_lowrank_intact(root, ()))
        root.block.proj_down = nn.Parameter(torch.zeros(2, 2))
        self.assertFalse(_lowrank_intact(root, stamp))

    def test_load_callback_forgets_a_fused_signature(self):
        root = _Root()
        root._vitoom_lora_stamp = (("block", id(root.block.proj_down)),)
        root._vitoom_lora_signature = (("/tmp/lora.safetensors", 1.0),)

        class _Model:
            diffusion_model = root

        class _Patcher:
            model = _Model()

        _invalidate_lora_on_load(_Patcher())
        self.assertIsNone(root._vitoom_lora_signature)

    def test_load_callback_leaves_an_unfused_model_alone(self):
        root = _Root()
        root._vitoom_lora_signature = ()

        class _Model:
            diffusion_model = root

        class _Patcher:
            model = _Model()

        _invalidate_lora_on_load(_Patcher())
        self.assertEqual(root._vitoom_lora_signature, ())


if __name__ == "__main__":
    unittest.main()
