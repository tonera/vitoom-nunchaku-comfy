"""Use a Z-Image-Turbo checkpoint header to check that keys fill a ComfyUI NextDiT."""

import os
import unittest

import torch

from vitoom_nunchaku.archs.base import QuantConfig, parse_quant_config
from vitoom_nunchaku.archs.zimage import ZImageSpec, _group_size, _guard_down, clamp_turbo_ffn_activation
from vitoom_nunchaku.runtime.checkpoint import read_checkpoint

_CHECKPOINTS = [
    os.path.expanduser("~/models/Z-Image-Turbo-Nunchaku/svdq-fp4_r64-Z-Image-Turbo-Nunchaku.safetensors"),
    os.path.expanduser("~/models/Z-Image-Turbo-Nunchaku/svdq-int4_r64-Z-Image-Turbo-Nunchaku.safetensors"),
]

_DTYPE = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I8": torch.int8,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def _meta_from_header(tensors: dict) -> dict[str, torch.Tensor]:
    state = {}
    for key, meta in tensors.items():
        state[key] = torch.empty(meta["shape"], dtype=_DTYPE[meta["dtype"]], device="meta")
    return state


class ZImageKeyMapTest(unittest.TestCase):
    def test_checkpoint_covers_patched_model(self):
        present = [path for path in _CHECKPOINTS if os.path.isfile(path)]
        if not present:
            self.skipTest("No Z-Image-Turbo checkpoint")
        try:
            import comfy.supported_models  # noqa: F401
        except Exception as exc:
            self.skipTest(f"Cannot import ComfyUI: {exc}")
        for path in present:
            with self.subTest(path=os.path.basename(path)):
                self._check(path)

    def _check(self, path: str):
        info = read_checkpoint(path)
        spec = ZImageSpec()
        qcfg = parse_quant_config(info.quantization_config, spec)
        spec.validate(info.config, qcfg)
        converted = spec.convert_state_dict(_meta_from_header(info.tensors))
        self.assertEqual(len(converted), len(info.tensors))

        model_config = spec.comfy_model_config(info.config)
        model_config.set_inference_dtype(torch.bfloat16, None)
        model = model_config.get_model({}, "", device=torch.device("meta"))
        spec.patch_model(model.diffusion_model, qcfg, torch.bfloat16)
        reference = model.diffusion_model.state_dict()
        missing = sorted(set(reference) - set(converted))
        extra = sorted(set(converted) - set(reference))
        self.assertEqual(extra[:12], [], extra[:12])
        self.assertEqual(missing[:12], [], missing[:12])
        self.assertEqual(set(converted), set(reference))
        for key, tensor in converted.items():
            self.assertEqual(tuple(tensor.shape), tuple(reference[key].shape), key)
            self.assertEqual(tensor.dtype, reference[key].dtype, key)
        self.assertIn("layers.0.feed_forward.down.weight", converted)
        self.assertNotIn("layers.1.feed_forward.down.weight", converted)
        self.assertIn("layers.1.feed_forward.down.qweight", converted)


class ZImageGuardTest(unittest.TestCase):
    def test_group_size_reads_the_weight_block(self):
        self.assertEqual(_group_size({"weight": {"group_size": 16}}), 16)
        with self.assertRaises(ValueError):
            _group_size({"group_size": 16, "weight": {"group_size": 64}})
        spec = ZImageSpec()
        quant = QuantConfig({"weight": {"dtype": "fp4_e2m1_all", "group_size": 32}}, "nvfp4", 32, None)
        with self.assertRaises(ValueError):
            spec.validate({"_class_name": "ZImageTransformer2DModel"}, quant)

    def test_turbo_guard_follows_the_wheel(self):
        self.assertTrue(_guard_down("layers.0", set(), True))
        self.assertFalse(_guard_down("layers.1", set(), True))
        self.assertFalse(_guard_down("layers.0", {"layers.0.feed_forward.net.2"}, True))
        self.assertFalse(_guard_down("layers.0", set(), False))

        hidden = torch.zeros(2, 10079)
        self.assertIs(clamp_turbo_ffn_activation(hidden), hidden)
        hidden[0, 10078] = 5000
        guarded = clamp_turbo_ffn_activation(hidden)
        self.assertEqual(guarded[0, 10078].item(), 4096)
        self.assertEqual(hidden[0, 10078].item(), 5000)


if __name__ == "__main__":
    unittest.main()
