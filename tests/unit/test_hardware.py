import unittest
from unittest.mock import patch

import torch

from vitoom_nunchaku.runtime.patcher import assert_hardware, hardware_hint


def _capability(pair):
    def get_device_capability(index=0):
        return pair

    return get_device_capability


class HardwareHintTest(unittest.TestCase):
    def test_blackwell_rejects_int4(self):
        quant = {"weight": {"dtype": "int4", "group_size": 64}, "precision": "int4"}
        with patch("torch.cuda.get_device_capability", _capability((12, 0))):
            hint = hardware_hint(quant, "cuda")
        self.assertIn("svdq-fp4", hint)

    def test_ada_rejects_nvfp4(self):
        quant = {"weight": {"dtype": "fp4_e2m1_all", "group_size": 16}, "precision": "nvfp4"}
        with patch("torch.cuda.get_device_capability", _capability((8, 6))):
            hint = hardware_hint(quant, "cuda")
        self.assertIn("svdq-int4", hint)

    def test_blackwell_accepts_nvfp4(self):
        quant = {"weight": {"dtype": "fp4_e2m1_all", "group_size": 16}, "precision": "nvfp4"}
        with patch("torch.cuda.get_device_capability", _capability((12, 0))):
            self.assertIsNone(hardware_hint(quant, "cuda"))

    def test_ada_accepts_int4(self):
        quant = {"weight": {"dtype": "int4", "group_size": 64}, "precision": "int4"}
        with patch("torch.cuda.get_device_capability", _capability((8, 9))):
            self.assertIsNone(hardware_hint(quant, "cuda"))

    def test_assert_hardware_wraps_wheel_error(self):
        try:
            import nunchaku.utils  # noqa: F401
        except Exception:
            self.skipTest("nunchaku is not installed")
        quant = {"weight": {"dtype": "int4", "group_size": 64}, "precision": "int4"}
        with patch("torch.cuda.get_device_capability", _capability((12, 0))):
            with self.assertRaises(ValueError) as raised:
                assert_hardware(quant, "cuda")
        self.assertIn("svdq-fp4", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
