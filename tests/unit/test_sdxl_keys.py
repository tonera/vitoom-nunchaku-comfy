"""Use a oneObsession v18 checkpoint header to check that SDXL keys fill a ComfyUI UNet."""

import os
import unittest

import torch

from vitoom_nunchaku.archs.base import parse_quant_config
from vitoom_nunchaku.archs.sdxl import SDXLSpec
from vitoom_nunchaku.runtime.checkpoint import read_checkpoint

_CHECKPOINTS = [
    os.path.expanduser("~/models/oneObsession_v18-nunchaku/svdq-fp4_r32-oneObsession_v18.safetensors"),
    os.path.expanduser("~/models/oneObsession_v18-nunchaku/svdq-int4_r32-oneObsession_v18.safetensors"),
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


class SDXLKeyMapTest(unittest.TestCase):
    def test_checkpoint_covers_patched_model(self):
        present = [path for path in _CHECKPOINTS if os.path.isfile(path)]
        if not present:
            self.skipTest("No oneObsession SDXL checkpoint")
        try:
            import comfy.supported_models  # noqa: F401
        except Exception as exc:
            self.skipTest(f"Cannot import ComfyUI: {exc}")
        for path in present:
            with self.subTest(path=os.path.basename(path)):
                self._check(path)

    def _check(self, path: str):
        info = read_checkpoint(path)
        spec = SDXLSpec()
        qcfg = parse_quant_config(info.quantization_config, spec)
        spec.validate(info.config, qcfg)
        converted = spec.convert_state_dict(_meta_from_header(info.tensors))
        self.assertEqual(len(converted), len(info.tensors))

        model_config = spec.comfy_model_config(info.config)
        model_config.set_inference_dtype(torch.bfloat16, None)
        model = model_config.get_model({}, "", device=torch.device("meta"))
        spec.patch_model(model.diffusion_model, qcfg, torch.bfloat16)
        reference = model.diffusion_model.state_dict()
        self.assertEqual(set(converted), set(reference))
        for key, tensor in converted.items():
            self.assertEqual(tuple(tensor.shape), tuple(reference[key].shape), key)
            self.assertEqual(tensor.dtype, reference[key].dtype, key)


if __name__ == "__main__":
    unittest.main()
