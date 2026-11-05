"""Use a real FLUX.3 checkpoint header to check that keys and shapes fill a ComfyUI model."""

import os
import unittest

import torch

from vitoom_nunchaku.archs.base import QuantConfig
from vitoom_nunchaku.archs.flux3 import Flux3Spec, swap_scale_shift
from vitoom_nunchaku.runtime.checkpoint import read_checkpoint

_CHECKPOINT = os.environ.get(
    "VITOOM_FLUX3_CHECKPOINT",
    os.path.expanduser(
        "~/models/flux-3-image-klein-nunchaku/svdq-fp4_r32-flux-3-image-klein-nunchaku.safetensors"
    ),
)

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
        dtype = _DTYPE[meta["dtype"]]
        state[key] = torch.empty(meta["shape"], dtype=dtype, device="meta")
    return state


class Flux3KeyMapTest(unittest.TestCase):
    def test_swap_scale_shift_is_involutive(self):
        weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        swapped = swap_scale_shift(weight)
        self.assertTrue(torch.equal(swap_scale_shift(swapped), weight))
        self.assertTrue(torch.equal(swapped[:2], weight[2:]))

    def test_checkpoint_covers_patched_model(self):
        if not os.path.isfile(_CHECKPOINT):
            self.skipTest(f"No checkpoint {_CHECKPOINT}")
        try:
            import comfy.supported_models  # noqa: F401
        except Exception as exc:
            self.skipTest(f"Cannot import ComfyUI: {exc}")

        info = read_checkpoint(_CHECKPOINT)
        spec = Flux3Spec()
        qcfg = QuantConfig(info.quantization_config, info.precision, info.rank, info.quantization_config.get("swiglu_order"))
        spec.validate(info.config, qcfg)
        converted = spec.convert_state_dict(_meta_from_header(info.tensors))
        self.assertEqual(len(converted), len(info.tensors) - 10)  # 5 image_cond blocks each pack q/k/v into one tensor

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
