"""Use a real Qwen-Image 2.1 checkpoint header to check that keys and shapes fill a ComfyUI model."""

import os
import unittest

import torch

from vitoom_nunchaku.archs.base import QuantConfig
from vitoom_nunchaku.archs.qwen_image21 import QwenImage21Spec
from vitoom_nunchaku.runtime.checkpoint import read_checkpoint

_CHECKPOINT = os.environ.get(
    "VITOOM_QWEN_IMAGE21_CHECKPOINT",
    os.path.expanduser(
        "~/models/Qwen-Image-2.1-Nunchaku/svdq-fp4_r32-Qwen-Image-2.1-Nunchaku.safetensors"
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


class QwenImage21KeyMapTest(unittest.TestCase):
    def test_checkpoint_covers_patched_model(self):
        if not os.path.isfile(_CHECKPOINT):
            self.skipTest(f"No checkpoint {_CHECKPOINT}")
        try:
            import comfy.supported_models  # noqa: F401
        except Exception as exc:
            self.skipTest(f"Cannot import ComfyUI: {exc}")

        info = read_checkpoint(_CHECKPOINT)
        spec = QwenImage21Spec()
        qcfg = QuantConfig(
            info.quantization_config,
            info.precision,
            info.rank,
            info.quantization_config.get("swiglu_order"),
        )
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

    def test_split_qkv_is_rejected(self):
        spec = QwenImage21Spec()
        with self.assertRaises(KeyError):
            spec.convert_state_dict({"transformer_blocks.0.attn.to_q.weight": torch.empty(1)})


if __name__ == "__main__":
    unittest.main()
