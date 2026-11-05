"""Use a real FLUX.2 Klein checkpoint header to check that keys and shapes fill a ComfyUI model."""

import os
import unittest

import torch

from vitoom_nunchaku.archs.base import QuantConfig
from vitoom_nunchaku.archs.flux2 import Flux2Spec
from vitoom_nunchaku.runtime.checkpoint import read_checkpoint

_CHECKPOINTS = [
    os.environ.get(
        "VITOOM_FLUX2_CHECKPOINT",
        os.path.expanduser(
            "~/models/FLUX.2-klein-4B-Nunchaku/svdq-fp4_r32-FLUX.2-klein-4B-Nunchaku.safetensors"
        ),
    ),
    os.path.expanduser(
        "~/models/FLUX.2-klein-9B-Nunchaku/svdq-fp4_r32-FLUX.2-klein-9B-Nunchaku.safetensors"
    ),
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
        dtype = _DTYPE[meta["dtype"]]
        state[key] = torch.empty(meta["shape"], dtype=dtype, device="meta")
    return state


class Flux2KeyMapTest(unittest.TestCase):
    def test_checkpoint_covers_patched_model(self):
        present = [path for path in _CHECKPOINTS if os.path.isfile(path)]
        if not present:
            self.skipTest("No FLUX.2 Klein checkpoint")
        try:
            import comfy.supported_models  # noqa: F401
        except Exception as exc:
            self.skipTest(f"Cannot import ComfyUI: {exc}")

        for path in present:
            with self.subTest(path=os.path.basename(path)):
                self._check(path)

    def _check(self, path: str):
        info = read_checkpoint(path)
        spec = Flux2Spec()
        qcfg = QuantConfig(
            info.quantization_config,
            info.precision,
            info.rank,
            info.quantization_config.get("swiglu_order"),
        )
        spec.validate(info.config, qcfg)
        converted = spec.convert_state_dict(_meta_from_header(info.tensors))
        self.assertEqual(len(converted), len(info.tensors))
        # wtscale is a Python float on SVDQ. patch_scale_key pulls it out at load time, so it is not in the state_dict.
        wtscale = [key for key in converted if key.endswith(".wtscale")]
        self.assertTrue(wtscale)
        for key in wtscale:
            del converted[key]

        model_config = spec.comfy_model_config(info.config)
        model_config.set_inference_dtype(torch.bfloat16, None)
        model = model_config.get_model({}, "", device=torch.device("meta"))
        spec.patch_model(model.diffusion_model, qcfg, torch.bfloat16)
        reference = model.diffusion_model.state_dict()
        self.assertEqual(set(converted), set(reference))
        for key, tensor in converted.items():
            self.assertEqual(tuple(tensor.shape), tuple(reference[key].shape), key)
            if key.endswith(".wcscales") and tensor.dtype == torch.float8_e4m3fn and reference[key].dtype == torch.bfloat16:
                continue
            self.assertEqual(tensor.dtype, reference[key].dtype, key)
        swapped = converted["final_layer.adaLN_modulation.1.weight"]
        original = info.tensors["norm_out.linear.weight"]
        self.assertEqual(tuple(swapped.shape), tuple(original["shape"]))

    def test_fused_single_block_is_rejected(self):
        spec = Flux2Spec()
        with self.assertRaises(KeyError):
            spec.convert_state_dict({"single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight": torch.empty(1)})


if __name__ == "__main__":
    unittest.main()
