"""Use a real Anima checkpoint header to check that DiT keys and the LLM adapter fill a ComfyUI model."""

import os
import unittest

import torch

from vitoom_nunchaku.archs.anima import AnimaSpec, _adapter_pairs, find_llm_adapter_path
from vitoom_nunchaku.archs.base import parse_quant_config
from vitoom_nunchaku.archs.flux2 import Flux2Spec
from vitoom_nunchaku.runtime.checkpoint import read_checkpoint, read_header

_CHECKPOINTS = [
    os.path.expanduser(
        "~/models/Anima-Turbo-v1.1-Nunchaku/svdq-int8_r0-Anima-Turbo-v1.1-Nunchaku.safetensors"
    ),
    os.path.expanduser("~/models/Anima-2.9B-Nunchaku/svdq-int8_r0-Anima-2.9B-Nunchaku.safetensors"),
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


class AnimaKeyMapTest(unittest.TestCase):
    def test_rank_zero_is_anima_only(self):
        info = read_checkpoint(_CHECKPOINTS[0])
        if not os.path.isfile(_CHECKPOINTS[0]):
            self.skipTest("No Anima Turbo checkpoint")
        qcfg = parse_quant_config(info.quantization_config, AnimaSpec())
        self.assertEqual(qcfg.rank, 0)
        self.assertEqual(qcfg.precision, "int8")
        with self.assertRaises(ValueError):
            parse_quant_config({"precision": "int4", "rank": 0, "weight": {"dtype": "int4"}}, Flux2Spec())

    def test_checkpoint_covers_patched_model(self):
        present = [path for path in _CHECKPOINTS if os.path.isfile(path)]
        if not present:
            self.skipTest("No Anima checkpoint")
        try:
            import comfy.supported_models  # noqa: F401
        except Exception as exc:
            self.skipTest(f"Cannot import ComfyUI: {exc}")
        for path in present:
            with self.subTest(path=os.path.basename(path)):
                self._check(path)

    def _check(self, path: str):
        info = read_checkpoint(path)
        spec = AnimaSpec()
        qcfg = parse_quant_config(info.quantization_config, spec)
        spec.validate(info.config, qcfg)
        converted = spec.convert_state_dict(_meta_from_header(info.tensors))
        self.assertEqual(len(converted), len(info.tensors))

        adapter_path = find_llm_adapter_path(path)
        _metadata, adapter_header = read_header(adapter_path)
        adapter = {}
        for source_key, target_key in _adapter_pairs(list(adapter_header)):
            adapter[target_key] = torch.empty(
                adapter_header[source_key]["shape"],
                dtype=_DTYPE[adapter_header[source_key]["dtype"]],
                device="meta",
            )
        converted.update(adapter)

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
