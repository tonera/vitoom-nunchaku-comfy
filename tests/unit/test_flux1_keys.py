"""Check FLUX.1 dev, depth, canny, fill, Kontext, and Krea headers against the patched ComfyUI Flux network."""

import os
import unittest

import torch

from vitoom_nunchaku.archs.base import parse_quant_config
from vitoom_nunchaku.archs.flux1 import Flux1Spec
from vitoom_nunchaku.runtime.checkpoint import read_checkpoint

_ROOTS = [
    os.path.expanduser("~/weights/nunchaku-flux.1-dev"),
    os.path.expanduser("~/weights/nunchaku-flux.1-canny-dev"),
    os.path.expanduser("~/weights/nunchaku-flux.1-depth-dev"),
    os.path.expanduser("~/weights/nunchaku-flux.1-fill-dev"),
    os.path.expanduser("~/weights/nunchaku-flux.1-kontext-dev"),
    os.path.expanduser("~/weights/nunchaku-flux.1-krea-dev"),
]

_DTYPE = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I8": torch.int8,
    "I32": torch.int32,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def _checkpoints() -> list[str]:
    found = []
    for root in _ROOTS:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if name.endswith(".safetensors") and name.startswith("svdq-"):
                found.append(os.path.join(root, name))
    return found


def _meta_from_header(tensors: dict) -> dict[str, torch.Tensor]:
    state = {}
    for key, meta in tensors.items():
        state[key] = torch.empty(meta["shape"], dtype=_DTYPE[meta["dtype"]], device="meta")
    return state


class Flux1KeyMapTest(unittest.TestCase):
    def test_checkpoint_covers_patched_model(self):
        present = _checkpoints()
        if not present:
            self.skipTest("No FLUX.1 checkpoints")
        try:
            import comfy.supported_models  # noqa: F401
        except Exception as exc:
            self.skipTest(f"Cannot import ComfyUI: {exc}")
        for path in present:
            with self.subTest(path=os.path.basename(path)):
                self._check(path)

    def _check(self, path: str):
        from nunchaku.models.transformers.utils import patch_scale_key

        info = read_checkpoint(path)
        spec = Flux1Spec()
        qcfg = parse_quant_config(info.quantization_config, spec, info.tensors)
        spec.validate(info.config, qcfg)
        converted = spec.convert_state_dict(_meta_from_header(info.tensors))
        self.assertEqual(len(converted), len(info.tensors))

        model_config = spec.comfy_model_config(info.config)
        model_config.set_inference_dtype(torch.bfloat16, None)
        model = model_config.get_model({}, "", device=torch.device("meta"))
        spec.patch_model(model.diffusion_model, qcfg, torch.bfloat16)
        patch_scale_key(model.diffusion_model, converted)
        reference = model.diffusion_model.state_dict()
        missing = sorted(set(reference) - set(converted))
        extra = sorted(set(converted) - set(reference))
        self.assertEqual(extra[:12], [], extra[:12])
        self.assertEqual(missing[:12], [], missing[:12])
        self.assertEqual(set(converted), set(reference))
        for key, tensor in converted.items():
            self.assertEqual(tuple(tensor.shape), tuple(reference[key].shape), key)
            self.assertEqual(tensor.dtype, reference[key].dtype, key)


if __name__ == "__main__":
    unittest.main()
