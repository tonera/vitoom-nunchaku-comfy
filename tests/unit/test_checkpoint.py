import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from vitoom_nunchaku.runtime.checkpoint import (
    filename_from_choice,
    precision_from_config,
    read_checkpoint,
    read_header,
)


class CheckpointHeaderTest(unittest.TestCase):
    def test_precision(self):
        self.assertEqual(precision_from_config({"weight": {"dtype": "fp4_e2m1_all", "group_size": 16}}), "nvfp4")
        self.assertEqual(precision_from_config({"weight": {"dtype": "int4", "group_size": 64}}), "int4")
        self.assertEqual(precision_from_config({"weight": {"dtype": "int8"}}), "int8")

    def test_roundtrip_header(self):
        config = {
            "_class_name": "Flux3Transformer2DModel",
            "num_attention_heads": 24,
            "attention_head_dim": 128,
            "num_layers": 5,
            "num_single_layers": 28,
            "flow_map": True,
        }
        quant = {
            "precision": "nvfp4",
            "rank": 32,
            "weight": {"dtype": "fp4_e2m1_all", "group_size": 16},
            "swiglu_order": "up_gate",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "toy.safetensors")
            save_file(
                {"weight": torch.zeros(2, 2)},
                path,
                metadata={
                    "model_class": "NunchakuFlux3Transformer2DModel",
                    "config": json.dumps(config),
                    "quantization_config": json.dumps(quant),
                    "comfy_config": "{}",
                },
            )
            info = read_checkpoint(path)
            self.assertEqual(info.model_class, "NunchakuFlux3Transformer2DModel")
            self.assertEqual(info.config["num_layers"], 5)
            self.assertTrue(info.config["flow_map"])
            self.assertEqual(info.precision, "nvfp4")
            self.assertEqual(info.rank, 32)
            self.assertIn("weight", info.tensors)
            again = read_checkpoint(path)
            self.assertIs(again, info)

    def test_truncated_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            payload = b"{}"
            path.write_bytes(struct.pack("<Q", 100) + payload)
            with self.assertRaises(ValueError):
                read_header(str(path))

    def test_choice_label_roundtrip(self):
        label = "svdq-fp4_r32-flux-3-image-klein-nunchaku.safetensors [nvfp4 r32]"
        self.assertEqual(
            filename_from_choice(label),
            "svdq-fp4_r32-flux-3-image-klein-nunchaku.safetensors",
        )
        self.assertEqual(filename_from_choice("plain.safetensors"), "plain.safetensors")


if __name__ == "__main__":
    unittest.main()
