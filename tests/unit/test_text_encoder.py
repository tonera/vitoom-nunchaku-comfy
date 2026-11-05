import os
import unittest

import torch

from vitoom_nunchaku.runtime.text_encoder import (
    FLUX3_PAD_TOKEN,
    QWEN_IMAGE21_IM_START,
    QWEN_IMAGE21_IMAGE_PAD,
    attention_mask_from_ids,
    flux3_position_ids,
    iter_text_encoder_files,
    stack_flux3_hidden,
    text_encoder_kind,
    text_encoder_recipe,
    token_ids_from_weights,
    trim_qwen_image21,
)


class TextEncoderRecipeTest(unittest.TestCase):
    def test_w4a16_kind(self):
        metadata = {"quantization_config": '{"text": {"format": "tinychat_w4x16y16", "weight": {"dtype": "int4"}}}'}
        self.assertEqual(text_encoder_kind(metadata), "w4a16")

    def test_pad_mask_and_positions(self):
        ids = token_ids_from_weights([[(11, 1.0), (12, 1.0), (FLUX3_PAD_TOKEN, 1.0)]])
        self.assertEqual(tuple(ids.shape), (1, 80))
        self.assertTrue(torch.equal(ids[0, 3:], torch.full((77,), FLUX3_PAD_TOKEN)))
        mask = attention_mask_from_ids(ids)
        self.assertEqual(int(mask.sum()), 2)
        positions = flux3_position_ids(mask)
        self.assertEqual(int(positions[0, 0]), 0)
        self.assertEqual(int(positions[0, 1]), 1)
        self.assertTrue(torch.all(positions[0, 2:] == 1))

    def test_stack_hidden_states(self):
        hidden = [torch.zeros(1, 3, 4) for _ in range(33)]
        for index in (4, 8, 12, 16, 20, 24, 28, 32):
            hidden[index] = torch.full((1, 3, 4), float(index))
        stacked = stack_flux3_hidden(hidden)
        self.assertEqual(tuple(stacked.shape), (1, 3, 32))
        self.assertEqual(float(stacked[0, 0, 0]), 4.0)
        self.assertEqual(float(stacked[0, 0, 4]), 8.0)

    def test_local_weights_are_listed(self):
        path = os.path.expanduser(
            "~/weights/Qwen3-VL-4B-Instruct-Nunchaku/svdq-int4-Qwen3-VL-4B-Instruct-Nunchaku.safetensors"
        )
        if not os.path.isfile(path):
            self.skipTest("Local text encoder is not in ~/weights")
        labels = dict(iter_text_encoder_files())
        label = "svdq-int4-Qwen3-VL-4B-Instruct-Nunchaku.safetensors [w4a16 flux3]"
        self.assertIn(label, labels)
        self.assertEqual(os.path.realpath(labels[label]), os.path.realpath(path))

    def test_recipe_follows_hidden_size(self):
        self.assertEqual(text_encoder_recipe({"config": '{"text_config": {"hidden_size": 2560}}'}), "flux3")
        self.assertEqual(text_encoder_recipe({"config": '{"text_config": {"hidden_size": 4096}}'}), "qwen_image21")
        flux2 = {
            "model_class": "NunchakuQwenEncoderModel",
            "config": '{"text_config": {"hidden_size": 2560}}',
        }
        self.assertEqual(text_encoder_recipe(flux2), "flux2")
        flux2["config"] = '{"text_config": {"hidden_size": 4096}}'
        self.assertEqual(text_encoder_recipe(flux2), "flux2")

    def test_local_qwen3_4b_text_is_listed(self):
        path = os.path.expanduser(
            "~/weights/Qwen3-4B-text-Nunchaku/svdq-int4-Qwen3-4B-text-Nunchaku.safetensors"
        )
        if not os.path.isfile(path):
            self.skipTest("Local FLUX.2 text encoder is not in ~/weights")
        labels = dict(iter_text_encoder_files())
        label = "svdq-int4-Qwen3-4B-text-Nunchaku.safetensors [w4a16 flux2]"
        self.assertIn(label, labels)
        self.assertEqual(os.path.realpath(labels[label]), os.path.realpath(path))

    def test_trim_drops_system_prompt_and_image_tokens(self):
        # system, user text, one image token, trailing text
        ids = torch.tensor([
            QWEN_IMAGE21_IM_START, 1, 2,
            QWEN_IMAGE21_IM_START, 3, 4,
            QWEN_IMAGE21_IMAGE_PAD, QWEN_IMAGE21_IMAGE_PAD,
            5,
        ])
        hidden = torch.arange(ids.shape[0], dtype=torch.float32).view(1, -1, 1)
        trimmed, slots = trim_qwen_image21(hidden, ids, [(6, 2)], keep_vision=False)
        self.assertEqual(slots, [3])
        self.assertEqual(tuple(trimmed.shape), (1, 4, 1))
        self.assertEqual([int(value) for value in trimmed[0, :, 0]], [3, 4, 5, 8])

    def test_local_8b_instruct_is_listed(self):
        path = os.path.expanduser(
            "~/weights/Qwen3-VL-8B-Instruct-Nunchaku/svdq-int4-Qwen3-VL-8B-Instruct-Nunchaku.safetensors"
        )
        if not os.path.isfile(path):
            self.skipTest("Local 8B text encoder is not in ~/weights")
        labels = dict(iter_text_encoder_files())
        label = "svdq-int4-Qwen3-VL-8B-Instruct-Nunchaku.safetensors [w4a16 qwen_image21]"
        self.assertIn(label, labels)
        self.assertEqual(os.path.realpath(labels[label]), os.path.realpath(path))


if __name__ == "__main__":
    unittest.main()
