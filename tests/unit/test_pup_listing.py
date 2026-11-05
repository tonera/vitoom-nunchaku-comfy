import os
import unittest

from vitoom_nunchaku.runtime.pup import iter_pup_files, resolve_pup


class PupListingTest(unittest.TestCase):
    def test_quantized_pup_is_listed(self):
        found = iter_pup_files()
        labels = [label for label, _path in found]
        self.assertIn("svdq-int4-flux-3-image-klein-pup-nunchaku.safetensors [w4 pup]", labels)
        path = resolve_pup(labels[0])
        directory = os.path.dirname(path)
        self.assertTrue(os.path.isfile(os.path.join(directory, "system_prompts.json")))
        self.assertTrue(os.path.isfile(os.path.join(directory, "tokenizer.json")))
        self.assertNotIn("model.safetensors", os.path.basename(path))


if __name__ == "__main__":
    unittest.main()
