import copy
import sys
import types
import unittest


sys.modules.setdefault("yaml", types.ModuleType("yaml"))

from scripts.run_experiments import build_training_config  # noqa: E402


class BuildTrainingConfigTests(unittest.TestCase):
    def test_inference_path_tracks_generated_run_and_model_directory(self):
        base_config = {
            "model": {
                "model_dir": "models",
                "architecture": "unet",
                "encoder": "efficientnet-b3",
            },
            "training": {"run_name": "old_run"},
            "inference": {"model_path": "models/old_run.pth"},
        }
        original = copy.deepcopy(base_config)

        generated = build_training_config(
            base_config,
            {
                "run_name": "selected_run",
                "model_dir": "models/phase2_augmentation",
            },
        )

        self.assertEqual(
            generated["inference"]["model_path"],
            "models/phase2_augmentation/selected_run.pth",
        )
        self.assertEqual(base_config, original)


if __name__ == "__main__":
    unittest.main()
