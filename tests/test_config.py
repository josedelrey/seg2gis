import json
import tempfile
import unittest
from pathlib import Path

from src.config import (
    get_config_value,
    load_config,
    resolve_model_metadata_path,
    resolve_model_path,
)


class ConfigTests(unittest.TestCase):
    def test_load_config_reads_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            expected = {"model": {"architecture": "unet"}}
            config_path.write_text(json.dumps(expected), encoding="utf-8")

            self.assertEqual(load_config(config_path), expected)

    def test_get_config_value_reads_nested_value(self):
        config = {"training": {"batch_size": 8}}

        self.assertEqual(get_config_value(config, "training", "batch_size"), 8)

    def test_get_config_value_returns_default_for_missing_path(self):
        config = {"training": None}

        self.assertEqual(
            get_config_value(config, "training", "batch_size", default=4),
            4,
        )

    def test_model_paths_use_run_name(self):
        model_dir = Path("models") / "phase2"

        self.assertEqual(
            Path(resolve_model_path(model_dir, "baseline")),
            model_dir / "baseline.pth",
        )
        self.assertEqual(
            Path(resolve_model_metadata_path(model_dir, "baseline")),
            model_dir / "baseline.json",
        )


if __name__ == "__main__":
    unittest.main()
