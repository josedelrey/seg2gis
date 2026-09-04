import sys
import unittest
from pathlib import Path

import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import (  # noqa: E402
    count_values_above_thresholds,
    format_duration,
    require_bool_config,
)


class TrainHelperTests(unittest.TestCase):
    def test_threshold_counts_include_values_equal_to_threshold(self):
        values = torch.tensor([0.2, 0.3, 0.5, 0.9])
        thresholds = torch.tensor([0.3, 0.5, 0.8])

        counts = count_values_above_thresholds(values, thresholds)

        torch.testing.assert_close(
            counts,
            torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64),
        )

    def test_threshold_counts_handle_empty_input(self):
        thresholds = torch.tensor([0.3, 0.5])

        counts = count_values_above_thresholds(torch.tensor([]), thresholds)

        torch.testing.assert_close(
            counts,
            torch.zeros(2, dtype=torch.float64),
        )

    def test_require_bool_config_rejects_truthy_non_boolean(self):
        self.assertTrue(
            require_bool_config(
                {"training": {"enabled": True}},
                "training",
                "enabled",
            )
        )

        with self.assertRaisesRegex(TypeError, "JSON boolean"):
            require_bool_config(
                {"training": {"enabled": 1}},
                "training",
                "enabled",
            )

    def test_format_duration(self):
        self.assertEqual(format_duration(9.6), "10s")
        self.assertEqual(format_duration(65), "1m 05s")
        self.assertEqual(format_duration(3661), "1h 01m 01s")


if __name__ == "__main__":
    unittest.main()
