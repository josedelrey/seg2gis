import unittest

import torch
import torch.nn.functional as F

from src.losses import (
    boundary_weighted_bce_with_logits,
    build_loss_fn,
    describe_loss,
    make_boundary_mask,
)


class LossTests(unittest.TestCase):
    def test_boundary_mask_marks_edges_not_region_interior(self):
        mask = torch.zeros((1, 1, 7, 7))
        mask[:, :, 1:6, 1:6] = 1

        boundary = make_boundary_mask(mask, boundary_width=1)

        self.assertEqual(boundary[0, 0, 3, 3].item(), 0.0)
        self.assertEqual(boundary[0, 0, 1, 3].item(), 1.0)
        self.assertEqual(boundary[0, 0, 0, 3].item(), 1.0)

    def test_zero_boundary_weight_matches_binary_cross_entropy(self):
        logits = torch.tensor([[[[-1.0, 0.5], [2.0, -0.25]]]])
        masks = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])

        weighted = boundary_weighted_bce_with_logits(
            logits,
            masks,
            boundary_weight=0,
            boundary_width=2,
        )
        expected = F.binary_cross_entropy_with_logits(logits, masks)

        torch.testing.assert_close(weighted, expected)

    def test_loss_configuration_is_validated(self):
        with self.assertRaisesRegex(ValueError, "Unsupported loss name"):
            describe_loss({"name": "unknown"})

        with self.assertRaisesRegex(ValueError, "must be > 0"):
            describe_loss({"dice_weight": 0, "bce_weight": 0})

    def test_combined_loss_is_finite_and_supports_backpropagation(self):
        logits = torch.zeros((1, 1, 8, 8), requires_grad=True)
        masks = torch.zeros_like(logits)
        masks[:, :, 2:6, 2:6] = 1
        loss_fn = build_loss_fn(
            {
                "name": "dice_boundary_bce",
                "dice_weight": 1,
                "bce_weight": 1,
                "boundary_weight": 2,
                "boundary_width": 1,
            }
        )

        loss = loss_fn(logits, masks)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
