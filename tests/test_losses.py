import math
import unittest

import torch

from fiery.losses import SegmentationLoss, SpatialRegressionLoss


class LossTests(unittest.TestCase):
    def test_empty_regression_keeps_zero_gradient(self):
        prediction = torch.randn(1, 1, 2, 2, 2, requires_grad=True)
        target = torch.full_like(prediction, 255)
        loss = SpatialRegressionLoss(norm=1)(prediction, target)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.allclose(prediction.grad, torch.zeros_like(prediction)))

    def test_top_k_keeps_at_least_one_pixel(self):
        prediction = torch.zeros(1, 1, 2, 1, 1, requires_grad=True)
        target = torch.zeros(1, 1, 1, 1, 1, dtype=torch.long)
        loss = SegmentationLoss(torch.ones(2), use_top_k=True, top_k_ratio=.25)(prediction, target)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_ignored_pixels_do_not_dilute_segmentation_loss(self):
        prediction = torch.zeros(1, 1, 2, 1, 2, requires_grad=True)
        target = torch.tensor([[[[[0, 255]]]]])
        loss = SegmentationLoss(torch.ones(2))(prediction, target)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        self.assertTrue(torch.allclose(prediction.grad[..., 1], torch.zeros_like(prediction.grad[..., 1])))

    def test_all_ignored_segmentation_is_differentiable_zero(self):
        prediction = torch.zeros(1, 2, 2, 1, 2, requires_grad=True)
        target = torch.full((1, 2, 1, 1, 2), 255, dtype=torch.long)
        loss = SegmentationLoss(torch.ones(2), use_top_k=True, top_k_ratio=.25)(prediction, target)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.allclose(prediction.grad, torch.zeros_like(prediction)))

    def test_future_discount_is_preserved(self):
        prediction = torch.zeros(1, 2, 2, 1, 1)
        target = torch.zeros(1, 2, 1, 1, 1, dtype=torch.long)
        loss = SegmentationLoss(torch.ones(2), future_discount=.5)(prediction, target)
        self.assertAlmostEqual(loss.item(), .75 * math.log(2), places=6)

    def test_top_k_selects_the_hardest_valid_pixels(self):
        prediction = torch.tensor([[[[[0., 2., 0., 0.]], [[0., 0., 0., 0.]]]]])
        target = torch.tensor([[[[[0, 0, 255, 255]]]]])
        loss = SegmentationLoss(torch.ones(2), use_top_k=True, top_k_ratio=.5)(prediction, target)
        expected = (math.log(2) + math.log1p(math.exp(-2))) / 2
        self.assertAlmostEqual(loss.item(), expected, places=6)


if __name__ == '__main__':
    unittest.main()
