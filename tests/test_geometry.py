import math
import unittest

import torch

from fiery.utils.geometry import warp_features


class FeatureWarpTests(unittest.TestCase):
    def test_rotation_preserves_metric_distance_on_rectangular_bev(self):
        features = torch.zeros(1, 1, 9, 17)
        features[0, 0, 6, 8] = 1
        flow = torch.tensor([[0., 0., 0., 0., 0., math.pi / 2]])
        result = warp_features(features, flow, spatial_extent=(4.5, 8.5))
        expected = torch.zeros_like(features)
        expected[0, 0, 4, 10] = 1
        self.assertTrue(torch.allclose(result, expected))

    def test_square_bev_rotation(self):
        features = torch.zeros(1, 1, 9, 9)
        features[0, 0, 6, 4] = 1
        flow = torch.tensor([[0., 0., 0., 0., 0., math.pi / 2]])
        result = warp_features(features, flow, spatial_extent=(4.5, 4.5))
        expected = torch.zeros_like(features)
        expected[0, 0, 4, 6] = 1
        self.assertTrue(torch.allclose(result, expected))


if __name__ == '__main__':
    unittest.main()
