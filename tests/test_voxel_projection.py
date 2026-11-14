import unittest

import torch

from fiery.models.fiery import Fiery


class VoxelProjectionTests(unittest.TestCase):
    def setUp(self):
        # Projection only needs the BEV grid; no encoder weights or dataset.
        self.model = Fiery.__new__(Fiery)
        torch.nn.Module.__init__(self.model)
        self.model.bev_resolution = torch.ones(3)
        self.model.bev_start_position = torch.full((3,), .5)
        self.model.bev_dimension = torch.tensor([2, 2, 1])

    def test_fractionally_outside_lower_bounds_are_discarded(self):
        features = torch.tensor([1., 2., 4., 8.]).reshape(1, 1, 1, 1, 4, 1)
        geometry = torch.tensor([
            [-.25, .25, .25], [.25, -.25, .25], [.25, .25, -.25], [.25, .25, .25],
        ]).reshape(1, 1, 1, 1, 4, 3)
        result = self.model.projection_to_birds_eye_view(features, geometry)
        expected = torch.zeros(1, 1, 2, 2)
        expected[0, 0, 0, 0] = 8.
        self.assertTrue(torch.allclose(result, expected))

    def test_lower_boundary_included_and_upper_boundary_excluded(self):
        features = torch.tensor([3., 5.]).reshape(1, 1, 1, 1, 2, 1)
        geometry = torch.tensor([[0., 0., 0.], [2., 0., 0.]]).reshape(1, 1, 1, 1, 2, 3)
        result = self.model.projection_to_birds_eye_view(features, geometry)
        self.assertEqual(result.sum().item(), 3.)


if __name__ == '__main__':
    unittest.main()
