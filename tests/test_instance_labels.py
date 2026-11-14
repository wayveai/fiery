import unittest

import torch

from fiery.utils.instance import (
    convert_instance_mask_to_center_and_offset_label,
    make_instance_id_temporally_consistent,
    make_instance_seg_consecutive,
    predict_instance_segmentation_and_trajectories,
)


class InstanceLabelTests(unittest.TestCase):
    def test_flow_without_egomotion_compensation(self):
        instances = torch.zeros(2, 5, 5, dtype=torch.long)
        instances[0, 1, 1] = instances[1, 2, 3] = 1
        _, _, flow = convert_instance_mask_to_center_and_offset_label(
            instances, None, 1, subtract_egomotion=False,
        )
        self.assertTrue(torch.allclose(flow[0, :, 1, 1], torch.tensor([1., 2.])))
        self.assertTrue((flow[1] == 255).all())

    def test_full_foreground_remains_foreground(self):
        result = make_instance_seg_consecutive(torch.tensor([[8, 8], [3, 3]]))
        self.assertEqual(set(result.flatten().tolist()), {1, 2})
        self.assertEqual(result[0, 0], result[0, 1])
        self.assertNotEqual(result[0, 0], result[1, 0])

    def test_background_stays_zero(self):
        result = make_instance_seg_consecutive(torch.tensor([[0, 8], [3, 0]]))
        self.assertEqual(result.tolist(), [[0, 2], [1, 0]])

    def test_new_tracks_after_empty_frame_get_new_ids(self):
        instances = torch.tensor([[[[0, 1, 0]], [[0, 0, 0]], [[0, 1, 0]]]])
        flow = torch.zeros(1, 3, 2, 1, 3)
        result = make_instance_id_temporally_consistent(instances, flow)
        self.assertEqual(result[0, 0, 0, 1], 1)
        self.assertEqual(result[0, 2, 0, 1], 2)

    def test_matching_full_foreground_frames(self):
        instances = torch.tensor([[[[0, 1], [0, 0]], [[1, 1], [1, 1]]]])
        result = make_instance_id_temporally_consistent(instances, torch.zeros(1, 2, 2, 2, 2))
        self.assertTrue(torch.allclose(result, instances))

    def test_first_full_foreground_frame_keeps_its_track(self):
        instances = torch.tensor([[[[1, 1], [1, 1]], [[2, 2], [2, 2]]]])
        result = make_instance_id_temporally_consistent(instances, torch.zeros(1, 2, 2, 2, 2))
        self.assertTrue(torch.allclose(result, torch.ones_like(instances)))

    def test_sparse_future_ids_are_matched(self):
        instances = torch.tensor([[[[0, 1, 0, 2]], [[0, 3, 0, 8]]]])
        result = make_instance_id_temporally_consistent(instances, torch.zeros(1, 2, 2, 1, 4))
        self.assertTrue(torch.allclose(result[0, 1], instances[0, 0]))

    def test_matched_trajectory_without_background(self):
        segmentation = torch.zeros(1, 1, 2, 2, 2)
        segmentation[:, :, 1] = 1
        centers = torch.zeros(1, 1, 1, 2, 2)
        centers[0, 0, 0, 0, 0] = 1
        output = {'segmentation': segmentation, 'instance_center': centers,
                  'instance_offset': torch.zeros(1, 1, 2, 2, 2), 'instance_flow': None}
        instances, trajectories = predict_instance_segmentation_and_trajectories(
            output, compute_matched_centers=True, make_consistent=False,
        )
        self.assertTrue(torch.allclose(instances, torch.ones_like(instances)))
        self.assertIn(1, trajectories)
        self.assertEqual(trajectories[1].tolist(), [[.5, .5]])


if __name__ == '__main__':
    unittest.main()
