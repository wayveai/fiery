from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

from fiery.utils.geometry import mat2pose_vec, pose_vec2mat, warp_features


# set ignore index to 0 for vis
def convert_instance_mask_to_center_and_offset_label(instance_img, future_egomotion, num_instances, ignore_index=255,
                                                     subtract_egomotion=True, sigma=3, spatial_extent=None):
    seq_len, h, w = instance_img.shape
    center_label = torch.zeros(seq_len, 1, h, w)
    offset_label = ignore_index * torch.ones(seq_len, 2, h, w)
    future_displacement_label = ignore_index * torch.ones(seq_len, 2, h, w)
    # x is vertical displacement, y is horizontal displacement
    x, y = torch.meshgrid(torch.arange(h, dtype=torch.float), torch.arange(w, dtype=torch.float))

    if subtract_egomotion:
        future_egomotion_inv = mat2pose_vec(pose_vec2mat(future_egomotion).inverse())

    # Compute warped instance segmentation
    warped_instance_seg = {}
    for t in range(1, seq_len):
        warped_inst_t = warp_features(instance_img[t].unsqueeze(0).unsqueeze(1).float(),
                                      future_egomotion_inv[t - 1].unsqueeze(0), mode='nearest',
                                      spatial_extent=spatial_extent)
        warped_instance_seg[t] = warped_inst_t[0, 0]

    # Ignore id 0 which is the background
    for instance_id in range(1, num_instances+1):
        prev_xc = None
        prev_yc = None
        prev_mask = None
        for t in range(seq_len):
            instance_mask = (instance_img[t] == instance_id)
            if instance_mask.sum() == 0:
                # this instance is not in this frame
                prev_xc = None
                prev_yc = None
                prev_mask = None
                continue

            xc = x[instance_mask].mean().round().long()
            yc = y[instance_mask].mean().round().long()

            off_x = xc - x
            off_y = yc - y
            g = torch.exp(-(off_x ** 2 + off_y ** 2) / sigma ** 2)
            center_label[t, 0] = torch.maximum(center_label[t, 0], g)
            offset_label[t, 0, instance_mask] = off_x[instance_mask]
            offset_label[t, 1, instance_mask] = off_y[instance_mask]

            if prev_xc is not None:
                # old method
                # cur_pt = torch.stack((xc, yc)).unsqueeze(0).float()
                # if subtract_egomotion:
                #     cur_pt = warp_points(cur_pt, future_egomotion_inv[t - 1])
                # cur_pt = cur_pt.squeeze(0)

                warped_instance_mask = warped_instance_seg[t] == instance_id
                if warped_instance_mask.sum() > 0:
                    warped_xc = x[warped_instance_mask].mean().round()
                    warped_yc = y[warped_instance_mask].mean().round()

                    delta_x = warped_xc - prev_xc
                    delta_y = warped_yc - prev_yc
                    future_displacement_label[t - 1, 0, prev_mask] = delta_x
                    future_displacement_label[t - 1, 1, prev_mask] = delta_y

            prev_xc = xc
            prev_yc = yc
            prev_mask = instance_mask

    return center_label, offset_label, future_displacement_label


def find_instance_centers(center_prediction: torch.Tensor, conf_threshold: float = 0.1, nms_kernel_size: float = 3):
    assert len(center_prediction.shape) == 3
    center_prediction = F.threshold(center_prediction, threshold=conf_threshold, value=-1)

    nms_padding = (nms_kernel_size - 1) // 2
    maxpooled_center_prediction = F.max_pool2d(
        center_prediction, kernel_size=nms_kernel_size, stride=1, padding=nms_padding
    )

    # Filter all elements that are not the maximum (i.e. the center of the heatmap instance)
    center_prediction[center_prediction != maxpooled_center_prediction] = -1
    return torch.nonzero(center_prediction > 0)[:, 1:]


def group_pixels(centers: torch.Tensor, offset_predictions: torch.Tensor) -> torch.Tensor:
    width, height = offset_predictions.shape[-2:]
    x_grid = (
        torch.arange(width, dtype=offset_predictions.dtype, device=offset_predictions.device)
        .view(1, width, 1)
        .repeat(1, 1, height)
    )
    y_grid = (
        torch.arange(height, dtype=offset_predictions.dtype, device=offset_predictions.device)
        .view(1, 1, height)
        .repeat(1, width, 1)
    )
    pixel_grid = torch.cat((x_grid, y_grid), dim=0)
    center_locations = (pixel_grid + offset_predictions).view(2, width * height, 1).permute(2, 1, 0)
    centers = centers.view(-1, 1, 2)

    distances = torch.norm(centers - center_locations, dim=-1)

    instance_id = torch.argmin(distances, dim=0).reshape(1, width, height) + 1
    return instance_id


def get_instance_segmentation_and_centers(
    center_predictions: torch.Tensor,
    offset_predictions: torch.Tensor,
    foreground_mask: torch.Tensor,
    conf_threshold: float = 0.1,
    nms_kernel_size: float = 3,
    max_n_instance_centers: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    width, height = center_predictions.shape[-2:]
    center_predictions = center_predictions.view(1, width, height)
    offset_predictions = offset_predictions.view(2, width, height)
    foreground_mask = foreground_mask.view(1, width, height)

    centers = find_instance_centers(center_predictions, conf_threshold=conf_threshold, nms_kernel_size=nms_kernel_size)
    if not len(centers):
        return torch.zeros(center_predictions.shape, dtype=torch.int64, device=center_predictions.device), \
               torch.zeros((0, 2), device=centers.device)

    if len(centers) > max_n_instance_centers:
        print(f'There are a lot of detected instance centers: {centers.shape}')
        centers = centers[:max_n_instance_centers].clone()

    instance_ids = group_pixels(centers, offset_predictions)
    instance_seg = (instance_ids * foreground_mask.float()).long()

    # Make the indices of instance_seg consecutive
    instance_seg = make_instance_seg_consecutive(instance_seg)

    return instance_seg.long(), centers


def update_instance_ids(instance_seg, old_ids, new_ids):
    """
    Parameters
    ----------
        instance_seg: torch.Tensor arbitrary shape
        old_ids: 1D tensor containing the list of old ids, must be all present in instance_seg.
        new_ids: 1D tensor with the new ids, aligned with old_ids

    Returns
        new_instance_seg: torch.Tensor same shape as instance_seg with new ids
    """
    indices = torch.arange(old_ids.max() + 1, device=instance_seg.device)
    for old_id, new_id in zip(old_ids, new_ids):
        indices[old_id] = new_id

    return indices[instance_seg].long()


def make_instance_seg_consecutive(instance_seg):
    # Make the indices of instance_seg consecutive
    unique_ids = torch.unique(instance_seg)
    new_ids = torch.arange(len(unique_ids), device=instance_seg.device)
    instance_seg = update_instance_ids(instance_seg, unique_ids, new_ids)
    return instance_seg


def make_instance_id_temporally_consistent(pred_inst, future_flow, matching_threshold=3.0):
    """
    Parameters
    ----------
        pred_inst: torch.Tensor (1, seq_len, h, w)
        future_flow: torch.Tensor(1, seq_len, 2, h, w)
        matching_threshold: distance threshold for a match to be valid.

    Returns
    -------
    consistent_instance_seg: torch.Tensor(1, seq_len, h, w)

    1. time t. Loop over all detected instances. Use flow to compute new centers at time t+1.
    2. Store those centers
    3. time t+1. Re-identify instances by comparing position of actual centers, and flow-warped centers.
        Make the labels at t+1 consistent with the matching
    4. Repeat
    """
    assert pred_inst.shape[0] == 1, 'Assumes batch size = 1'

    # Initialise instance segmentations with prediction corresponding to the present
    consistent_instance_seg = [pred_inst[0, 0]]
    largest_instance_id = consistent_instance_seg[0].max().item()

    _, seq_len, h, w = pred_inst.shape
    device = pred_inst.device
    for t in range(seq_len - 1):
        # Compute predicted future instance means
        grid = torch.stack(torch.meshgrid(
            torch.arange(h, dtype=torch.float, device=device), torch.arange(w, dtype=torch.float, device=device)
        ))

        # Add future flow
        grid = grid + future_flow[0, t]
        warped_centers = []
        # Go through all ids, except the background
        t_instance_ids = torch.unique(consistent_instance_seg[-1])[1:].cpu().numpy()

        if len(t_instance_ids) == 0:
            # No instance so nothing to update
            consistent_instance_seg.append(pred_inst[0, t + 1])
            continue

        for instance_id in t_instance_ids:
            instance_mask = (consistent_instance_seg[-1] == instance_id)
            warped_centers.append(grid[:, instance_mask].mean(dim=1))
        warped_centers = torch.stack(warped_centers)

        # Compute actual future instance means
        centers = []
        grid = torch.stack(torch.meshgrid(
            torch.arange(h, dtype=torch.float, device=device), torch.arange(w, dtype=torch.float, device=device)
        ))
        n_instances = int(pred_inst[0, t + 1].max().item())

        if n_instances == 0:
            # No instance, so nothing to update.
            consistent_instance_seg.append(pred_inst[0, t + 1])
            continue

        for instance_id in range(1, n_instances + 1):
            instance_mask = (pred_inst[0, t + 1] == instance_id)
            centers.append(grid[:, instance_mask].mean(dim=1))
        centers = torch.stack(centers)

        # Compute distance matrix between warped centers and actual centers
        distances = torch.norm(centers.unsqueeze(0) - warped_centers.unsqueeze(1), dim=-1).cpu().numpy()
        # outputs (row, col) with row: index in frame t, col: index in frame t+1
        # the missing ids in col must be added (correspond to new instances)
        ids_t, ids_t_one = linear_sum_assignment(distances)
        matching_distances = distances[ids_t, ids_t_one]
        # Offset by one as id=0 is the background
        ids_t += 1
        ids_t_one += 1

        # swap ids_t with real ids. as those ids correspond to the position in the distance matrix.
        id_mapping = dict(zip(np.arange(1, len(t_instance_ids) + 1), t_instance_ids))
        ids_t = np.vectorize(id_mapping.__getitem__, otypes=[np.int64])(ids_t)

        # Filter low quality match
        ids_t = ids_t[matching_distances < matching_threshold]
        ids_t_one = ids_t_one[matching_distances < matching_threshold]

        # Elements that are in t+1, but weren't matched
        remaining_ids = set(torch.unique(pred_inst[0, t + 1]).cpu().numpy()).difference(set(ids_t_one))
        # remove background
        remaining_ids.remove(0)
        #  Set remaining_ids to a new unique id
        for remaining_id in list(remaining_ids):
            largest_instance_id += 1
            ids_t = np.append(ids_t, largest_instance_id)
            ids_t_one = np.append(ids_t_one, remaining_id)

        consistent_instance_seg.append(update_instance_ids(pred_inst[0, t + 1], old_ids=ids_t_one, new_ids=ids_t))

    consistent_instance_seg = torch.stack(consistent_instance_seg).unsqueeze(0)
    return consistent_instance_seg


def predict_instance_segmentation_and_trajectories(
        output, compute_matched_centers=False, make_consistent=True, vehicles_id=1,
):
    preds = output['segmentation'].detach()
    preds = torch.argmax(preds, dim=2, keepdims=True)
    foreground_masks = preds.squeeze(2) == vehicles_id

    batch_size, seq_len = preds.shape[:2]
    pred_inst = []
    for b in range(batch_size):
        pred_inst_batch = []
        for t in range(seq_len):
            pred_instance_t, _ = get_instance_segmentation_and_centers(
                output['instance_center'][b, t].detach(),
                output['instance_offset'][b, t].detach(),
                foreground_masks[b, t].detach()
            )
            pred_inst_batch.append(pred_instance_t)
        pred_inst.append(torch.stack(pred_inst_batch, dim=0))

    pred_inst = torch.stack(pred_inst).squeeze(2)

    if make_consistent:
        if output['instance_flow'] is None:
            print('Using zero flow because instance_future_output is None')
            output['instance_flow'] = torch.zeros_like(output['instance_offset'])
        consistent_instance_seg = []
        for b in range(batch_size):
            consistent_instance_seg.append(
                make_instance_id_temporally_consistent(pred_inst[b:b+1],
                                                       output['instance_flow'][b:b+1].detach())
            )
        consistent_instance_seg = torch.cat(consistent_instance_seg, dim=0)
    else:
        consistent_instance_seg = pred_inst

    if compute_matched_centers:
        assert batch_size == 1
        # Generate trajectories
        matched_centers = {}
        _, seq_len, h, w = consistent_instance_seg.shape
        grid = torch.stack(torch.meshgrid(
            torch.arange(h, dtype=torch.float, device=preds.device),
            torch.arange(w, dtype=torch.float, device=preds.device)
        ))

        for instance_id in torch.unique(consistent_instance_seg[0, 0])[1:].cpu().numpy():
            for t in range(seq_len):
                instance_mask = consistent_instance_seg[0, t] == instance_id
                if instance_mask.sum() > 0:
                    matched_centers[instance_id] = matched_centers.get(instance_id, []) + [
                        grid[:, instance_mask].mean(dim=-1)]

        for key, value in matched_centers.items():
            matched_centers[key] = torch.stack(value).cpu().numpy()[:, ::-1]

        return consistent_instance_seg, matched_centers

    return consistent_instance_seg


