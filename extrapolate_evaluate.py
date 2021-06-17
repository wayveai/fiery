import socket

import torch
import numpy as np
from scipy import interpolate
from tqdm import tqdm

from fiery.data import prepare_dataloaders
from fiery.trainer import TrainingModule
from fiery.metrics import PanopticMetric, NSamplesPanopticMetric
from fiery.utils.network import preprocess_batch
from fiery.utils.instance import predict_instance_segmentation_and_trajectories,\
    get_instance_segmentation_and_centers, make_instance_id_temporally_consistent, \
    make_instance_seg_consecutive
from fiery.utils.geometry import cumulative_warp_features

LOWER_BOUND = 5.0
MATCHING_THRESHOLD = 10.0

EVALUATE_N_SAMPLES = True
INCLUDE_TRAJECTORY_METRICS = True
VEHICLES_ID = 1

trainer_path = '/home/anthony/experiments/fiery_experiments/public_repo_weights/fiery_static.ckpt'

# 30mx30m, 100mx100m
EVALUATION_RANGES = {'30x30': (70, 130),
                     '100x100': (0, 200)
                     }

device = torch.device('cuda:0')
trainer = TrainingModule.load_from_checkpoint(trainer_path)
trainer.eval()
trainer.to(device)
model = trainer.model

cfg = model.cfg
cfg.GPUS = "[0]"
cfg.BATCHSIZE = 1

if socket.gethostname() == 'auris':
    cfg.DATASET.DATAROOT = '/home/anthony/datasets/nuscenes'
    cfg.DATASET.VERSION = 'trainval'

cfg.TIME_RECEPTIVE_FIELD = 3
cfg.N_FUTURE_FRAMES = 4
model.receptive_field = cfg.TIME_RECEPTIVE_FIELD
model.temporal_model.receptive_field = cfg.TIME_RECEPTIVE_FIELD
model.n_future = cfg.N_FUTURE_FRAMES

t_present = model.receptive_field - 1
n_future = model.n_future

bev_size = model.bev_size

_, valloader, _, val_dataset = prepare_dataloaders(cfg)

panoptic_metrics = {}
for key in EVALUATION_RANGES.keys():
    panoptic_metrics[key] = PanopticMetric(n_classes=2, temporally_consistent=True,
                                           include_trajectory_metrics=INCLUDE_TRAJECTORY_METRICS,
                                           pixel_resolution=cfg.LIFT.X_BOUND[-1],
                                           ).to(
        device)

if EVALUATE_N_SAMPLES:
    diversity_distance_metrics = {}
    for key in EVALUATION_RANGES.keys():
        diversity_distance_metrics[key] = NSamplesPanopticMetric(
            num_classes=2, temporally_consistent=True,
            include_trajectory_metrics=INCLUDE_TRAJECTORY_METRICS,
            pixel_resolution=cfg.LIFT.X_BOUND[-1],
        ).to(device)
    torch.manual_seed(0)

for i in tqdm(range(0, len(val_dataset), 100)):
    batch = val_dataset[i]
    preprocess_batch(batch, device, unsqueeze=True)

    imgs = batch['image']
    extrinsics = batch['extrinsics']
    intrinsics = batch['intrinsics']
    future_egomotions = batch['future_egomotion']

    batch_size = imgs.shape[0]

    labels, future_distribution_inputs = trainer.prepare_future_labels(batch)

    # Lifting features
    with torch.no_grad():
        x = model.calculate_birds_eye_view_features(imgs, intrinsics, extrinsics)
        # Decode, frame by frame
        output = model.decoder(x)

        # Output instance segmentation
        preds = output['segmentation'].detach()
        preds = torch.argmax(preds, dim=2, keepdims=True)
        foreground_masks = preds.squeeze(2) == VEHICLES_ID

        batch_size, seq_len, _, h, w = preds.shape
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

        pred_inst = torch.stack(pred_inst).squeeze(2)[:, :model.receptive_field].contiguous()

    # Warp to present's reference frame.
    pred_inst = cumulative_warp_features(
        pred_inst[:, :model.receptive_field].float().unsqueeze(2), future_egomotions[:, :model.receptive_field],
        mode='nearest', spatial_extent=trainer.spatial_extent
    ).long().squeeze(2)

    # Because of warping, remake instance ids consecutive:
    for t in range(pred_inst.shape[1]):
        pred_inst[:, t] = make_instance_seg_consecutive(pred_inst[:, t])

    # Make ids consistent
    zero_flow = torch.zeros((batch_size, seq_len, 2, h, w), device=preds.device)

    # Start making consistent from t_present
    consistent_instance_seg = make_instance_id_temporally_consistent(
        torch.flip(pred_inst, dims=(1,)), zero_flow, matching_threshold=MATCHING_THRESHOLD
    ).flip(dims=(1,))

    # Generate trajectories
    matched_centers = {}
    matched_timesteps = {}
    _, seq_len, h, w = consistent_instance_seg.shape
    grid = torch.stack(torch.meshgrid(
        torch.arange(h, dtype=torch.float, device=preds.device),
        torch.arange(w, dtype=torch.float, device=preds.device)
    ))
    for instance_id in torch.unique(consistent_instance_seg[0, -1])[1:].cpu().numpy():
        for t in range(seq_len):
            instance_mask = consistent_instance_seg[0, t] == instance_id
            if instance_mask.sum() > 0:
                matched_centers[instance_id] = matched_centers.get(instance_id, []) + [
                    grid[:, instance_mask].mean(dim=-1)]
                matched_timesteps[instance_id] = matched_timesteps.get(instance_id, []) + [t]

    for key, value in matched_centers.items():
        matched_centers[key] = torch.stack(value).cpu().numpy()[:, ::-1]

    # Extrapolate trajectories
    extrapolated_centers = {}
    for key, value in matched_centers.items():
        centers = matched_centers[key]

        # No past trajectory, so repeat.
        if len(centers) == 1:
            extrapolated_centers[key] = np.tile(centers, (n_future + 1, 1))
            continue

        timesteps = np.asarray(matched_timesteps[key])

        if np.linalg.norm(centers[-1] - centers[-2]) < LOWER_BOUND or len(timesteps) < 3:
            # Close centers, so it must be static. Or too short past trajectory to infer future trajectory.
            extrapolated_centers[key] = np.tile(centers[-1:], (n_future + 1, 1))
            continue

        print('Extrapolating trajectory')
        f = interpolate.interp1d(timesteps, centers, axis=0, fill_value='extrapolate', kind='linear')
        future_timesteps = np.arange(t_present, t_present + 1 + n_future)
        extrapolated_centers[key] = f(future_timesteps)

    # Compute extrapolated instance segmentation
    extrapolated_instance_seg = torch.zeros((1, n_future+1, h, w), dtype=torch.int64, device=preds.device)
    # Set first frame to the present.
    extrapolated_instance_seg[:, 0] = consistent_instance_seg[:, -1]
    for instance_id in extrapolated_centers.keys():
        for t in range(n_future):
            instance_mask = extrapolated_instance_seg[0, t] == instance_id
            if instance_mask.sum() > 0:
                translation = (extrapolated_centers[instance_id][t+1] - extrapolated_centers[instance_id][t])[::-1].copy()
                new_position = (grid[:, instance_mask] + torch.from_numpy(translation).to(preds.device).unsqueeze(1)).long()
                # Filter out of view
                out_of_view_mask = ((new_position[0] >= 0)
                                    & (new_position[0] < bev_size[0])
                                    & (new_position[1] >= 0)
                                    & (new_position[1] < bev_size[1])
                                    )
                new_position = new_position[:, out_of_view_mask]
                extrapolated_instance_seg[0, t+1, new_position[0, :], new_position[1, :]] = instance_id

    segmentation_pred = (extrapolated_instance_seg > 0).long()

    if EVALUATE_N_SAMPLES:
        extrapolated_instance_seg = extrapolated_instance_seg.unsqueeze(1)

    for key, grid in EVALUATION_RANGES.items():
        limits = slice(grid[0], grid[1])

        if not EVALUATE_N_SAMPLES:
            panoptic_metrics[key](extrapolated_instance_seg[..., limits, limits].contiguous().detach(),
                                  labels['instance'][..., limits, limits].contiguous()
                                  )
        else:
            diversity_distance_metrics[key](extrapolated_instance_seg[..., limits, limits].contiguous(),
                                            labels['instance'][..., limits, limits].contiguous())

results_json = {}

if not EVALUATE_N_SAMPLES:
    for key, grid in EVALUATION_RANGES.items():
        panoptic_scores = panoptic_metrics[key].compute()
        for panoptic_key, value in panoptic_scores.items():
            results_json[f'{panoptic_key}'] = results_json.get(f'{panoptic_key}', []) + [100 * value[1].item()]

    for panoptic_key in ['pq', 'sq', 'rq', 'ade', 'fde']:
        print(panoptic_key)
        print(' & '.join([f'{x:.1f}' for x in results_json[panoptic_key]]))

else:
    for key, grid in EVALUATION_RANGES.items():
        print(f'Range {grid}---------')
        diversity_scores = diversity_distance_metrics[key].compute()
        print(
            f'ADE: {diversity_scores["ade"][1]:.3f}, fde: {diversity_scores["fde"][1]:.3f},')
        print('\n')
