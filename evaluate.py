from argparse import ArgumentParser
from time import time

import torch
from tqdm import tqdm

from fiery.data import prepare_dataloaders
from fiery.trainer import TrainingModule
from fiery.metrics import IntersectionOverUnion, PanopticMetric, NSamplesPanopticMetric
from fiery.utils.network import preprocess_batch
from fiery.utils.instance import predict_instance_segmentation_and_trajectories

# 30mx30m, 100mx100m
EVALUATION_RANGES = {'30x30': (70, 130),
                     '100x100': (0, 200)
                     }

EVALUATE_ORACLE = False
INCLUDE_TRAJECTORY_METRICS = True
EVALUATE_N_SAMPLES = True
N_SAMPLES = 1
NOISE_SCALE = 20.0

# N SAMPLES

# 10 samples
# Range (70, 130)---------
# ADE: 0.338, fde: 0.300,
#
#
# Range (0, 200)---------
# ADE: 0.465, fde: 0.451,

# Mean prediction
# Range (70, 130)---------
# ADE: 0.382, fde: 0.469,
#
#
# Range (0, 200)---------
# ADE: 0.512, fde: 0.554,

# Static
# Range (70, 130)---------
# ADE: 0.402, fde: 0.399,
#
#
# Range (0, 200)---------
# ADE: 0.543, fde: 0.538,

# extrapolation
# Range (70, 130)---------
# ADE: 0.419, fde: 0.401,
#
#
# Range (0, 200)---------
# ADE: 0.541, fde: 0.527,

###

# Fiery
# ade
# 36.5 & 47.8
# fde
# 45.9 & 54.4

# static
# ade
# 40.7 & 50.5
# fde
# 47.1 & 54.3

# extrapolate
# ade
# 40.5 & 50.3
# fde
# 46.9 & 53.2


if EVALUATE_ORACLE:
    print('EVALUATING ORACLE!!!!!!!!!')

if EVALUATE_N_SAMPLES:
    print('EVALUATING N SAMPLES !!!!!!!')


def eval(checkpoint_path, dataroot, version):
    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    device = torch.device('cuda:0')
    trainer.to(device)
    model = trainer.model

    cfg = model.cfg
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1

    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.VERSION = version

    if EVALUATE_ORACLE:
        cfg.TIME_RECEPTIVE_FIELD = 3
        cfg.N_FUTURE_FRAMES = 4
        model.receptive_field = cfg.TIME_RECEPTIVE_FIELD
        model.temporal_model.receptive_field = cfg.TIME_RECEPTIVE_FIELD
        model.n_future = cfg.N_FUTURE_FRAMES

        cfg.EVAL.EVALUATE_ORACLE = True

    _, valloader, _, val_dataset = prepare_dataloaders(cfg)

    panoptic_metrics = {}
    iou_metrics = {}
    n_classes = len(cfg.SEMANTIC_SEG.WEIGHTS)
    for key in EVALUATION_RANGES.keys():
        panoptic_metrics[key] = PanopticMetric(
            n_classes=n_classes, temporally_consistent=True, include_trajectory_metrics=INCLUDE_TRAJECTORY_METRICS,
            pixel_resolution=cfg.LIFT.X_BOUND[-1],
        ).to(device)
        #iou_metrics[key] = IntersectionOverUnion(n_classes).to(device)

    if EVALUATE_N_SAMPLES:
        diversity_distance_metrics = {}
        for key in EVALUATION_RANGES.keys():
            diversity_distance_metrics[key] = NSamplesPanopticMetric(
                num_classes=n_classes, temporally_consistent=True,
                include_trajectory_metrics=INCLUDE_TRAJECTORY_METRICS,
                pixel_resolution=cfg.LIFT.X_BOUND[-1],
            ).to(device)
        torch.manual_seed(0)

    #for i, batch in enumerate(tqdm(valloader)):

    for i in tqdm(range(0, len(val_dataset), 100)):
        batch = val_dataset[i]
        preprocess_batch(batch, device, unsqueeze=True)
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']

        batch_size = image.shape[0]

        labels, future_distribution_inputs = trainer.prepare_future_labels(batch)

        with torch.no_grad():

            if not EVALUATE_N_SAMPLES:
                # Evaluate with mean prediction
                noise = torch.zeros((batch_size, 1, model.latent_dim), device=device)

                t0 = time()
                output = model(image, intrinsics, extrinsics, future_egomotion,
                               future_distribution_inputs, noise=noise)
                t1 = time()

                #  Consistent instance seg
                pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
                    output, compute_matched_centers=False, make_consistent=(not EVALUATE_ORACLE)
                )

                if EVALUATE_ORACLE:
                    b, _, h, w = pred_consistent_instance_seg.shape
                    pred_consistent_instance_seg = pred_consistent_instance_seg.expand(b, model.n_future + 1, h, w)
                    # segmentation_pred = segmentation_pred.expand(b, model.n_future + 1, 1, h, w)
            else:
                pred_instance_samples = []
                for k in range(N_SAMPLES):
                    noise = NOISE_SCALE * torch.randn((batch_size, 1, model.latent_dim), device=device)
                    if N_SAMPLES == 1:
                        noise = torch.zeros_like(noise)
                    output = model(image, intrinsics, extrinsics, future_egomotion,
                                   future_distribution_inputs, noise=noise)

                    #  Consistent instance seg
                    pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
                        output, compute_matched_centers=False, make_consistent=True
                    )

                    if EVALUATE_ORACLE:
                        b, _, h, w = pred_consistent_instance_seg.shape
                        pred_consistent_instance_seg = pred_consistent_instance_seg.expand(b, model.n_future + 1, h, w)
                        # segmentation_pred = segmentation_pred.expand(b, model.n_future + 1, 1, h, w)

                    pred_instance_samples.append(pred_consistent_instance_seg)
                pred_instance_samples = torch.stack(pred_instance_samples, dim=1)


        #segmentation_pred = output['segmentation'].detach()
        #segmentation_pred = torch.argmax(segmentation_pred, dim=2, keepdims=True)

        for key, grid in EVALUATION_RANGES.items():
            limits = slice(grid[0], grid[1])

            if not EVALUATE_N_SAMPLES:
                panoptic_metrics[key](pred_consistent_instance_seg[..., limits, limits].contiguous().detach(),
                                      labels['instance'][..., limits, limits].contiguous()
                                      )
            else:
                diversity_distance_metrics[key](pred_instance_samples[..., limits, limits].contiguous(),
                                                labels['instance'][..., limits, limits].contiguous())

            # iou_metrics[key](segmentation_pred[..., limits, limits].contiguous().detach(),
            #                  labels['segmentation'][..., limits, limits].contiguous()
            #                  )

    results = {}

    if not EVALUATE_N_SAMPLES:
        for key, grid in EVALUATION_RANGES.items():
            panoptic_scores = panoptic_metrics[key].compute()
            for panoptic_key, value in panoptic_scores.items():
                results[f'{panoptic_key}'] = results.get(f'{panoptic_key}', []) + [100 * value[1].item()]

            #iou_scores = iou_metrics[key].compute()
            #results['iou'] = results.get('iou', []) + [100 * iou_scores[1].item()]

        for panoptic_key in ['pq', 'sq', 'rq', 'ade', 'fde']: #['iou'
            print(panoptic_key)
            print(' & '.join([f'{x:.1f}' for x in results[panoptic_key]]))
    else:
        for key, grid in EVALUATION_RANGES.items():
            print(f'Range {grid}---------')
            diversity_scores = diversity_distance_metrics[key].compute()
            print(
                f'ADE: {diversity_scores["ade"][1]:.3f}, fde: {diversity_scores["fde"][1]:.3f},')
            print('\n')


if __name__ == '__main__':
    parser = ArgumentParser(description='Fiery evaluation')
    parser.add_argument('--checkpoint', default='./fiery.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default='./nuscenes', type=str, help='path to the dataset')
    parser.add_argument('--version', default='trainval', type=str, choices=['mini', 'trainval'],
                        help='dataset version')

    args = parser.parse_args()

    eval(args.checkpoint, args.dataroot, args.version)
