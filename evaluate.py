from argparse import ArgumentParser

import torch
from tqdm import tqdm

from fiery.data import prepare_dataloaders
from fiery.trainer import TrainingModule
from fiery.metrics import IntersectionOverUnion, PanopticMetric
from fiery.utils.network import preprocess_batch
from fiery.utils.instance import predict_instance_segmentation_and_trajectories

# 30mx30m, 100mx100m
EVALUATION_RANGES = {'30x30': (70, 130),
                     '100x100': (0, 200)
                     }

EVALUATE_ORACLE = True
INCLUDE_TRAJECTORY_METRICS = True


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

    #for i, batch in enumerate(tqdm(valloader)):
    for i in tqdm(range(0, len(val_dataset), 10)):
        batch = val_dataset[i]
        preprocess_batch(batch, device, unsqueeze=True)
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']

        batch_size = image.shape[0]

        labels, future_distribution_inputs = trainer.prepare_future_labels(batch)

        with torch.no_grad():
            # Evaluate with mean prediction
            noise = torch.zeros((batch_size, 1, model.latent_dim), device=device)
            output = model(image, intrinsics, extrinsics, future_egomotion,
                           future_distribution_inputs, noise=noise)

        # Consistent instance seg
        pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
            output, compute_matched_centers=False, make_consistent=(not EVALUATE_ORACLE)
        )

        segmentation_pred = output['segmentation'].detach()
        segmentation_pred = torch.argmax(segmentation_pred, dim=2, keepdims=True)

        if EVALUATE_ORACLE:
            b, _, h, w = pred_consistent_instance_seg.shape
            pred_consistent_instance_seg = pred_consistent_instance_seg.expand(b, model.n_future + 1, h, w)
            #segmentation_pred = segmentation_pred.expand(b, model.n_future + 1, 1, h, w)


        for key, grid in EVALUATION_RANGES.items():
            limits = slice(grid[0], grid[1])
            panoptic_metrics[key](pred_consistent_instance_seg[..., limits, limits].contiguous().detach(),
                                  labels['instance'][..., limits, limits].contiguous()
                                  )

            # iou_metrics[key](segmentation_pred[..., limits, limits].contiguous().detach(),
            #                  labels['segmentation'][..., limits, limits].contiguous()
            #                  )

    results = {}
    for key, grid in EVALUATION_RANGES.items():
        panoptic_scores = panoptic_metrics[key].compute()
        for panoptic_key, value in panoptic_scores.items():
            results[f'{panoptic_key}'] = results.get(f'{panoptic_key}', []) + [100 * value[1].item()]

        #iou_scores = iou_metrics[key].compute()
        #results['iou'] = results.get('iou', []) + [100 * iou_scores[1].item()]

    for panoptic_key in ['pq', 'sq', 'rq', 'ade', 'fde']: #['iou'
        print(panoptic_key)
        print(' & '.join([f'{x:.1f}' for x in results[panoptic_key]]))


if __name__ == '__main__':
    parser = ArgumentParser(description='Fiery evaluation')
    parser.add_argument('--checkpoint', default='./fiery.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default='./nuscenes', type=str, help='path to the dataset')
    parser.add_argument('--version', default='trainval', type=str, choices=['mini', 'trainval'],
                        help='dataset version')

    args = parser.parse_args()

    eval(args.checkpoint, args.dataroot, args.version)
