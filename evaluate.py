import os
import json
import torch
import operator
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser


from fiery.data import prepare_dataloaders
from fiery.trainer import TrainingModule
from fiery.metrics import IntersectionOverUnion, PanopticMetric
from fiery.utils.network import preprocess_batch
from fiery.utils.instance import predict_instance_segmentation_and_trajectories
from fiery.utils.geometry import make_grid

from nuscenes import NuScenes
from pyquaternion import Quaternion
from fiery.object_losses import compute_loss
from fiery.utils.object_encoder import ObjectEncoder
from fiery.utils.object_evaluation_utils import evaluate_json, cls_attr_dist, lidar_egopose_to_world

# 30mx30m, 100mx100m
EVALUATION_RANGES = {'30x30': (70, 130),
                     '100x100': (0, 200)
                     }
encoder = ObjectEncoder()


def get_gt_encoded(batch):
    heatmaps = batch['heatmaps']
    gt_pos_offsets = batch['gt_pos_offsets']
    gt_dim_offsets = batch['gt_dim_offsets']
    gt_ang_offsets = batch['gt_ang_offsets']
    mask = batch['mask']
    # Reshape for compute loss
    # b: batch_size, s: time_length
    b, s = heatmaps.shape[:2]
    # print("b, s: ", b, s)
    #  reshape elements in gt_encoded
    heatmaps = heatmaps.view(b * s, *heatmaps.shape[2:])
    gt_pos_offsets = gt_pos_offsets.view(b * s, *gt_pos_offsets.shape[2:])
    gt_dim_offsets = gt_dim_offsets.view(b * s, *gt_dim_offsets.shape[2:])
    gt_ang_offsets = gt_ang_offsets.view(b * s, *gt_ang_offsets.shape[2:])
    mask = mask.view(b * s, *mask.shape[2:])
    # print("heatmaps: ", heatmaps.shape)
    # print("gt_pos_offsets: ", gt_pos_offsets.shape)
    # print("gt_dim_offsets: ", gt_dim_offsets.shape)
    # print("gt_ang_offsets: ", gt_ang_offsets.shape)
    # print("mask: ", mask.shape)

    return heatmaps, gt_pos_offsets, gt_dim_offsets, gt_ang_offsets, mask


def get_pre_encoded(output):
    score = output['score']
    pos_offsets = output['pos_offsets']
    dim_offsets = output['dim_offsets']
    ang_offsets = output['ang_offsets']
    # Reshape for compute loss
    # b: batch_size, s: time_length
    b, s = score.shape[:2]
    # print("b, s: ", b, s)
    #  reshape elements in pre_encoded
    score = score.view(b * s, *score.shape[2:])
    pos_offsets = pos_offsets.view(b * s, *pos_offsets.shape[2:])
    dim_offsets = dim_offsets.view(b * s, *dim_offsets.shape[2:])
    ang_offsets = ang_offsets.view(b * s, *ang_offsets.shape[2:])
    # print("score: ", score.shape)
    # print("pos_offsets: ", pos_offsets.shape)
    # print("dim_offsets: ", dim_offsets.shape)
    # print("ang_offsets: ", ang_offsets.shape)

    return score, pos_offsets, dim_offsets, ang_offsets


def get_objects(pre_encoded, gt_encoded, cfg):
    # Visualzation
    heatmaps, gt_pos_offsets, gt_dim_offsets, gt_ang_offsets, mask = gt_encoded
    score, pos_offsets, dim_offsets, ang_offsets = pre_encoded

    # N = batch_size * time_length
    N = score.shape[0]
    grid = make_grid(
        [cfg.LIFT.X_BOUND[1] - cfg.LIFT.X_BOUND[0], cfg.LIFT.Y_BOUND[1] - cfg.LIFT.Y_BOUND[0]],
        [cfg.LIFT.X_BOUND[0], cfg.LIFT.Y_BOUND[0], 1.74],
        cfg.LIFT.X_BOUND[2]
    )
    # print("grid: ", grid)
    # print("grid.shape: ", grid.shape)

    grids = grid.unsqueeze(0).repeat(N, 1, 1, 1).cuda()
    # print("grids.shape: ", grids.shape)
    # print("grids: ", grids)

    objects = encoder.decode_batch(heatmaps=heatmaps,
                                   pos_offsets=gt_pos_offsets,
                                   dim_offsets=gt_dim_offsets,
                                   ang_offsets=gt_ang_offsets,
                                   grids=grids)
    pre_objects = encoder.decode_batch(heatmaps=score,
                                       pos_offsets=pos_offsets,
                                       dim_offsets=dim_offsets,
                                       ang_offsets=ang_offsets,
                                       grids=grids)
    # print("object[0]: ", objects[0])
    # print("pre_objects[0]: ", pre_objects[0])

    return pre_objects, objects, grids


def eval(checkpoint_path, dataroot, version):
    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    device = torch.device('cuda:0')
    trainer.to(device)
    model = trainer.model

    cfg = model.cfg
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 4

    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.VERSION = version

    _, valloader = prepare_dataloaders(cfg)

    # panoptic_metrics = {}
    # iou_metrics = {}
    # n_classes = len(cfg.SEMANTIC_SEG.WEIGHTS)
    #####
    dataroot = os.path.join(cfg.DATASET.DATAROOT, version)
    nusc = NuScenes(version='{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=False)
    nusc_annos = {'results': {}, 'meta': None}

    # for key in EVALUATION_RANGES.keys():
    #     panoptic_metrics[key] = PanopticMetric(n_classes=n_classes, temporally_consistent=True).to(
    #         device)
    #     iou_metrics[key] = IntersectionOverUnion(n_classes).to(device)

    for i, batch in enumerate(tqdm(valloader)):
        preprocess_batch(batch, device)
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']

        batch_size = image.shape[0]

        labels, future_distribution_inputs = trainer.prepare_future_labels(batch)

        with torch.no_grad():
            # Evaluate with mean prediction
            noise = torch.zeros((batch_size, 1, model.latent_dim), device=device)
            output = model(image, intrinsics, extrinsics, future_egomotion,
                           future_distribution_inputs, noise=noise)
            #####
            # Encoded label
            #####
            gt_encoded = get_gt_encoded(batch)
            pre_encoded = get_pre_encoded(output)

            #####
            # Loss computation
            #####
            loss, loss_dict = compute_loss(pre_encoded, gt_encoded)
            batch_pred_objects, objects, grids = get_objects(pre_encoded, gt_encoded, cfg)

            tokens = batch['sample_token']
            tokens = [token for tokens_time_dim in tokens for token in tokens_time_dim]
            # b, s = tokens.shape[:2]
            # tokens = tokens.view(b * s, *tokens.shape[2:])
            # batch_pred_objects = pre_objects
            # print("tokens: ", (tokens))
            # print("batch_pred_objects.shape: ", len(batch_pred_objects))

            for pred_objects, token in zip(batch_pred_objects, tokens):
                annos = []
                for pred_object in pred_objects:
                    egopose_to_world_trans, egopose_to_world_rot = lidar_egopose_to_world(token, nusc)

                    # transform from egopose to world -> translation
                    translation = pred_object.position
                    translation = egopose_to_world_rot.rotation_matrix @ translation.numpy()
                    translation += egopose_to_world_trans

                    # transform from egopose to world -> rotation
                    rotation = Quaternion(axis=[0, 0, 1], angle=pred_object.angle)
                    rotation = egopose_to_world_rot * rotation

                    # transform from egopose to world -> velocity
                    velocity = np.array([0.0, 0.0, 0.0])
                    velocity = egopose_to_world_rot.rotation_matrix @ velocity

                    size = pred_object.dimensions
                    name = pred_object.classname
                    score = pred_object.score

                    nusc_anno = {
                        'sample_token': token,
                        'translation': translation.tolist(),
                        'size': size.tolist(),
                        'rotation': rotation.elements.tolist(),
                        'velocity': [velocity[0], velocity[1]],
                        'detection_name': name,
                        'detection_score': score.tolist(),
                        'attribute_name': max(cls_attr_dist[name].items(),
                                              key=operator.itemgetter(1))[0],
                    }
                    # align six camera
                    if token in nusc_annos['results']:
                        nms_flag = 0
                        for all_cam_ann in nusc_annos['results'][token]:
                            if nusc_anno['detection_name'] == all_cam_ann['detection_name']:
                                translation = nusc_anno['translation']
                                ref_translation = all_cam_ann['translation']
                                translation_diff = (translation[0] - ref_translation[0],
                                                    translation[1] - ref_translation[1],
                                                    translation[2] - ref_translation[2])
                                if nusc_anno['detection_name'] in ['pedestrian']:
                                    nms_dist = 1
                                else:
                                    nms_dist = 2
                                if np.linalg.norm(translation_diff[:2]) < nms_dist:
                                    if all_cam_ann['detection_score'] < nusc_anno['detection_score']:
                                        all_cam_ann = nusc_anno
                                    nms_flag = 1
                                    break
                        if nms_flag == 0:
                            annos.append(nusc_anno)
                    else:
                        annos.append(nusc_anno)
                max_boxes_per_sample = 450
                if token in nusc_annos['results']:
                    if len(annos) < max_boxes_per_sample:
                        annos += nusc_annos['results'][token]
                        # print('len(annos): ', len(annos))
                nusc_annos['results'].update({token: annos})

        # Consistent instance seg
        # pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
        #     output, compute_matched_centers=False, make_consistent=True
        # )

        # segmentation_pred = output['segmentation'].detach()
        # segmentation_pred = torch.argmax(segmentation_pred, dim=2, keepdims=True)

        # for key, grid in EVALUATION_RANGES.items():
        #     limits = slice(grid[0], grid[1])
        #     panoptic_metrics[key](pred_consistent_instance_seg[..., limits, limits].contiguous().detach(),
        #                           labels['instance'][..., limits, limits].contiguous()
        #                           )

        #     iou_metrics[key](segmentation_pred[..., limits, limits].contiguous(),
        #                      labels['segmentation'][..., limits, limits].contiguous()
        #                      )

    nusc_annos['meta'] = {
        "use_camera": True,
        "use_lidar": False,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }
    print("number of token: ", len(nusc_annos["results"]))
    with open(os.path.join(cfg.EVA_DIR, 'detection_result.json'), "w") as f:
        json.dump(nusc_annos, f)

    evaluate_json(cfg.EVA_DIR, cfg.DATASET.VERSION, cfg.DATASET.DATAROOT)

    # results = {}
    # for key, grid in EVALUATION_RANGES.items():
    #     panoptic_scores = panoptic_metrics[key].compute()
    #     for panoptic_key, value in panoptic_scores.items():
    #         results[f'{panoptic_key}'] = results.get(f'{panoptic_key}', []) + [100 * value[1].item()]

    #     iou_scores = iou_metrics[key].compute()
    #     results['iou'] = results.get('iou', []) + [100 * iou_scores[1].item()]

    # for panoptic_key in ['iou', 'pq', 'sq', 'rq']:
    #     print(panoptic_key)
    #     print(' & '.join([f'{x:.1f}' for x in results[panoptic_key]]))


if __name__ == '__main__':
    parser = ArgumentParser(description='Fiery evaluation')
    parser.add_argument('--checkpoint', default='./fiery.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default='/home/master/10/cytseng/data/sets/nuscenes/',
                        type=str, help='path to the dataset')
    parser.add_argument('--version', default='v1.0-trainval', type=str, choices=['v1.0-mini', 'v1.0-trainval'],
                        help='dataset version')

    args = parser.parse_args()
    ckpt_path = '/home/master/10/cytseng/fiery/tensorboard_logs/lss_oft_multi_cam/version_2/checkpoints/' + 'epoch=0-step=1405.ckpt'
    eval(ckpt_path, args.dataroot, args.version)
