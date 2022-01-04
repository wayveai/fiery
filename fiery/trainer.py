import torch
import torch.nn as nn
import pytorch_lightning as pl

from fiery.config import get_cfg
from fiery.models.fiery import Fiery
from fiery.losses import ProbabilisticLoss, SpatialRegressionLoss, SegmentationLoss
from fiery.metrics import IntersectionOverUnion, PanopticMetric
from fiery.utils.geometry import cumulative_warp_features_reverse, make_grid
from fiery.utils.instance import predict_instance_segmentation_and_trajectories
from fiery.utils.visualisation import visualise_output


from fiery.object_losses import compute_loss
from fiery.utils.object_encoder import ObjectEncoder
from fiery.utils.object_visualisation import visualize_bev

from fiery.utils.object_evaluation_utils import (cls_attr_dist, evaluate_json, lidar_egopose_to_world)

from nuscenes.nuscenes import NuScenes
import os
import json
from pyquaternion.quaternion import Quaternion
import numpy as np
import operator


class TrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # see config.py for details
        self.hparams = hparams
        # pytorch lightning does not support saving YACS CfgNone
        cfg = get_cfg(cfg_dict=self.hparams)
        self.cfg = cfg
        self.n_classes = len(self.cfg.SEMANTIC_SEG.WEIGHTS)

        # Bird's-eye view extent in meters
        assert self.cfg.LIFT.X_BOUND[1] > 0 and self.cfg.LIFT.Y_BOUND[1] > 0
        self.spatial_extent = (
            self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])

        # Model
        self.model = Fiery(cfg)

        # Losses
        self.losses_fn = nn.ModuleDict()
        self.losses_fn['segmentation'] = SegmentationLoss(
            class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.WEIGHTS),
            use_top_k=self.cfg.SEMANTIC_SEG.USE_TOP_K,
            top_k_ratio=self.cfg.SEMANTIC_SEG.TOP_K_RATIO,
            future_discount=self.cfg.FUTURE_DISCOUNT,
        )

        # Uncertainty weighting
        self.model.segmentation_weight = nn.Parameter(
            torch.tensor(0.0), requires_grad=True)

        self.metric_iou_val = IntersectionOverUnion(self.n_classes)

        self.losses_fn['instance_center'] = SpatialRegressionLoss(
            norm=2, future_discount=self.cfg.FUTURE_DISCOUNT
        )
        self.losses_fn['instance_offset'] = SpatialRegressionLoss(
            norm=1, future_discount=self.cfg.FUTURE_DISCOUNT, ignore_index=self.cfg.DATASET.IGNORE_INDEX
        )

        # Uncertainty weighting
        self.model.centerness_weight = nn.Parameter(
            torch.tensor(0.0), requires_grad=True)
        self.model.offset_weight = nn.Parameter(
            torch.tensor(0.0), requires_grad=True)

        self.metric_panoptic_val = PanopticMetric(n_classes=self.n_classes)

        if self.cfg.INSTANCE_FLOW.ENABLED:
            self.losses_fn['instance_flow'] = SpatialRegressionLoss(
                norm=1, future_discount=self.cfg.FUTURE_DISCOUNT, ignore_index=self.cfg.DATASET.IGNORE_INDEX
            )
            # Uncertainty weighting
            self.model.flow_weight = nn.Parameter(
                torch.tensor(0.0), requires_grad=True)

        if self.cfg.PROBABILISTIC.ENABLED:
            self.losses_fn['probabilistic'] = ProbabilisticLoss()

        self.training_step_count = 0

        # ObjectEncoder
        self.encoder = ObjectEncoder()

        # Nuscene
        if cfg.DATASET.NAME == 'nuscenes':
            # 28130 train and 6019 val
            self.version = cfg.DATASET.VERSION
            self.dataroot = os.path.join(cfg.DATASET.DATAROOT, self.version)
            self.nusc = NuScenes(version='{}'.format(cfg.DATASET.VERSION), dataroot=self.dataroot, verbose=False)
            self.nusc_annos = {'results': {}, 'meta': None}

    def prepare_future_labels(self, batch):
        labels = {}
        future_distribution_inputs = []

        segmentation_labels = batch['segmentation']
        instance_center_labels = batch['centerness']
        instance_offset_labels = batch['offset']
        instance_flow_labels = batch['flow']
        gt_instance = batch['instance']
        future_egomotion = batch['future_egomotion']

        # Warp labels to present's reference frame
        segmentation_labels = cumulative_warp_features_reverse(
            segmentation_labels[:, (self.model.receptive_field - 1):].float(),
            future_egomotion[:, (self.model.receptive_field - 1):],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).long().contiguous()
        labels['segmentation'] = segmentation_labels
        future_distribution_inputs.append(segmentation_labels)

        # Warp instance labels to present's reference frame
        gt_instance = cumulative_warp_features_reverse(
            gt_instance[:, (self.model.receptive_field - 1):].float().unsqueeze(2),
            future_egomotion[:, (self.model.receptive_field - 1):],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).long().contiguous()[:, :, 0]
        labels['instance'] = gt_instance

        instance_center_labels = cumulative_warp_features_reverse(
            instance_center_labels[:, (self.model.receptive_field - 1):],
            future_egomotion[:, (self.model.receptive_field - 1):],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).contiguous()
        labels['centerness'] = instance_center_labels

        instance_offset_labels = cumulative_warp_features_reverse(
            instance_offset_labels[:, (self.model.receptive_field - 1):],
            future_egomotion[:, (self.model.receptive_field - 1):],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).contiguous()
        labels['offset'] = instance_offset_labels

        future_distribution_inputs.append(instance_center_labels)
        future_distribution_inputs.append(instance_offset_labels)

        if self.cfg.INSTANCE_FLOW.ENABLED:
            instance_flow_labels = cumulative_warp_features_reverse(
                instance_flow_labels[:, (self.model.receptive_field - 1):],
                future_egomotion[:, (self.model.receptive_field - 1):],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()
            labels['flow'] = instance_flow_labels

            future_distribution_inputs.append(instance_flow_labels)

        if len(future_distribution_inputs) > 0:
            future_distribution_inputs = torch.cat(
                future_distribution_inputs, dim=2)

        return labels, future_distribution_inputs

    def get_gt_encoded(self, batch):
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

    def get_pre_encoded(self, output):
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

    def get_objects(self, pre_encoded, gt_encoded):
        # Visualzation
        heatmaps, gt_pos_offsets, gt_dim_offsets, gt_ang_offsets, mask = gt_encoded
        score, pos_offsets, dim_offsets, ang_offsets = pre_encoded

        # N = batch_size * time_length
        N = score.shape[0]
        grid = make_grid(
            [self.cfg.LIFT.X_BOUND[1] - self.cfg.LIFT.X_BOUND[0], self.cfg.LIFT.Y_BOUND[1] - self.cfg.LIFT.Y_BOUND[0]],
            [self.cfg.LIFT.X_BOUND[0], self.cfg.LIFT.Y_BOUND[0], 1.74],
            self.cfg.LIFT.X_BOUND[2]
        )
        # print("grid: ", grid)
        # print("grid.shape: ", grid.shape)

        grids = grid.unsqueeze(0).repeat(N, 1, 1, 1).cuda()
        # print("grids.shape: ", grids.shape)
        # print("grids: ", grids)

        objects = self.encoder.decode_batch(heatmaps=heatmaps,
                                            pos_offsets=gt_pos_offsets,
                                            dim_offsets=gt_dim_offsets,
                                            ang_offsets=gt_ang_offsets,
                                            grids=grids)
        pre_objects = self.encoder.decode_batch(heatmaps=score,
                                                pos_offsets=pos_offsets,
                                                dim_offsets=dim_offsets,
                                                ang_offsets=ang_offsets,
                                                grids=grids)
        # print("object[0]: ", objects[0])
        # print("pre_objects[0]: ", pre_objects[0])

        return pre_objects, objects, grids

    def shared_step(self, batch, is_train):
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']

        # print("image.shape: ", image.shape)

        # Warp labels
        labels, future_distribution_inputs = self.prepare_future_labels(batch)
        #####
        # Forward pass
        #####
        output = self.model(
            image, intrinsics, extrinsics, future_egomotion, future_distribution_inputs
        )
        #####
        # Encoded label
        #####
        gt_encoded = self.get_gt_encoded(batch)
        pre_encoded = self.get_pre_encoded(output)

        #####
        # OBJ Loss computation
        #####
        loss, loss_dict = compute_loss(pre_encoded, gt_encoded)
        #####
        # SEG Loss computation
        #####
        seg_loss = {}
        segmentation_factor = 1 / torch.exp(self.model.segmentation_weight)
        seg_loss['segmentation'] = segmentation_factor * self.losses_fn['segmentation'](
            output['segmentation'], labels['segmentation']
        )

        seg_loss['segmentation_uncertainty'] = 0.5 * self.model.segmentation_weight

        centerness_factor = 1 / (2 * torch.exp(self.model.centerness_weight))
        seg_loss['instance_center'] = centerness_factor * self.losses_fn['instance_center'](
            output['instance_center'], labels['centerness']
        )

        offset_factor = 1 / (2 * torch.exp(self.model.offset_weight))
        seg_loss['instance_offset'] = offset_factor * self.losses_fn['instance_offset'](
            output['instance_offset'], labels['offset']
        )

        seg_loss['centerness_uncertainty'] = 0.5 * self.model.centerness_weight
        seg_loss['offset_uncertainty'] = 0.5 * self.model.offset_weight

        if self.cfg.INSTANCE_FLOW.ENABLED:
            flow_factor = 1 / (2 * torch.exp(self.model.flow_weight))
            seg_loss['instance_flow'] = flow_factor * self.losses_fn['instance_flow'](
                output['instance_flow'], labels['flow']
            )

            seg_loss['flow_uncertainty'] = 0.5 * self.model.flow_weight

        if self.cfg.PROBABILISTIC.ENABLED:
            seg_loss['probabilistic'] = self.cfg.PROBABILISTIC.WEIGHT * \
                self.losses_fn['probabilistic'](output)

        # Metrics
        if not is_train:
            seg_prediction = output['segmentation'].detach()
            seg_prediction = torch.argmax(seg_prediction, dim=2, keepdims=True)
            self.metric_iou_val(seg_prediction, labels['segmentation'])

            pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
                output, compute_matched_centers=False
            )

            self.metric_panoptic_val(
                pred_consistent_instance_seg, labels['instance'])

        return pre_encoded, gt_encoded, loss, loss_dict, output, labels, seg_loss

    def visualise(self, labels, output, batch_idx, prefix='train'):
        visualisation_video = visualise_output(labels, output, self.cfg)
        name = f'{prefix}_outputs'
        # if prefix == 'val':
        # name = name + f'_{batch_idx}'
        self.logger.experiment.add_video(
            name, visualisation_video, global_step=self.training_step_count, fps=2)

    def training_step(self, batch, batch_idx):
        pre_encoded, gt_encoded, loss, loss_dict, output, labels, seg_loss = self.shared_step(batch, True)
        self.training_step_count += 1

        for key, value in seg_loss.items():
            self.logger.experiment.add_scalar(
                key, value, global_step=self.training_step_count)
        if self.training_step_count % self.cfg.VIS_INTERVAL == 0:
            self.visualise(labels, output, batch_idx, prefix='train')
        # return sum(loss.values())

        # Loggers
        for key, value in loss_dict.items():
            self.log(f'train_loss/{key}', value)
        self.log('train_loss', loss)

        if self.cfg.LOSS.SEG_USE is True:
            loss = self.cfg.LOSS.OBJ_LOSS_WEIGHT.ALL * loss + self.cfg.LOSS.SEG_LOSS_WEIGHT.ALL * sum(seg_loss.values())
        else:
            loss = (self.cfg.LOSS.OBJ_LOSS_WEIGHT.ALL + self.cfg.LOSS.SEG_LOSS_WEIGHT.ALL) * loss

        return {'loss': loss, 'log': loss_dict, 'progress_bar': loss_dict}

    def validation_step(self, batch, batch_idx):
        pre_encoded, gt_encoded, loss, loss_dict, output, labels, seg_loss = self.shared_step(batch, False)

        for key, value in seg_loss.items():
            self.log('val_' + key, value)
        self.visualise(labels, output, batch_idx, prefix='val')

        # Loggers
        for key, value in loss_dict.items():
            self.log(f'val_loss/{key}', value)
        self.log('val_loss', loss)

        # Visualzation
        pre_objects, objects, grids = self.get_objects(pre_encoded, gt_encoded)
        self.logger.experiment.add_figure(
            'val_visualize_bev',
            visualize_bev(
                objects,
                gt_encoded[0],
                pre_objects,
                pre_encoded[0],
                grids
            ),
            global_step=self.global_step
        )

        return {'val_loss': loss, 'log': loss_dict, 'progress_bar': loss_dict, 'pred_objects': pre_objects}

    def on_validation_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int):
        if self.cfg.EVALUATION is False:
            return

        tokens = batch['sample_token']
        tokens = [token for tokens_time_dim in tokens for token in tokens_time_dim]
        # b, s = tokens.shape[:2]
        # tokens = tokens.view(b * s, *tokens.shape[2:])
        batch_pred_objects = outputs['pred_objects']
        # print("tokens: ", (tokens))
        # print("batch_pred_objects.shape: ", len(batch_pred_objects))

        for pred_objects, token in zip(batch_pred_objects, tokens):
            annos = []
            for pred_object in pred_objects:
                egopose_to_world_trans, egopose_to_world_rot = lidar_egopose_to_world(token, self.nusc)

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
                if token in self.nusc_annos['results']:
                    nms_flag = 0
                    for all_cam_ann in self.nusc_annos['results'][token]:
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
            if token in self.nusc_annos['results']:
                if len(annos) < max_boxes_per_sample:
                    annos += self.nusc_annos['results'][token]
                    # print('len(annos): ', len(annos))
            self.nusc_annos['results'].update({token: annos})

    def on_validation_epoch_start(self):
        if self.cfg.EVALUATION is False:
            return

        self.nusc_annos = {'results': {}, 'meta': None}

    def on_validation_epoch_end(self) -> None:
        if self.cfg.EVALUATION is False:
            return

        self.nusc_annos['meta'] = {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
        print("number of token: ", len(self.nusc_annos["results"]))
        with open(os.path.join(self.cfg.EVA_DIR, 'detection_result.json'), "w") as f:
            json.dump(self.nusc_annos, f)

        evaluate_json(self.cfg.EVA_DIR, self.cfg.DATASET.VERSION, self.cfg.DATASET.DATAROOT)

    # def shared_epoch_end(self, step_outputs, is_train):
    #     # log per class iou metrics
    #     class_names = ['background', 'dynamic']
    #     if not is_train:
    #         scores = self.metric_iou_val.compute()
    #         for key, value in zip(class_names, scores):
    #             self.logger.experiment.add_scalar(
    #                 'val_iou_' + key, value, global_step=self.training_step_count)
    #         self.metric_iou_val.reset()

    #     if not is_train:
    #         scores = self.metric_panoptic_val.compute()
    #         for key, value in scores.items():
    #             for instance_name, score in zip(['background', 'vehicles'], value):
    #                 if instance_name != 'background':
    #                     self.logger.experiment.add_scalar(f'val_{key}_{instance_name}', score.item(),
    #                                                       global_step=self.training_step_count)
    #         self.metric_panoptic_val.reset()

    #     self.logger.experiment.add_scalar('segmentation_weight',
    #                                       1 /
    #                                       (torch.exp(self.model.segmentation_weight)),
    #                                       global_step=self.training_step_count)
    #     self.logger.experiment.add_scalar('centerness_weight',
    #                                       1 /
    #                                       (2 * torch.exp(self.model.centerness_weight)),
    #                                       global_step=self.training_step_count)
    #     self.logger.experiment.add_scalar('offset_weight', 1 / (2 * torch.exp(self.model.offset_weight)),
    #                                       global_step=self.training_step_count)
    #     if self.cfg.INSTANCE_FLOW.ENABLED:
    #         self.logger.experiment.add_scalar('flow_weight', 1 / (2 * torch.exp(self.model.flow_weight)),
    #                                           global_step=self.training_step_count)

    # def training_epoch_end(self, step_outputs):
    #     self.shared_epoch_end(step_outputs, True)

    # def validation_epoch_end(self, step_outputs):
    #     self.shared_epoch_end(step_outputs, False)

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = torch.optim.Adam(
            params, lr=self.cfg.OPTIMIZER.LR, weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY
        )

        return optimizer
