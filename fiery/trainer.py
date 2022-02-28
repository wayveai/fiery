import os
import json
import operator
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl

from fiery.config import get_cfg
from fiery.models.fiery import Fiery
from fiery.losses import ProbabilisticLoss, SpatialRegressionLoss, SegmentationLoss
from fiery.metrics import IntersectionOverUnion, PanopticMetric
from fiery.utils.geometry import cumulative_warp_features_reverse
from fiery.utils.instance import predict_instance_segmentation_and_trajectories
from fiery.utils.visualisation import visualise_output


from fiery.utils.nuscenes_visualization import visualize_bbox, visualize_sample, visualize_center

from fiery.utils.object_evaluation_utils import (cls_attr_dist, evaluate_json, lidar_egopose_to_world)
from fiery.utils.mm_obj_evaluation_utils import output_to_nusc_box

from nuscenes.nuscenes import NuScenes
from pyquaternion.quaternion import Quaternion
from mmdet3d.core import bbox3d2result


class TrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # see config.py for details
        # self.hparams = hparams
        self.save_hyperparameters(hparams)
        # pytorch lightning does not support saving YACS CfgNone
        cfg = get_cfg(cfg_dict=self.hparams)
        self.cfg = cfg

        if self.cfg.SEMANTIC_SEG.NUSCENE_CLASS:
            SEMANTIC_SEG_WEIGHTS = [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        else:
            SEMANTIC_SEG_WEIGHTS = [1.0, 2.0]

        # self.n_classes = len(self.cfg.SEMANTIC_SEG.WEIGHTS)
        self.n_classes = len(SEMANTIC_SEG_WEIGHTS)

        # Bird's-eye view extent in meters
        assert self.cfg.LIFT.X_BOUND[1] > 0 and self.cfg.LIFT.Y_BOUND[1] > 0
        self.spatial_extent = (
            self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])

        # Model
        self.model = Fiery(cfg)

        if self.cfg.LOSS.SEG_USE:
            # Losses
            self.losses_fn = nn.ModuleDict()

            self.losses_fn['segmentation'] = SegmentationLoss(
                # class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.WEIGHTS),
                class_weights=torch.Tensor(SEMANTIC_SEG_WEIGHTS),

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

    def shared_step(self, batch, is_train):
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']

        # Warp labels
        labels, future_distribution_inputs = self.prepare_future_labels(batch)
        #####
        # Forward pass
        #####
        output = self.model(
            image, intrinsics, extrinsics, future_egomotion, future_distribution_inputs
        )

        #####
        # OBJ Loss computation
        #####
        if self.cfg.OBJ.HEAD_NAME == 'mm':
            detection_output = output['detection_output']
            loss_dict = self.model.detection_head.loss(batch, detection_output)
            loss = torch.stack([loss_value for loss_value in loss_dict.values()]).sum()
            loss_dict['total'] = loss

        #####
        # SEG Loss computation
        #####
        seg_loss = {}
        if self.cfg.LOSS.SEG_USE:
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

        return loss, loss_dict, output, labels, seg_loss

    def visualise(self, labels, output, global_step=None, prefix='train'):
        visualisation_video = visualise_output(labels, output, self.cfg)
        name = f'{prefix}_seg_outputs'
        # if prefix == 'val':
        # name = name + f'_{batch_idx}'
        self.logger.experiment.add_video(
            name,
            visualisation_video,
            global_step=self.global_step if global_step is None else global_step,
            fps=2
        )

    #####
    # training_step
    #####
    def training_step(self, batch, batch_idx):
        loss, loss_dict, output, labels, seg_loss = self.shared_step(batch, True)

        if self.cfg.LOSS.SEG_USE:
            #####
            # SEG Loss Logger
            #####
            for key, value in seg_loss.items():
                self.log(f'train_seg_loss/{key}', value)
            if batch_idx == 0:
                self.visualise(labels, output, prefix='train')

            loss = self.cfg.LOSS.OBJ_LOSS_WEIGHT.ALL * loss + self.cfg.LOSS.SEG_LOSS_WEIGHT.ALL * sum(seg_loss.values())
        else:
            loss = (self.cfg.LOSS.OBJ_LOSS_WEIGHT.ALL + self.cfg.LOSS.SEG_LOSS_WEIGHT.ALL) * loss

        if self.cfg.OBJ.HEAD_NAME == 'mm':
            if batch_idx == 0:
                self.mm_visualize(
                    batch,
                    output['detection_output'],
                    prefix='train'
                )
        #####
        # OBJ Loss Logger
        #####
        for key, value in loss_dict.items():
            self.log(f'train_obj_loss/{key}', value)

        return {'loss': loss}

    #####
    # validation_step
    #####
    def validation_step(self, batch, batch_idx):
        loss, loss_dict, output, labels, seg_loss = self.shared_step(batch, False)

        if self.cfg.LOSS.SEG_USE:
            #####
            # SEG Loss Logger
            #####
            for key, value in seg_loss.items():
                self.log(f'val_seg_loss/{key}', value, batch_size=self.cfg.VAL_BATCHSIZE)
            if batch_idx == 0:
                self.visualise(labels, output, prefix='val')
            loss = self.cfg.LOSS.OBJ_LOSS_WEIGHT.ALL * loss + self.cfg.LOSS.SEG_LOSS_WEIGHT.ALL * sum(seg_loss.values())
        else:
            loss = (self.cfg.LOSS.OBJ_LOSS_WEIGHT.ALL + self.cfg.LOSS.SEG_LOSS_WEIGHT.ALL) * loss

        #####
        # OBJ Loss Logger
        #####
        for key, value in loss_dict.items():
            self.log(f'val_obj_loss/{key}', value, batch_size=self.cfg.VAL_BATCHSIZE)

        output_dict = {'val_loss': loss}
        # Visualzation & Evaluation
        tokens = batch['sample_token']
        tokens = [token for tokens_time_dim in tokens for token in tokens_time_dim]
        if self.cfg.OBJ.HEAD_NAME == 'mm':
            pred_bboxes_list = self.model.detection_head.get_bboxes(batch, output['detection_output'])
            if batch_idx == 0:
                self.mm_visualize(
                    batch,
                    output['detection_output'],
                    pred_bboxes_list=pred_bboxes_list,
                    tokens=tokens,
                    prefix='val'
                )
            if self.cfg.EVALUATION:
                self.mm_obj_evaluation(tokens, pred_bboxes_list)

        return output_dict

    def mm_obj_evaluation(self, tokens, detections):
        for (bboxes, scores, labels), token in zip(detections, tokens):
            annos = []
            pred_boxes = output_to_nusc_box(
                bbox3d2result(
                    bboxes,
                    scores.detach().cpu(),
                    labels.detach().cpu()
                ),
                token,
                is_eval=True
            )
            for pred_box in pred_boxes:
                egopose_to_world_trans, egopose_to_world_rot = lidar_egopose_to_world(token, self.nusc)

                # transform from egopose to world
                pred_box.rotate(Quaternion(egopose_to_world_rot))
                pred_box.translate(np.array(egopose_to_world_trans))

                nusc_anno = {
                    'sample_token': pred_box.token,
                    'translation': pred_box.center.tolist(),
                    'size': pred_box.wlh.tolist(),
                    'rotation': pred_box.orientation.elements.tolist(),
                    'velocity': pred_box.velocity[:2].tolist(),
                    'detection_name': pred_box.name,
                    'detection_score': pred_box.score,
                    'attribute_name': max(cls_attr_dist[pred_box.name].items(),
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
        if self.cfg.EVALUATION:
            self.nusc_annos = {'results': {}, 'meta': None}

    def on_validation_epoch_end(self) -> None:
        if self.cfg.EVALUATION:
            self.nusc_annos['meta'] = {
                "use_camera": True,
                "use_lidar": False,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            }
            print("number of token: ", len(self.nusc_annos["results"]))

            os.makedirs(self.cfg.EVA_DIR, exist_ok=True)

            with open(os.path.join(self.cfg.EVA_DIR, 'detection_result.json'), "w") as f:
                json.dump(self.nusc_annos, f)

            evaluate_json(self.cfg.EVA_DIR, self.cfg.DATASET.VERSION, self.cfg.DATASET.DATAROOT)

        self.trainer.save_checkpoint(os.path.join(self.trainer.log_dir, 'checkpoints',
                                     f'{self.trainer.current_epoch}.ckpt'))

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
        if self.cfg.OPTIMIZER.NAME == 'AdamW':
            optimizer = torch.optim.AdamW(
                params, lr=self.cfg.OPTIMIZER.LR, weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY
            )
        elif self.cfg.OPTIMIZER.NAME == 'Adam':
            optimizer = torch.optim.Adam(
                params, lr=self.cfg.OPTIMIZER.LR, weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY
            )

        return optimizer

    def mm_visualize(self, batch, preds_dicts, pred_bboxes_list=None, tokens=None, global_step=None, prefix='val'):
        if global_step is None:
            global_step = self.global_step
        if tokens is None:
            tokens = batch['sample_token']
            tokens = [token for tokens_time_dim in tokens for token in tokens_time_dim]

        gt_bboxes_3d = [item[0] for item in batch['gt_bboxes_3d']]
        gt_labels_3d = [item[0] for item in batch['gt_labels_3d']]

        # if pred_bboxes_list is not None, use it. Otherwise get bboxes from the detection head
        if pred_bboxes_list is None:
            pred_bboxes_list = self.model.detection_head.get_bboxes(batch, preds_dicts)
        preds_heatmaps, gt_heatmaps = self.model.detection_head.get_heatmaps(batch, preds_dicts)

        for pred_bboxes, gt_bbox_3d, gt_label_3d, token, pred_heatmap, gt_heatmap in zip(
            pred_bboxes_list,
            gt_bboxes_3d,
            gt_labels_3d,
            tokens,
            preds_heatmaps['task_0.heatmap'],
            gt_heatmaps[0]
        ):
            # show bbox between mm_gt(green) and mm_pred(blue)
            self.logger.experiment.add_figure(
                f'{prefix}_mm_bev',
                visualize_bbox(
                    pred_bboxes=pred_bboxes,
                    gt_bbox_3d=gt_bbox_3d,
                    gt_label_3d=gt_label_3d,
                    token=token,
                ),
                global_step=global_step
            )

            # show bbox between mm_gt(green) and nusc_gt(red)
            self.logger.experiment.add_figure(
                f'{prefix}_eval_bev',
                visualize_sample(
                    nusc=self.nusc,
                    pred_boxes=output_to_nusc_box(
                        bbox3d2result(gt_bbox_3d, torch.ones_like(gt_label_3d), gt_label_3d),
                        token,
                        is_eval=True
                    ),
                    sample_token=token,
                ),
                global_step=global_step
            )

            # get heatmap from task 0 and batch_idx 0
            pred_heatmap = pred_heatmap.sum(dim=0)
            gt_heatmap = gt_heatmap.sum(dim=0)

            # show center heatmap between mm_gt(left) and mm_pred(right)
            self.logger.experiment.add_image(
                f'{prefix}_mm_heatmap',
                visualize_center(pred_heatmap, gt_heatmap),
                dataformats='HWC',
                global_step=global_step
            )
            break

    #####
    # test_step
    #####
    def test_step(self, batch, batch_idx):
        loss, loss_dict, output, labels, seg_loss = self.shared_step(batch, False)

        if self.cfg.LOSS.SEG_USE:
            #####
            # SEG Loss Logger
            #####
            for key, value in seg_loss.items():
                self.log(f'test_seg_loss/{key}', value, batch_size=self.cfg.VAL_BATCHSIZE)
            if batch_idx % 150 == 0:
                self.visualise(labels, output, global_step=batch_idx, prefix='test')
            loss = self.cfg.LOSS.OBJ_LOSS_WEIGHT.ALL * loss + self.cfg.LOSS.SEG_LOSS_WEIGHT.ALL * sum(seg_loss.values())
        else:
            loss = (self.cfg.LOSS.OBJ_LOSS_WEIGHT.ALL + self.cfg.LOSS.SEG_LOSS_WEIGHT.ALL) * loss

        #####
        # OBJ Loss Logger
        #####
        for key, value in loss_dict.items():
            self.log(f'test_obj_loss/{key}', value, batch_size=self.cfg.VAL_BATCHSIZE)

        output_dict = {'test_loss': loss}

        # Visualzation & Evaluation
        tokens = batch['sample_token']
        tokens = [token for tokens_time_dim in tokens for token in tokens_time_dim]
        if self.cfg.OBJ.HEAD_NAME == 'mm':
            pred_bboxes_list = self.model.detection_head.get_bboxes(batch, output['detection_output'])
            if batch_idx % 150 == 0:
                self.mm_visualize(
                    batch,
                    output['detection_output'],
                    pred_bboxes_list=pred_bboxes_list,
                    tokens=tokens,
                    global_step=batch_idx,
                    prefix='test'
                )
            self.mm_obj_evaluation(tokens, pred_bboxes_list)

        return output_dict

    def on_test_epoch_start(self):
        self.nusc_annos = {'results': {}, 'meta': None}

    def on_test_epoch_end(self) -> None:
        self.nusc_annos['meta'] = {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
        print("number of token: ", len(self.nusc_annos["results"]))

        os.makedirs(self.cfg.EVA_DIR, exist_ok=True)

        with open(os.path.join(self.cfg.EVA_DIR, 'detection_result.json'), "w") as f:
            json.dump(self.nusc_annos, f)

        evaluate_json(self.cfg.EVA_DIR, self.cfg.DATASET.VERSION, self.cfg.DATASET.DATAROOT, self.cfg.TEST_TRAINSET)

        # self.trainer.save_checkpoint(os.path.join(self.trainer.log_dir, 'checkpoints',
        #                              f'{self.trainer.current_epoch}.ckpt'))
