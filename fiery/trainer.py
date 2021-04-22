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
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])

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
        self.model.segmentation_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.metric_iou_val = IntersectionOverUnion(self.n_classes)

        self.losses_fn['instance_center'] = SpatialRegressionLoss(
            norm=2, future_discount=self.cfg.FUTURE_DISCOUNT
        )
        self.losses_fn['instance_offset'] = SpatialRegressionLoss(
            norm=1, future_discount=self.cfg.FUTURE_DISCOUNT, ignore_index=self.cfg.DATASET.IGNORE_INDEX
        )

        # Uncertainty weighting
        self.model.centerness_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.model.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.metric_panoptic_val = PanopticMetric(n_classes=self.n_classes)

        if self.cfg.INSTANCE_FLOW.ENABLED:
            self.losses_fn['instance_flow'] = SpatialRegressionLoss(
                norm=1, future_discount=self.cfg.FUTURE_DISCOUNT, ignore_index=self.cfg.DATASET.IGNORE_INDEX
            )
            # Uncertainty weighting
            self.model.flow_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        if self.cfg.PROBABILISTIC.ENABLED:
            self.losses_fn['probabilistic'] = ProbabilisticLoss()

        self.training_step_count = 0

    def shared_step(self, batch, is_train):
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']

        # Warp labels
        labels, future_distribution_inputs = self.prepare_future_labels(batch)

        # Forward pass
        output = self.model(
            image, intrinsics, extrinsics, future_egomotion, future_distribution_inputs
        )

        #####
        # Loss computation
        #####
        loss = {}
        segmentation_factor = 1 / torch.exp(self.model.segmentation_weight)
        loss['segmentation'] = segmentation_factor * self.losses_fn['segmentation'](
            output['segmentation'], labels['segmentation']
        )
        loss['segmentation_uncertainty'] = 0.5 * self.model.segmentation_weight

        centerness_factor = 1 / (2*torch.exp(self.model.centerness_weight))
        loss['instance_center'] = centerness_factor * self.losses_fn['instance_center'](
            output['instance_center'], labels['centerness']
        )

        offset_factor = 1 / (2*torch.exp(self.model.offset_weight))
        loss['instance_offset'] = offset_factor * self.losses_fn['instance_offset'](
            output['instance_offset'], labels['offset']
        )

        loss['centerness_uncertainty'] = 0.5 * self.model.centerness_weight
        loss['offset_uncertainty'] = 0.5 * self.model.offset_weight

        if self.cfg.INSTANCE_FLOW.ENABLED:
            flow_factor = 1 / (2*torch.exp(self.model.flow_weight))
            loss['instance_flow'] = flow_factor * self.losses_fn['instance_flow'](
                output['instance_flow'], labels['flow']
            )

            loss['flow_uncertainty'] = 0.5 * self.model.flow_weight

        if self.cfg.PROBABILISTIC.ENABLED:
            loss['probabilistic'] = self.cfg.PROBABILISTIC.WEIGHT * self.losses_fn['probabilistic'](output)

        # Metrics
        if not is_train:
            seg_prediction = output['segmentation'].detach()
            seg_prediction = torch.argmax(seg_prediction, dim=2, keepdims=True)
            self.metric_iou_val(seg_prediction, labels['segmentation'])

            pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
                output, compute_matched_centers=False
            )

            self.metric_panoptic_val(pred_consistent_instance_seg, labels['instance'])

        return output, labels, loss

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
            instance_offset_labels[:, (self.model.receptive_field- 1):],
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
            future_distribution_inputs = torch.cat(future_distribution_inputs, dim=2)

        return labels, future_distribution_inputs

    def visualise(self, labels, output, batch_idx, prefix='train'):
        visualisation_video = visualise_output(labels, output, self.cfg)
        name = f'{prefix}_outputs'
        if prefix == 'val':
            name = name + f'_{batch_idx}'
        self.logger.experiment.add_video(name, visualisation_video, global_step=self.training_step_count, fps=2)

    def training_step(self, batch, batch_idx):
        output, labels, loss = self.shared_step(batch, True)
        self.training_step_count += 1
        for key, value in loss.items():
            self.logger.experiment.add_scalar(key, value, global_step=self.training_step_count)
        if self.training_step_count % self.cfg.VIS_INTERVAL == 0:
            self.visualise(labels, output, batch_idx, prefix='train')
        return sum(loss.values())

    def validation_step(self, batch, batch_idx):
        output, labels, loss = self.shared_step(batch, False)
        for key, value in loss.items():
            self.log('val_' + key, value)

        if batch_idx == 0:
            self.visualise(labels, output, batch_idx, prefix='val')

    def shared_epoch_end(self, step_outputs, is_train):
        # log per class iou metrics
        class_names = ['background', 'dynamic']
        if not is_train:
            scores = self.metric_iou_val.compute()
            for key, value in zip(class_names, scores):
                self.logger.experiment.add_scalar('val_iou_' + key, value, global_step=self.training_step_count)
            self.metric_iou_val.reset()

        if not is_train:
            scores = self.metric_panoptic_val.compute()
            for key, value in scores.items():
                for instance_name, score in zip(['background', 'vehicles'], value):
                    if instance_name != 'background':
                        self.logger.experiment.add_scalar(f'val_{key}_{instance_name}', score.item(),
                                                          global_step=self.training_step_count)
            self.metric_panoptic_val.reset()

        self.logger.experiment.add_scalar('segmentation_weight',
                                          1 / (torch.exp(self.model.segmentation_weight)),
                                          global_step=self.training_step_count)
        self.logger.experiment.add_scalar('centerness_weight',
                                          1 / (2 * torch.exp(self.model.centerness_weight)),
                                          global_step=self.training_step_count)
        self.logger.experiment.add_scalar('offset_weight', 1 / (2 * torch.exp(self.model.offset_weight)),
                                          global_step=self.training_step_count)
        if self.cfg.INSTANCE_FLOW.ENABLED:
            self.logger.experiment.add_scalar('flow_weight', 1 / (2 * torch.exp(self.model.flow_weight)),
                                              global_step=self.training_step_count)

    def training_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, True)

    def validation_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, False)

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = torch.optim.Adam(
            params, lr=self.cfg.OPTIMIZER.LR, weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY
        )

        return optimizer
