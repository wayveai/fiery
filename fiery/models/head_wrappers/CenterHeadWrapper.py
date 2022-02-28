from mmdet3d.models.builder import HEADS
from mmdet3d.models.dense_heads import CenterHead
import torch


@HEADS.register_module()
class CenterHeadWrapper(CenterHead):
    """CenterHead for CenterPoint.

    Args:
        mode (str): Mode of the head. Default: '3d'.
        in_channels (list[int] | int): Channels of the input feature map.
            Default: [128].
        tasks (list[dict]): Task information including class number
            and class names. Default: None.
        dataset (str): Name of the dataset. Default: 'nuscenes'.
        weight (float): Weight for location loss. Default: 0.25.
        code_weights (list[int]): Code weights for location loss. Default: [].
        common_heads (dict): Conv information for common heads.
            Default: dict().
        loss_cls (dict): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int): Output channels for share_conv_layer.
            Default: 64.
        num_heatmap_convs (int): Number of conv layers for heatmap conv layer.
            Default: 2.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels=[128],
                 tasks=None,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 common_heads=dict(),
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(
                     type='L1Loss', reduction='none', loss_weight=0.25),
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 norm_bbox=True,
                 init_cfg=None):
        super().__init__(
            in_channels=in_channels,
            tasks=tasks,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            bbox_coder=bbox_coder,
            common_heads=common_heads,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            separate_head=separate_head,
            share_conv_channel=share_conv_channel,
            num_heatmap_convs=num_heatmap_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias,
            norm_bbox=norm_bbox,
            init_cfg=init_cfg,
        )
        self.out_size_factor = train_cfg.out_size_factor

    def loss(self, batch, preds_dicts, **kwargs):
        gt_bboxes_3d = [item[0] for item in batch['gt_bboxes_3d']]
        gt_labels_3d = [item[0] for item in batch['gt_labels_3d']]
        loss_dict = super().loss(gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs)
        loss_dict = {key: loss_tensor.mean() for key, loss_tensor in loss_dict.items()}
        return loss_dict

    def get_bboxes(self, batch, preds_dicts):
        img_metas = [item[0] for item in batch['input_metas']]
        return super().get_bboxes(preds_dicts, img_metas)

    def get_heatmaps(self, batch, preds_dicts):
        gt_bboxes_3d = [item[0] for item in batch['gt_bboxes_3d']]
        gt_labels_3d = [item[0] for item in batch['gt_labels_3d']]
        gt_heatmaps, anno_boxes, inds, masks = self.get_targets(gt_bboxes_3d, gt_labels_3d)

        preds_heatmaps = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            pred_heatmaps = torch.clamp(torch.sigmoid(preds_dict[0]['heatmap'].detach()), 1e-4, 1 - 1e-4)
            preds_heatmaps[f'task_{task_id}.heatmap'] = pred_heatmaps
        return preds_heatmaps, gt_heatmaps

    def get_additional_tags(self) -> str:
        return f'osf_{self.out_size_factor}'
