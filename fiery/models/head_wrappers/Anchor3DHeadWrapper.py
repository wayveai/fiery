import torch
from mmdet3d.models.builder import HEADS
from mmdet3d.models.dense_heads import Anchor3DHead


@HEADS.register_module()
class Anchor3DHeadWrapper(Anchor3DHead):
    """Anchor head for SECOND/PointPillars/MVXNet/PartA2.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        train_cfg (dict): Train configs.
        test_cfg (dict): Test configs.
        feat_channels (int): Number of channels of the feature map.
        use_direction_classifier (bool): Whether to add a direction classifier.
        anchor_generator(dict): Config dict of anchor generator.
        assigner_per_size (bool): Whether to do assignment for each separate
            anchor size.
        assign_per_class (bool): Whether to do assignment for each class.
        diff_rad_by_sin (bool): Whether to change the difference into sin
            difference for box regression loss.
        dir_offset (float | int): The offset of BEV rotation angles.
            (TODO: may be moved into box coder)
        dir_limit_offset (float | int): The limited range of BEV
            rotation angles. (TODO: may be moved into box coder)
        bbox_coder (dict): Config dict of box coders.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_dir (dict): Config of direction classifier loss.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 train_cfg,
                 test_cfg,
                 feat_channels=256,
                 use_direction_classifier=True,
                 anchor_generator=dict(
                     type='Anchor3DRangeGenerator',
                     range=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
                     strides=[2],
                     sizes=[[1.6, 3.9, 1.56]],
                     rotations=[0, 1.57],
                     custom_values=[],
                     reshape_out=False),
                 assigner_per_size=False,
                 assign_per_class=False,
                 diff_rad_by_sin=True,
                 dir_offset=0,
                 dir_limit_offset=1,
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_dir=dict(type='CrossEntropyLoss', loss_weight=0.2),
                 init_cfg=None):
        super().__init__(
            num_classes,
            in_channels,
            train_cfg,
            test_cfg,
            feat_channels=feat_channels,
            use_direction_classifier=use_direction_classifier,
            anchor_generator=anchor_generator,
            assigner_per_size=assigner_per_size,
            assign_per_class=assign_per_class,
            diff_rad_by_sin=diff_rad_by_sin,
            dir_offset=dir_offset,
            dir_limit_offset=dir_limit_offset,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dir=loss_dir,
            init_cfg=init_cfg,
        )

    def loss(self, batch, preds_dicts):
        gt_bboxes = [item[-1] for item in batch['gt_bboxes_3d']]
        # print("gt_bboxes: ", gt_bboxes)

        gt_labels = [item[-1] for item in batch['gt_labels_3d']]
        input_metas = [item[-1] for item in batch['input_metas']]
        preds_dicts = preds_dicts + (gt_bboxes, gt_labels, input_metas,)
        loss_dict = super().loss(*preds_dicts)
        loss_dict = {key: torch.stack(loss_value_list).mean() for key, loss_value_list in loss_dict.items()}
        return loss_dict

    def get_bboxes(self, batch, preds_dicts):
        img_metas = [item[-1] for item in batch['input_metas']]
        return super().get_bboxes(*preds_dicts, img_metas)
