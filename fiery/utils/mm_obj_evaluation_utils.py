import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.detection.data_classes import DetectionBox

# mmdet3d
from mmdet3d.core import Box3DMode
from mmdet3d.core.bbox import LiDARInstance3DBoxes
# from mmdet3d.core.bbox import get_box_type
from fiery.data import NUSCENE_CLASS_NAMES


def mm_bbox3d2result(bboxes, scores, labels, attrs=None, gt=False):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
        labels (torch.Tensor): Labels with shape of (n, ).
        scores (torch.Tensor): Scores with shape of (n, ).
        attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.

    """
    if gt is False:
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu())

        # result_dict = dict(
        #     boxes_3d=bboxes,
        #     scores_3d=scores,
        #     labels_3d=labels)
    else:
        result_dict = dict(
            boxes_3d=bboxes.to('cpu'),
            # scores_3d=scores.cpu(),
            labels_3d=labels.cpu())

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()

    return result_dict


def output_to_nusc_box(detection, token):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (0.0, 0.0, 0.0)

        # velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)

        box = NuScenesBox(
            center=box_gravity_center[i],
            size=box_dims[i],
            orientation=quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
            name=NUSCENE_CLASS_NAMES[labels[i]],
            token=token,
        )

        # box = DetectionBox(
        #     translation=box_gravity_center[i],
        #     size=box_dims[i],
        #     rotation=quat,
        #     # label=labels[i],
        #     detection_score=scores[i],
        #     velocity=velocity,
        #     detection_name=NUSCENE_CLASS_NAMES[labels[i]],
        #     sample_token=token,
        # )

        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
