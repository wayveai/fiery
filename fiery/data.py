import os
from PIL import Image

import numpy as np
import cv2
import torch
import torchvision

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from lyft_dataset_sdk.lyftdataset import LyftDataset

from fiery.utils.geometry import (
    resize_and_crop_image,
    update_intrinsics,
    calculate_birds_eye_view_parameters,
    convert_egopose_to_matrix_numpy,
    pose_vec2mat,
    mat2pose_vec,
    invert_matrix_egopose_numpy,
)
from fiery.utils.instance import convert_instance_mask_to_center_and_offset_label
from fiery.utils.lyft_splits import TRAIN_LYFT_INDICES, VAL_LYFT_INDICES

from collections import namedtuple
import random
# mmdet3d
from mmdet3d.core import Box3DMode
from mmdet3d.core.bbox import LiDARInstance3DBoxes
# from mmdet3d.core.bbox import get_box_type


general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    # "human.pedestrian.wheelchair": "ignore",
    # "human.pedestrian.stroller": "ignore",
    # "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    # "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    # "vehicle.emergency.ambulance": "ignore",
    # "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    # "movable_object.pushable_pullable": "ignore",
    # "movable_object.debris": "ignore",
    # "static_object.bicycle_rack": "ignore",
}

NUSCENE_CLASS_NAMES = [
    'car',
    'truck',
    'trailer',
    'bus',
    'construction_vehicle',
    'bicycle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'barrier',
    # "ignore",
]

ObjectData = namedtuple(
    'ObjectData',
    [
        'classname',
        'position',
        'dimensions',
        'angle',
        'score',
        'rec',
    ]
)


class FuturePredictionDataset(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, cfg):
        self.nusc = nusc
        self.is_train = is_train
        self.cfg = cfg

        self.is_lyft = isinstance(nusc, LyftDataset)

        if self.is_lyft:
            self.dataroot = self.nusc.data_path
        else:
            self.dataroot = self.nusc.dataroot

        self.mode = 'train' if self.is_train else 'val'

        self.sequence_length = cfg.TIME_RECEPTIVE_FIELD + cfg.N_FUTURE_FRAMES

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()
        self.indices = self.get_indices()

        # Image resizing and cropping
        self.augmentation_parameters = self.get_resizing_and_cropping_parameters()

        # Normalising input images
        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ]
        )

        # Bird's-eye view parameters
        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
        )
        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])

    def get_scenes(self):

        if self.is_lyft:
            scenes = [row['name'] for row in self.nusc.scene]

            # Split in train/val
            indices = TRAIN_LYFT_INDICES if self.is_train else VAL_LYFT_INDICES
            scenes = [scenes[i] for i in indices]
        else:
            # filter by scene split
            split = {'v1.0-trainval': {True: 'train', False: 'val'},
                     'v1.0-mini': {True: 'mini_train', False: 'mini_val'}, }[
                self.nusc.version
            ][self.is_train]

            scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))
        print(len(samples))
        return samples

    def get_indices(self):
        indices = []
        for index in range(len(self.ixes)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.sequence_length):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.ixes):
                    is_valid_data = False
                    break
                rec = self.ixes[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices)

    def get_resizing_and_cropping_parameters(self):
        original_height, original_width = self.cfg.IMAGE.ORIGINAL_HEIGHT, self.cfg.IMAGE.ORIGINAL_WIDTH
        final_height, final_width = self.cfg.IMAGE.FINAL_DIM

        if self.cfg.IMAGE.IMAGE_AUG:
            low, high = self.cfg.IMAGE.RANDOM_RESIZE_RANGE
            resize_scale = np.random.uniform(low=low, high=high)

            resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
            resized_width, resized_height = resize_dims
            crop_h = int(max(0, (resized_height - final_height)))
            crop_w = int(np.random.randint(0, (resized_width - final_width)))
        else:
            resize_scale = self.cfg.IMAGE.RESIZE_SCALE
            resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
            resized_width, resized_height = resize_dims
            crop_h = self.cfg.IMAGE.TOP_CROP
            crop_w = int(max(0, (resized_width - final_width) / 2))
            if resized_width != final_width:
                print('Zero padding left and right parts of the image.')
            if crop_h + final_height != resized_height:
                print('Zero padding bottom part of the image.')

        # Left, top, right, bottom crops.
        crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

        return {'scale_width': resize_scale,
                'scale_height': resize_scale,
                'resize_dims': resize_dims,
                'crop': crop,
                }

    def get_input_data(self, rec):
        """
        Parameters
        ----------
            rec: nuscenes identifier for a given timestamp

        Returns
        -------
            images: torch.Tensor<float> (N, 3, H, W)
            intrinsics: torch.Tensor<float> (3, 3)
            extrinsics: torch.Tensor(N, 4, 4)
        """
        images = []
        intrinsics = []
        extrinsics = []
        cameras = self.cfg.IMAGE.NAMES
        n_camera = self.cfg.IMAGE.N_CAMERA
        # The extrinsics we want are from the camera sensor to "flat egopose" as defined
        # https://github.com/nutonomy/nuscenes-devkit/blob/9b492f76df22943daf1dc991358d3d606314af27/python-sdk/nuscenes/nuscenes.py#L279
        # which corresponds to the position of the lidar.
        # This is because the labels are generated by projecting the 3D bounding box in this lidar's reference frame.

        # From lidar egopose to world.
        lidar_sample = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        lidar_pose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])
        yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
        lidar_rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        lidar_translation = np.array(lidar_pose['translation'])[:, None]
        lidar_to_world = np.vstack([
            np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
            np.array([0, 0, 0, 1])
        ])
        cams = random.sample(cameras, n_camera)
        # print("cams: ", cams)
        for cam in cams:
            camera_sample = self.nusc.get('sample_data', rec['data'][cam])

            # Transformation from world to egopose
            car_egopose = self.nusc.get('ego_pose', camera_sample['ego_pose_token'])
            egopose_rotation = Quaternion(car_egopose['rotation']).inverse
            egopose_translation = -np.array(car_egopose['translation'])[:, None]
            world_to_car_egopose = np.vstack([
                np.hstack((egopose_rotation.rotation_matrix, egopose_rotation.rotation_matrix @ egopose_translation)),
                np.array([0, 0, 0, 1])
            ])

            # From egopose to sensor
            sensor_sample = self.nusc.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])
            intrinsic = torch.Tensor(sensor_sample['camera_intrinsic'])
            sensor_rotation = Quaternion(sensor_sample['rotation'])
            sensor_translation = np.array(sensor_sample['translation'])[:, None]
            car_egopose_to_sensor = np.vstack([
                np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
                np.array([0, 0, 0, 1])
            ])
            car_egopose_to_sensor = np.linalg.inv(car_egopose_to_sensor)

            # Combine all the transformation.
            # From sensor to lidar.
            lidar_to_sensor = car_egopose_to_sensor @ world_to_car_egopose @ lidar_to_world
            sensor_to_lidar = torch.from_numpy(np.linalg.inv(lidar_to_sensor)).float()

            # Load image
            image_filename = os.path.join(self.dataroot, camera_sample['filename'])
            img = Image.open(image_filename)
            # Resize and crop
            img = resize_and_crop_image(
                img, resize_dims=self.augmentation_parameters['resize_dims'], crop=self.augmentation_parameters['crop']
            )
            # Normalise image
            normalised_img = self.normalise_image(img)

            # Combine resize/cropping in the intrinsics
            top_crop = self.augmentation_parameters['crop'][1]
            left_crop = self.augmentation_parameters['crop'][0]
            intrinsic = update_intrinsics(
                intrinsic, top_crop, left_crop,
                scale_width=self.augmentation_parameters['scale_width'],
                scale_height=self.augmentation_parameters['scale_height']
            )

            images.append(normalised_img.unsqueeze(0).unsqueeze(0))
            intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))
            extrinsics.append(sensor_to_lidar.unsqueeze(0).unsqueeze(0))

        images, intrinsics, extrinsics = (torch.cat(images, dim=1),
                                          torch.cat(intrinsics, dim=1),
                                          torch.cat(extrinsics, dim=1)
                                          )

        return images, intrinsics, extrinsics

    def _get_top_lidar_pose(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data',
                                              rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
        return trans, rot

    def _get_poly_region_in_image(self, instance_annotation, ego_translation, ego_rotation, token):
        box = Box(
            instance_annotation['translation'],
            instance_annotation['size'],
            Quaternion(instance_annotation['rotation']),
            name=instance_annotation['category_name'],
            token=token,
        )
        box.translate(ego_translation)
        box.rotate(ego_rotation)

        pts = box.bottom_corners()[:2].T
        pts = np.round(
            (pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]

        z = box.bottom_corners()[2, 0]

        return box, pts, z

    def get_birds_eye_view_label(self, rec, instance_map):
        translation, rotation = self._get_top_lidar_pose(rec)
        segmentation = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        # Background is ID 0
        instance = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        z_position = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        attribute_label = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        objects = list()
        boxes = []
        for annotation_token in rec['anns']:
            # Filter out all non vehicle instances -> no -> for all classes !!
            annotation = self.nusc.get('sample_annotation', annotation_token)

            if not self.is_lyft:
                # NuScenes filter

                # if self.cfg.LOSS.SEG_USE is True:
                #     if 'vehicle' not in annotation['category_name']:
                #         continue704×256

                if annotation['category_name'] not in general_to_detection:
                    continue
                if self.cfg.DATASET.FILTER_INVISIBLE_VEHICLES and int(annotation['visibility_token']) == 1:
                    continue
            else:
                # Lyft filter
                if annotation['category_name'] not in ['bus', 'car', 'construction_vehicle', 'trailer', 'truck']:
                    continue

            if annotation['instance_token'] not in instance_map:
                instance_map[annotation['instance_token']] = len(instance_map) + 1
            instance_id = instance_map[annotation['instance_token']]

            if not self.is_lyft:
                instance_attribute = int(annotation['visibility_token'])
            else:
                instance_attribute = 0

            box, poly_region, z = self._get_poly_region_in_image(annotation,
                                                                 translation,
                                                                 rotation,
                                                                 rec['token'],
                                                                 )
            if self.cfg.SEMANTIC_SEG.NUSCENE_CLASS:
                cv2.fillPoly(segmentation, [poly_region], NUSCENE_CLASS_NAMES.index(
                    general_to_detection[box.name]) + 1.0)
            else:
                cv2.fillPoly(segmentation, [poly_region], 1.0)

            cv2.fillPoly(instance, [poly_region], instance_id)
            cv2.fillPoly(z_position, [poly_region], z)
            cv2.fillPoly(attribute_label, [poly_region], instance_attribute)

            # print("instance: ", instance.shape)
            # print("segmentation: ", segmentation.shape)
            # print("z_position: ", z_position.shape)

            objects.append(
                ObjectData(
                    classname=general_to_detection[box.name],
                    dimensions=box.wlh,
                    position=box.center,
                    angle=box.orientation.radians,
                    score=1,
                    rec=rec['data']['LIDAR_TOP'],
                )
            )
            boxes.append(box)

        return segmentation, instance, z_position, instance_map, attribute_label, objects, boxes

    def _get_annos(self, boxes):
        # gt_bboxes_3d: [N, 7] or [N, 9]
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)  # [x, y, z]
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)  # [y_size, x_size, z_size]

        # Swap X, Y axis
        locs[:, [1, 0]] = locs[:, [0, 1]]

        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)

        # Revise Rotation angle for mirror line:  y = x
        gt_bboxes_3d_list = [locs, dims, -rots + np.pi / 2]

        if self.cfg.DATASET.INCLUDE_VELOCITY:
            gt_bboxes_3d_list.append(np.zeros((len(boxes), 2)))

        gt_bboxes_3d = np.concatenate(gt_bboxes_3d_list, axis=1)
        gt_bboxes_3d = torch.from_numpy(gt_bboxes_3d).float()

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(Box3DMode.LIDAR)

        # gt_names_3d: [N,]
        gt_names_3d = [b.name for b in boxes]
        for i in range(len(gt_names_3d)):
            if gt_names_3d[i] in general_to_detection:
                gt_names_3d[i] = general_to_detection[gt_names_3d[i]]
        # gt_names_3d = np.array(gt_names_3d)

        # gt_labels_3d: [N,]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in NUSCENE_CLASS_NAMES:
                gt_labels_3d.append(NUSCENE_CLASS_NAMES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        gt_labels_3d = torch.from_numpy(gt_labels_3d)

        # input_metas
        input_metas = dict(
            boxes_3d=gt_bboxes_3d,
            box_mode_3d=Box3DMode.LIDAR,
            box_type_3d=LiDARInstance3DBoxes,
        )
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names_3d=gt_names_3d,
            input_metas=input_metas,
        )

        return anns_results

    def get_label(self, rec, instance_map):
        segmentation_np, instance_np, z_position_np, instance_map, attribute_label_np, objects, boxes = \
            self.get_birds_eye_view_label(rec, instance_map)

        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0).unsqueeze(0)
        instance = torch.from_numpy(instance_np).long().unsqueeze(0)
        z_position = torch.from_numpy(z_position_np).float().unsqueeze(0).unsqueeze(0)
        attribute_label = torch.from_numpy(attribute_label_np).long().unsqueeze(0).unsqueeze(0)

        anns_results = self._get_annos(boxes)
        return segmentation, instance, z_position, instance_map, attribute_label, anns_results

    def get_future_egomotion(self, rec, index):
        rec_t0 = rec

        # Identity
        future_egomotion = np.eye(4, dtype=np.float32)

        if index < len(self.ixes) - 1:
            rec_t1 = self.ixes[index + 1]

            if rec_t0['scene_token'] == rec_t1['scene_token']:
                egopose_t0 = self.nusc.get(
                    'ego_pose', self.nusc.get('sample_data', rec_t0['data']['LIDAR_TOP'])['ego_pose_token']
                )
                egopose_t1 = self.nusc.get(
                    'ego_pose', self.nusc.get('sample_data', rec_t1['data']['LIDAR_TOP'])['ego_pose_token']
                )

                egopose_t0 = convert_egopose_to_matrix_numpy(egopose_t0)
                egopose_t1 = convert_egopose_to_matrix_numpy(egopose_t1)

                future_egomotion = invert_matrix_egopose_numpy(egopose_t1).dot(egopose_t0)
                future_egomotion[3, :3] = 0.0
                future_egomotion[3, 3] = 1.0

        future_egomotion = torch.Tensor(future_egomotion).float()

        # Convert to 6DoF vector
        future_egomotion = mat2pose_vec(future_egomotion)
        return future_egomotion.unsqueeze(0)

    def __len__(self):
        # return len(self.indices)
        return len(self.ixes)

    def __getitem__(self, index):
        """
        Returns
        -------
            data: dict with the following keys:
                image: torch.Tensor<float> (T, N, 3, H, W)
                    normalised cameras images with T the sequence length, and N the number of cameras.
                intrinsics: torch.Tensor<float> (T, N, 3, 3)
                    intrinsics containing resizing and cropping parameters.
                extrinsics: torch.Tensor<float> (T, N, 4, 4)
                    6 DoF pose from world coordinates to camera coordinates.
                segmentation: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                    (H_bev, W_bev) are the pixel dimensions in bird's-eye view.
                instance: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                centerness: torch.Tensor<float> (T, 1, H_bev, W_bev)
                offset: torch.Tensor<float> (T, 2, H_bev, W_bev)
                flow: torch.Tensor<float> (T, 2, H_bev, W_bev)
                future_egomotion: torch.Tensor<float> (T, 6)
                    6 DoF egomotion t -> t+1
                sample_token: List<str> (T,)
                'z_position': list_z_position,
                'attribute': list_attribute_label,


        """
        data = {}
        keys = ['image', 'intrinsics', 'extrinsics',
                'segmentation', 'instance', 'centerness', 'offset', 'flow', 'future_egomotion',
                'sample_token',
                'z_position', 'attribute',
                'gt_bboxes_3d', 'gt_labels_3d', 'gt_names_3d', 'input_metas',
                ]
        for key in keys:
            data[key] = []

        instance_map = {}
        # Loop over all the frames in the sequence.
        # for index_t in self.indices[index]:
        #     rec = self.ixes[index_t]
        for i in range(1):
            rec = self.ixes[index]

            images, intrinsics, extrinsics = self.get_input_data(rec)
            segmentation, instance, z_position, instance_map, attribute_label, anns_results = \
                self.get_label(rec, instance_map)

            # future_egomotion = self.get_future_egomotion(rec, index_t)
            future_egomotion = self.get_future_egomotion(rec, index)

            data['gt_bboxes_3d'].append(anns_results['gt_bboxes_3d'])
            data['gt_labels_3d'].append(anns_results['gt_labels_3d'])
            data['gt_names_3d'].append(anns_results['gt_names_3d'])
            data['input_metas'].append(anns_results['input_metas'])

            data['image'].append(images)
            data['intrinsics'].append(intrinsics)
            data['extrinsics'].append(extrinsics)

            data['segmentation'].append(segmentation)
            data['instance'].append(instance)
            data['future_egomotion'].append(future_egomotion)
            data['sample_token'].append(rec['token'])
            data['z_position'].append(z_position)
            data['attribute'].append(attribute_label)

        for key, value in data.items():
            if key in ['sample_token', 'centerness', 'offset', 'flow', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_names_3d', 'input_metas', ]:
                continue
            data[key] = torch.cat(value, dim=0)

        # If lyft need to subsample, and update future_egomotions
        if self.cfg.MODEL.SUBSAMPLE:
            for key, value in data.items():
                if key in ['future_egomotion', 'sample_token', 'centerness', 'offset', 'flow']:
                    continue
                data[key] = data[key][::2].clone()
            data['sample_token'] = data['sample_token'][::2]

            # Update future egomotions
            future_egomotions_matrix = pose_vec2mat(data['future_egomotion'])
            future_egomotion_accum = torch.zeros_like(future_egomotions_matrix)
            future_egomotion_accum[:-1] = future_egomotions_matrix[:-1] @ future_egomotions_matrix[1:]
            future_egomotion_accum = mat2pose_vec(future_egomotion_accum)
            data['future_egomotion'] = future_egomotion_accum[::2].clone()

        instance_centerness, instance_offset, instance_flow = convert_instance_mask_to_center_and_offset_label(
            data['instance'], data['future_egomotion'],
            num_instances=len(instance_map), ignore_index=self.cfg.DATASET.IGNORE_INDEX, subtract_egomotion=True,
            spatial_extent=self.spatial_extent,
        )
        data['centerness'] = instance_centerness
        data['offset'] = instance_offset
        data['flow'] = instance_flow
        return data


class DeviceDict(dict):
    def __init__(self, *args):
        super(DeviceDict, self).__init__(*args)

    def to(self, device):
        dd = DeviceDict()
        for k, v in self.items():
            if torch.is_tensor(v):
                dd[k] = v.to(device)
            else:
                dd[k] = v
        return dd


def collate_helper(elems, key):
    if key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_names_3d', 'input_metas', ]:
        return elems
    else:
        return torch.utils.data.dataloader.default_collate(elems)


def mm_collact_fn(batch):
    elem = batch[0]
    return DeviceDict({key: collate_helper([d[key] for d in batch], key) for key in elem})


def prepare_dataloaders(cfg, return_dataset=False):

    version = cfg.DATASET.VERSION
    train_on_training_data = True

    if cfg.DATASET.NAME == 'nuscenes':
        # 28130 train and 6019 val
        dataroot = os.path.join(cfg.DATASET.DATAROOT, version)
        nusc = NuScenes(version='{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=False)
    elif cfg.DATASET.NAME == 'lyft':
        # train contains 22680 samples
        # we split in 16506 6174
        dataroot = os.path.join(cfg.DATASET.DATAROOT, 'trainval')
        nusc = LyftDataset(data_path=dataroot,
                           json_path=os.path.join(dataroot, 'train_data'),
                           verbose=True)

    traindata = FuturePredictionDataset(nusc, train_on_training_data, cfg)
    valdata = FuturePredictionDataset(nusc, False, cfg)
    testdata = FuturePredictionDataset(nusc, False, cfg) if not cfg.TEST_TRAINSET else traindata

    # if cfg.DATASET.VERSION == 'v1.0-mini':
    #     traindata.indices = traindata.indices[:10]
    #     valdata.indices = valdata.indices[:10]

    if cfg.DATASET.TRAINING_SAMPLES != -1:
        traindata.ixes = traindata.ixes[:cfg.DATASET.TRAINING_SAMPLES]

    if cfg.DATASET.VALIDATING_SAMPLES != -1:
        valdata.ixes = valdata.ixes[:cfg.DATASET.VALIDATING_SAMPLES]

    print("traindata.__len__(): ", traindata.__len__())
    print("valdata.__len__(): ", valdata.__len__())
    print("testdata.__len__(): ", testdata.__len__())

    nworkers = cfg.N_WORKERS
    trainloader = torch.utils.data.DataLoader(
        traindata, batch_size=cfg.BATCHSIZE, shuffle=True, collate_fn=mm_collact_fn, num_workers=nworkers, pin_memory=True, drop_last=True
    )
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=cfg.VAL_BATCHSIZE, shuffle=False, collate_fn=mm_collact_fn, num_workers=nworkers, pin_memory=True, drop_last=False
    )
    testloader = torch.utils.data.DataLoader(
        testdata, batch_size=cfg.VAL_BATCHSIZE, shuffle=False, collate_fn=mm_collact_fn, num_workers=nworkers, pin_memory=True, drop_last=False
    )
    if return_dataset:
        return trainloader, valloader, testloader, traindata, valdata, testdata
    else:
        return trainloader, valloader, testloader
