import os
from PIL import Image

import numpy as np
import cv2
import torch
import pytorch_lightning as pl
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

        self.scenes = frozenset(self.get_scenes())
        self.prepro()
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
                     'v1.0-mini': {True: 'mini_train', False: 'mini_val'},}[
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

        n = len(samples)
        sensor_names = ['LIDAR_TOP'] + list(self.cfg.IMAGE.NAMES)
        self._sensor_col = {name: i for i, name in enumerate(sensor_names)}

        self._scene_tokens = np.empty(n, dtype='<U64')
        self._tokens = np.empty(n, dtype='<U64')
        self._data_tokens = np.empty((n, len(sensor_names)), dtype='<U64')

        all_anns = []
        anns_offsets = np.empty(n + 1, dtype=np.int64)
        anns_offsets[0] = 0

        for i, samp in enumerate(samples):
            self._scene_tokens[i] = samp['scene_token']
            self._tokens[i] = samp['token']
            for j, sensor in enumerate(sensor_names):
                self._data_tokens[i, j] = samp['data'].get(sensor, '')
            ann_list = samp['anns']
            all_anns.extend(ann_list)
            anns_offsets[i + 1] = anns_offsets[i] + len(ann_list)

        self._anns_flat = np.array(all_anns, dtype='<U64') if all_anns else np.empty(0, dtype='<U64')
        self._anns_offsets = anns_offsets

        # Pre-cache all NuScenes lookup data into plain numpy/dict structures
        # so we can release self.nusc and avoid duplicating it in DataLoader workers.
        self._cache_nusc_lookups()
        self.nusc = None

    def _cache_nusc_lookups(self):
        """Extract all data needed by __getitem__ from self.nusc into numpy-friendly caches."""
        # Collect all sample_data tokens referenced by this split
        all_sd_tokens = set()
        for i in range(len(self._tokens)):
            for j in range(self._data_tokens.shape[1]):
                tok = str(self._data_tokens[i, j])
                if tok:
                    all_sd_tokens.add(tok)
        # Also need next-frame sample_data for future_egomotion lookups
        # (index_t + 1 may reference tokens already in _data_tokens)

        # Cache sample_data: token -> {ego_pose_token, calibrated_sensor_token, filename}
        self._sd_cache = {}
        for tok in all_sd_tokens:
            rec = self.nusc.get('sample_data', tok)
            self._sd_cache[tok] = {
                'ego_pose_token': rec['ego_pose_token'],
                'calibrated_sensor_token': rec['calibrated_sensor_token'],
                'filename': rec['filename'],
            }

        # Cache ego_pose: token -> {translation (3,), rotation (4,)}
        ego_tokens = set(v['ego_pose_token'] for v in self._sd_cache.values())
        self._ego_cache = {}
        for tok in ego_tokens:
            rec = self.nusc.get('ego_pose', tok)
            self._ego_cache[tok] = {
                'translation': np.array(rec['translation'], dtype=np.float64),
                'rotation': np.array(rec['rotation'], dtype=np.float64),
            }

        # Cache calibrated_sensor: token -> {translation (3,), rotation (4,), camera_intrinsic (3,3)}
        cs_tokens = set(v['calibrated_sensor_token'] for v in self._sd_cache.values())
        self._cs_cache = {}
        for tok in cs_tokens:
            rec = self.nusc.get('calibrated_sensor', tok)
            self._cs_cache[tok] = {
                'translation': np.array(rec['translation'], dtype=np.float64),
                'rotation': np.array(rec['rotation'], dtype=np.float64),
                'camera_intrinsic': np.array(rec['camera_intrinsic'], dtype=np.float64) if rec['camera_intrinsic'] else np.zeros((3, 3), dtype=np.float64),
            }

        # Cache sample_annotation: token -> {category_name, visibility_token, instance_token, translation, size, rotation}
        self._ann_cache = {}
        for ann_tok in self._anns_flat:
            ann_tok_str = str(ann_tok)
            if ann_tok_str in self._ann_cache:
                continue
            rec = self.nusc.get('sample_annotation', ann_tok_str)
            self._ann_cache[ann_tok_str] = {
                'category_name': rec['category_name'],
                'visibility_token': rec['visibility_token'],
                'instance_token': rec['instance_token'],
                'translation': rec['translation'],
                'size': rec['size'],
                'rotation': rec['rotation'],
            }

    def get_indices(self):
        indices = []
        n_samples = len(self._scene_tokens)
        for index in range(n_samples):
            is_valid_data = True
            previous_scene_token = None
            current_indices = []
            for t in range(self.sequence_length):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= n_samples:
                    is_valid_data = False
                    break
                scene_token = self._scene_tokens[index_t]
                # Check if scene is the same
                if (previous_scene_token is not None) and (scene_token != previous_scene_token):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_scene_token = scene_token

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices)

    def get_resizing_and_cropping_parameters(self):
        original_height, original_width = self.cfg.IMAGE.ORIGINAL_HEIGHT, self.cfg.IMAGE.ORIGINAL_WIDTH
        final_height, final_width = self.cfg.IMAGE.FINAL_DIM

        resize_scale = self.cfg.IMAGE.RESIZE_SCALE
        resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
        resized_width, resized_height = resize_dims

        crop_h = self.cfg.IMAGE.TOP_CROP
        crop_w = int(max(0, (resized_width - final_width) / 2))
        # Left, top, right, bottom crops.
        crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

        if resized_width != final_width:
            print('Zero padding left and right parts of the image.')
        if crop_h + final_height != resized_height:
            print('Zero padding bottom part of the image.')

        return {'scale_width': resize_scale,
                'scale_height': resize_scale,
                'resize_dims': resize_dims,
                'crop': crop,
                }

    def get_input_data(self, index_t):
        """
        Parameters
        ----------
            index_t: int, index into the structured numpy arrays

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

        # The extrinsics we want are from the camera sensor to "flat egopose" as defined
        # https://github.com/nutonomy/nuscenes-devkit/blob/9b492f76df22943daf1dc991358d3d606314af27/python-sdk/nuscenes/nuscenes.py#L279
        # which corresponds to the position of the lidar.
        # This is because the labels are generated by projecting the 3D bounding box in this lidar's reference frame.

        # From lidar egopose to world.
        lidar_token = str(self._data_tokens[index_t, self._sensor_col['LIDAR_TOP']])
        lidar_sample = self._sd_cache[lidar_token]
        lidar_pose = self._ego_cache[lidar_sample['ego_pose_token']]
        yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
        lidar_rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        lidar_translation = np.array(lidar_pose['translation'])[:, None]
        lidar_to_world = np.vstack([
            np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
            np.array([0, 0, 0, 1])
        ])

        for cam in cameras:
            camera_sample = self._sd_cache[str(self._data_tokens[index_t, self._sensor_col[cam]])]

            # Transformation from world to egopose
            car_egopose = self._ego_cache[camera_sample['ego_pose_token']]
            egopose_rotation = Quaternion(car_egopose['rotation']).inverse
            egopose_translation = -np.array(car_egopose['translation'])[:, None]
            world_to_car_egopose = np.vstack([
                np.hstack((egopose_rotation.rotation_matrix, egopose_rotation.rotation_matrix @ egopose_translation)),
                np.array([0, 0, 0, 1])
            ])

            # From egopose to sensor
            sensor_sample = self._cs_cache[camera_sample['calibrated_sensor_token']]
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

    def _get_top_lidar_pose(self, index_t):
        lidar_token = str(self._data_tokens[index_t, self._sensor_col['LIDAR_TOP']])
        egopose = self._ego_cache[self._sd_cache[lidar_token]['ego_pose_token']]
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
        return trans, rot

    def get_birds_eye_view_label(self, index_t, instance_map):
        translation, rotation = self._get_top_lidar_pose(index_t)
        segmentation = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        # Background is ID 0
        instance = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        z_position = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        attribute_label = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))

        ann_start = self._anns_offsets[index_t]
        ann_end = self._anns_offsets[index_t + 1]
        for annotation_token in self._anns_flat[ann_start:ann_end]:
            # Filter out all non vehicle instances
            annotation = self._ann_cache[str(annotation_token)]

            if not self.is_lyft:
                # NuScenes filter
                if 'vehicle' not in annotation['category_name']:
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

            poly_region, z = self._get_poly_region_in_image(annotation, translation, rotation)
            cv2.fillPoly(instance, [poly_region], instance_id)
            cv2.fillPoly(segmentation, [poly_region], 1.0)
            cv2.fillPoly(z_position, [poly_region], z)
            cv2.fillPoly(attribute_label, [poly_region], instance_attribute)

        return segmentation, instance, z_position, instance_map, attribute_label

    def _get_poly_region_in_image(self, instance_annotation, ego_translation, ego_rotation):
        box = Box(
            instance_annotation['translation'], instance_annotation['size'], Quaternion(instance_annotation['rotation'])
        )
        box.translate(ego_translation)
        box.rotate(ego_rotation)

        pts = box.bottom_corners()[:2].T
        pts = np.round((pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]

        z = box.bottom_corners()[2, 0]
        return pts, z

    def get_label(self, index_t, instance_map):
        segmentation_np, instance_np, z_position_np, instance_map, attribute_label_np = \
            self.get_birds_eye_view_label(index_t, instance_map)
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0).unsqueeze(0)
        instance = torch.from_numpy(instance_np).long().unsqueeze(0)
        z_position = torch.from_numpy(z_position_np).float().unsqueeze(0).unsqueeze(0)
        attribute_label = torch.from_numpy(attribute_label_np).long().unsqueeze(0).unsqueeze(0)

        return segmentation, instance, z_position, instance_map, attribute_label

    def get_future_egomotion(self, index_t):
        # Identity
        future_egomotion = np.eye(4, dtype=np.float32)

        if index_t < len(self._scene_tokens) - 1:
            if self._scene_tokens[index_t] == self._scene_tokens[index_t + 1]:
                lidar_col = self._sensor_col['LIDAR_TOP']
                egopose_t0 = self._ego_cache[
                    self._sd_cache[str(self._data_tokens[index_t, lidar_col])]['ego_pose_token']
                ]
                egopose_t1 = self._ego_cache[
                    self._sd_cache[str(self._data_tokens[index_t + 1, lidar_col])]['ego_pose_token']
                ]

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
        return len(self.indices)

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
                'z_position', 'attribute'
                ]
        for key in keys:
            data[key] = []

        instance_map = {}
        # Loop over all the frames in the sequence.
        for index_t in self.indices[index]:
            images, intrinsics, extrinsics = self.get_input_data(index_t)
            segmentation, instance, z_position, instance_map, attribute_label = self.get_label(index_t, instance_map)

            future_egomotion = self.get_future_egomotion(index_t)

            data['image'].append(images)
            data['intrinsics'].append(intrinsics)
            data['extrinsics'].append(extrinsics)
            data['segmentation'].append(segmentation)
            data['instance'].append(instance)
            data['future_egomotion'].append(future_egomotion)
            data['sample_token'].append(str(self._tokens[index_t]))
            data['z_position'].append(z_position)
            data['attribute'].append(attribute_label)

        for key, value in data.items():
            if key in ['sample_token', 'centerness', 'offset', 'flow']:
                continue
            data[key] = torch.cat(value, dim=0)

        # If lyft need to subsample, and update future_egomotions
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


def prepare_dataloaders(cfg, return_dataset=False):
    version = cfg.DATASET.VERSION
    train_on_training_data = True

    if cfg.DATASET.NAME == 'nuscenes':
        # 28130 train and 6019 val
        dataroot = os.path.join(cfg.DATASET.DATAROOT, version)
        nusc = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=False)
    elif cfg.DATASET.NAME == 'lyft':
        # train contains 22680 samples
        # we split in 16506 6174
        dataroot = os.path.join(cfg.DATASET.DATAROOT, 'trainval')
        nusc = LyftDataset(data_path=dataroot,
                           json_path=os.path.join(dataroot, 'train_data'),
                           verbose=True)

    traindata = FuturePredictionDataset(nusc, train_on_training_data, cfg)
    valdata = FuturePredictionDataset(nusc, False, cfg)

    if cfg.DATASET.VERSION == 'mini':
        traindata.indices = traindata.indices[:10]
        valdata.indices = valdata.indices[:10]

    nworkers = cfg.N_WORKERS
    trainloader = torch.utils.data.DataLoader(
        traindata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=nworkers, pin_memory=True, drop_last=True,
        persistent_workers=(nworkers > 0), prefetch_factor=2 if nworkers > 0 else None,
    )
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False,
        persistent_workers=(nworkers > 0), prefetch_factor=2 if nworkers > 0 else None,
    )

    if return_dataset:
        return trainloader, valloader, traindata, valdata
    else:
        return trainloader, valloader


class FieryDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._train_loader = None
        self._val_loader = None

    def setup(self, stage=None):
        if self._train_loader is None or self._val_loader is None:
            self._train_loader, self._val_loader = prepare_dataloaders(self.cfg)

    def train_dataloader(self):
        if self._train_loader is None:
            self.setup()
        return self._train_loader

    def val_dataloader(self):
        if self._val_loader is None:
            self.setup()
        return self._val_loader
