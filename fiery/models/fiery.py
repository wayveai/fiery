import torch
import torch.nn as nn
# import random

from fiery.models.encoder import Encoder
from fiery.models.temporal_model import TemporalModelIdentity, TemporalModel
from fiery.models.distributions import DistributionModule
from fiery.models.future_prediction import FuturePrediction
from fiery.models.decoder import Decoder
from fiery.utils.network import pack_sequence_dim, unpack_sequence_dim, set_bn_momentum
from fiery.utils.geometry import cumulative_warp_features, calculate_birds_eye_view_parameters, VoxelsSumming
from fiery.models.head_wrappers.CenterHeadWrapper import CenterHeadWrapper
from fiery.models.head_wrappers.Anchor3DHeadWrapper import Anchor3DHeadWrapper
from mmdet3d.models import build_head, build_backbone, build_neck


class Fiery(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND
        )
        self.bev_resolution = nn.Parameter(bev_resolution, requires_grad=False)
        self.bev_start_position = nn.Parameter(bev_start_position, requires_grad=False)
        self.bev_dimension = nn.Parameter(bev_dimension, requires_grad=False)

        self.encoder_downsample = self.cfg.MODEL.ENCODER.DOWNSAMPLE
        self.encoder_out_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS

        self.frustum = self.create_frustum()
        self.depth_channels, _, _, _ = self.frustum.shape

        if self.cfg.TIME_RECEPTIVE_FIELD == 1:
            assert self.cfg.MODEL.TEMPORAL_MODEL.NAME == 'identity'

        # temporal block
        self.receptive_field = self.cfg.TIME_RECEPTIVE_FIELD
        self.n_future = self.cfg.N_FUTURE_FRAMES
        self.latent_dim = self.cfg.MODEL.DISTRIBUTION.LATENT_DIM

        if self.cfg.MODEL.SUBSAMPLE:
            assert self.cfg.DATASET.NAME == 'lyft'
            self.receptive_field = 3
            self.n_future = 5

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])
        self.bev_size = (self.bev_dimension[0].item(), self.bev_dimension[1].item())

        # Encoder
        self.encoder = Encoder(cfg=self.cfg.MODEL.ENCODER, D=self.depth_channels)

        # Temporal model
        temporal_in_channels = self.encoder_out_channels
        if self.cfg.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE:
            temporal_in_channels += 6
        if self.cfg.MODEL.TEMPORAL_MODEL.NAME == 'identity':
            self.temporal_model = TemporalModelIdentity(temporal_in_channels, self.receptive_field)
        elif cfg.MODEL.TEMPORAL_MODEL.NAME == 'temporal_block':
            self.temporal_model = TemporalModel(
                temporal_in_channels,
                self.receptive_field,
                input_shape=self.bev_size,
                start_out_channels=self.cfg.MODEL.TEMPORAL_MODEL.START_OUT_CHANNELS,
                extra_in_channels=self.cfg.MODEL.TEMPORAL_MODEL.EXTRA_IN_CHANNELS,
                n_spatial_layers_between_temporal_layers=self.cfg.MODEL.TEMPORAL_MODEL.INBETWEEN_LAYERS,
                use_pyramid_pooling=self.cfg.MODEL.TEMPORAL_MODEL.PYRAMID_POOLING,
            )
        else:
            raise NotImplementedError(f'Temporal module {self.cfg.MODEL.TEMPORAL_MODEL.NAME}.')

        self.future_pred_in_channels = self.temporal_model.out_channels
        if self.n_future > 0:
            # probabilistic sampling
            if self.cfg.PROBABILISTIC.ENABLED:
                # Distribution networks
                self.present_distribution = DistributionModule(
                    self.future_pred_in_channels,
                    self.latent_dim,
                    min_log_sigma=self.cfg.MODEL.DISTRIBUTION.MIN_LOG_SIGMA,
                    max_log_sigma=self.cfg.MODEL.DISTRIBUTION.MAX_LOG_SIGMA,
                )

                future_distribution_in_channels = (self.future_pred_in_channels
                                                   + self.n_future * self.cfg.PROBABILISTIC.FUTURE_DIM
                                                   )
                self.future_distribution = DistributionModule(
                    future_distribution_in_channels,
                    self.latent_dim,
                    min_log_sigma=self.cfg.MODEL.DISTRIBUTION.MIN_LOG_SIGMA,
                    max_log_sigma=self.cfg.MODEL.DISTRIBUTION.MAX_LOG_SIGMA,
                )

            # Future prediction
            self.future_prediction = FuturePrediction(
                in_channels=self.future_pred_in_channels,
                latent_dim=self.latent_dim,
                n_gru_blocks=self.cfg.MODEL.FUTURE_PRED.N_GRU_BLOCKS,
                n_res_layers=self.cfg.MODEL.FUTURE_PRED.N_RES_LAYERS,
            )
        if self.cfg.SEMANTIC_SEG.NUSCENE_CLASS:
            SEMANTIC_SEG_WEIGHTS = [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        else:
            SEMANTIC_SEG_WEIGHTS = [1.0, 2.0]

        if self.cfg.LOSS.SEG_USE:
            print("Use segmentation loss to regress.")

            if self.cfg.MODEL.MM.SEG_CAT_BACKBONE:
                print("seg encoded_bev cat backbone.")
            if self.cfg.MODEL.MM.SEG_ADD_BACKBONE:
                print("seg encoded_bev add backbone.")
                self.seg_add_backbone_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
            # Decoder
            self.decoder = Decoder(
                in_channels=self.future_pred_in_channels,
                # n_classes=len(self.cfg.SEMANTIC_SEG.WEIGHTS),
                n_classes=len(SEMANTIC_SEG_WEIGHTS),
                predict_future_flow=self.cfg.INSTANCE_FLOW.ENABLED,
            )

        self.detection_backbone = build_backbone(self.cfg.MODEL.MM.BBOX_BACKBONE)
        self.detection_neck = build_neck(self.cfg.MODEL.MM.BBOX_NECK)
        self.detection_head = build_head(self.cfg.MODEL.MM.BBOX_HEAD)

        set_bn_momentum(self, self.cfg.MODEL.BN_MOMENTUM)

    def create_frustum(self):
        # Create grid in image plane
        h, w = self.cfg.IMAGE.FINAL_DIM

        downsampled_h, downsampled_w = h // self.encoder_downsample, w // self.encoder_downsample

        # Depth grid
        depth_grid = torch.arange(*self.cfg.LIFT.D_BOUND, dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, downsampled_h, downsampled_w)
        n_depth_slices = depth_grid.shape[0]

        # x and y grids
        x_grid = torch.linspace(0, w - 1, downsampled_w, dtype=torch.float)
        x_grid = x_grid.view(1, 1, downsampled_w).expand(n_depth_slices, downsampled_h, downsampled_w)
        y_grid = torch.linspace(0, h - 1, downsampled_h, dtype=torch.float)
        y_grid = y_grid.view(1, downsampled_h, 1).expand(n_depth_slices, downsampled_h, downsampled_w)

        # Dimension (n_depth_slices, downsampled_h, downsampled_w, 3)
        # containing data points in the image: left-right, top-bottom, depth
        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def forward(self, image, intrinsics, extrinsics, future_egomotion, future_distribution_inputs=None, noise=None):
        output = {}

        # Only process features from the past and present
        image = image[:, :self.receptive_field].contiguous()
        intrinsics = intrinsics[:, :self.receptive_field].contiguous()
        extrinsics = extrinsics[:, :self.receptive_field].contiguous()
        future_egomotion = future_egomotion[:, :self.receptive_field].contiguous()

        # Lifting features and project to bird's-eye view
        x = self.calculate_birds_eye_view_features(image, intrinsics, extrinsics)

        # Warp past features to the present's reference frame
        x = cumulative_warp_features(
            x.clone(), future_egomotion,
            mode='bilinear', spatial_extent=self.spatial_extent,
        )

        if self.cfg.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE:
            b, s, c = future_egomotion.shape
            h, w = x.shape[-2:]
            future_egomotions_spatial = future_egomotion.view(b, s, c, 1, 1).expand(b, s, c, h, w)
            # at time 0, no egomotion so feed zero vector
            future_egomotions_spatial = torch.cat([torch.zeros_like(future_egomotions_spatial[:, :1]),
                                                   future_egomotions_spatial[:, :(self.receptive_field - 1)]],
                                                  dim=1)
            x = torch.cat([x, future_egomotions_spatial], dim=-3)

        #  Temporal model
        states = self.temporal_model(x)

        if self.n_future > 0:
            # Split into present and future features (for the probabilistic model)
            present_state = states[:, :1].contiguous()
            if self.cfg.PROBABILISTIC.ENABLED:
                # Do probabilistic computation
                sample, output_distribution = self.distribution_forward(
                    present_state, future_distribution_inputs, noise
                )
                output = {**output, **output_distribution}

            # Prepare future prediction input
            b, _, _, h, w = present_state.shape
            hidden_state = present_state[:, 0]

            if self.cfg.PROBABILISTIC.ENABLED:
                future_prediction_input = sample.expand(-1, self.n_future, -1, -1, -1)
            else:
                future_prediction_input = hidden_state.new_zeros(b, self.n_future, self.latent_dim, h, w)

            # Recursively predict future states
            future_states = self.future_prediction(future_prediction_input, hidden_state)

            # Concatenate present state
            future_states = torch.cat([present_state, future_states], dim=1)

        # Predict bird's-eye view outputs
        if self.n_future > 0:
            decoder_input = future_states
            detection_input = future_states.flatten(0, 1)
            # cls_scores, bbox_preds, dir_cls_preds = self.detection_head([future_states])
        else:
            decoder_input = states[:, -1:]
            detection_input = states[:, -1:].flatten(0, 1)

        bev_output = {}
        if self.cfg.LOSS.SEG_USE:
            bev_output = self.decoder(decoder_input)

            if self.cfg.MODEL.MM.SEG_ADD_BACKBONE:
                detection_input = detection_input + self.seg_add_backbone_weight * bev_output['decoded_bev']

            if self.cfg.MODEL.MM.SEG_CAT_BACKBONE:
                detection_input = torch.cat([detection_input, bev_output['decoded_bev']], dim=-3)

        detection_backbone_output = self.detection_backbone(detection_input)
        detection_neck_output = self.detection_neck(detection_backbone_output)
        detection_output = self.detection_head(detection_neck_output)
        # print([[d.keys() for d in ld] for ld in detection_output])
        output = {'detection_output': detection_output, **output, **bev_output}

        return output

    def get_geometry(self, intrinsics, extrinsics):
        """Calculate the (x, y, z) 3D position of the features.
        """
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        B, N, _ = translation.shape
        # Add batch, camera dimension, and a dummy dimension at the end
        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # print("frustum.shape: ", self.frustum.shape)
        # print("point.shape: ", points.shape)

        # Camera to ego reference frame
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combined_transformation = rotation.matmul(torch.inverse(intrinsics))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(B, N, 1, 1, 1, 3)
        # print("transform point.shape: ", pointss.shape)

        # The 3 dimensions in the ego reference frame are: (forward, sides, height)
        return points

    def encoder_forward(self, x):
        # batch, n_cameras, channels, height, width
        # r = random.randint(0, 5)
        # x = x[:, r, :, :, :].unsqueeze(1)
        # print("x.shape: ", x.shape)
        b, n, c, h, w = x.shape

        x = x.view(b * n, c, h, w)
        x = self.encoder(x)
        # print("x.shape: ", x.shape)

        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2)
        # print("x.shape: ", x.shape)

        return x

    def projection_to_birds_eye_view(self, x, geometry):
        """ Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/models.py#L200"""
        # batch, n_cameras, depth, height, width, channels
        batch, n, d, h, w, c = x.shape
        output = torch.zeros(
            (batch, c, self.bev_dimension[0], self.bev_dimension[1]), dtype=torch.float, device=x.device
        )
        # centers = torch.zeros(
        #     (batch, c, self.bev_dimension[0] - 1, self.bev_dimension[1] - 1), dtype=torch.float, device=x.device
        # )
        # print("geometry.shape: ", geometry.shape)
        # Number of 3D points
        N = n * d * h * w
        for b in range(batch):
            # flatten x
            x_b = x[b].reshape(N, c)

            # Convert positions to integer indices
            geometry_b = ((geometry[b] - (self.bev_start_position - self.bev_resolution / 2.0)) / self.bev_resolution)
            # print("geometry.shape: ", geometry.shape)
            # print("geometry: ", geometry)
            # print("geometry_b[0][0].shape: ", geometry_b[0][0].shape)
            # print("geometry_b: ", geometry_b[0][0])

            geometry_b = geometry_b.view(N, 3).long()

            # print("(self.bev_start_position - self.bev_resolution / 2.0): ",
            #       (self.bev_start_position - self.bev_resolution / 2.0))

            # Mask out points that are outside the considered spatial extent.
            mask = (
                (geometry_b[:, 0] >= 0)
                & (geometry_b[:, 0] < self.bev_dimension[0])
                & (geometry_b[:, 1] >= 0)
                & (geometry_b[:, 1] < self.bev_dimension[1])
                & (geometry_b[:, 2] >= 0)
                & (geometry_b[:, 2] < self.bev_dimension[2])
            )
            x_b = x_b[mask]
            geometry_b = geometry_b[mask]

            # Sort tensors so that those within the same voxel are consecutives.
            ranks = (
                geometry_b[:, 0] * (self.bev_dimension[1] * self.bev_dimension[2])
                + geometry_b[:, 1] * (self.bev_dimension[2])
                + geometry_b[:, 2]
            )
            ranks_indices = ranks.argsort()
            x_b, geometry_b, ranks = x_b[ranks_indices], geometry_b[ranks_indices], ranks[ranks_indices]

            # Project to bird's-eye view by summing voxels.
            x_b, geometry_b = VoxelsSumming.apply(x_b, geometry_b, ranks)

            bev_feature = torch.zeros((self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1], c),
                                      device=x_b.device)
            bev_feature[geometry_b[:, 2], geometry_b[:, 0], geometry_b[:, 1]] = x_b

            # Put channel in second position and remove z dimension
            bev_feature = bev_feature.permute((0, 3, 1, 2))
            bev_feature = bev_feature.squeeze(0)

            output[b] = bev_feature
            # centers[b] = (bev_feature[:, 1:, 1:] + bev_feature[:, :-1, :-1]) / 2.0
            # print("centers[b].shape: ", centers[b].shape)
            # print("bev_feature.shape: ", bev_feature.shape)
            # print("output: ", output)

        return output

    def calculate_birds_eye_view_features(self, x, intrinsics, extrinsics):
        b, s, n, c, h, w = x.shape
        # Reshape
        x = pack_sequence_dim(x)
        intrinsics = pack_sequence_dim(intrinsics)
        extrinsics = pack_sequence_dim(extrinsics)

        geometry = self.get_geometry(intrinsics, extrinsics)
        x = self.encoder_forward(x)
        x = self.projection_to_birds_eye_view(x, geometry)
        x = unpack_sequence_dim(x, b, s)
        return x

    def distribution_forward(self, present_features, future_distribution_inputs=None, noise=None):
        """
        Parameters
        ----------
            present_features: 5-D output from dynamics module with shape (b, 1, c, h, w)
            future_distribution_inputs: 5-D tensor containing labels shape (b, s, cfg.PROB_FUTURE_DIM, h, w)
            noise: a sample from a (0, 1) gaussian with shape (b, s, latent_dim). If None, will sample in function

        Returns


        -------
            sample: sample taken from present/future distribution, broadcast to shape (b, s, latent_dim, h, w)
            present_distribution_mu: shape (b, s, latent_dim)
            present_distribution_log_sigma: shape (b, s, latent_dim)
            future_distribution_mu: shape (b, s, latent_dim)
            future_distribution_log_sigma: shape (b, s, latent_dim)
        """
        b, s, _, h, w = present_features.size()
        assert s == 1

        present_mu, present_log_sigma = self.present_distribution(present_features)

        future_mu, future_log_sigma = None, None
        if future_distribution_inputs is not None:
            # Concatenate future labels to z_t
            future_features = future_distribution_inputs[:, 1:].contiguous().view(b, 1, -1, h, w)
            future_features = torch.cat([present_features, future_features], dim=2)
            future_mu, future_log_sigma = self.future_distribution(future_features)

        if noise is None:
            if self.training:
                noise = torch.randn_like(present_mu)
            else:
                noise = torch.zeros_like(present_mu)
        if self.training:
            mu = future_mu
            sigma = torch.exp(future_log_sigma)
        else:
            mu = present_mu
            sigma = torch.exp(present_log_sigma)
        sample = mu + sigma * noise

        # Spatially broadcast sample to the dimensions of present_features
        sample = sample.view(b, s, self.latent_dim, 1, 1).expand(b, s, self.latent_dim, h, w)

        output_distribution = {
            'present_mu': present_mu,
            'present_log_sigma': present_log_sigma,
            'future_mu': future_mu,
            'future_log_sigma': future_log_sigma,
        }

        return sample, output_distribution
