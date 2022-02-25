import PIL
import numpy as np
import torch

from pyquaternion import Quaternion


def make_grid(grid_size, grid_offset, grid_res):
    """
        Constructs an array representing the corners of an orthographic grid.
    """
    x_length, y_length = grid_size
    x_offset, y_offset, z_offset = grid_offset

    xcoords = torch.arange(0., x_length + grid_res, grid_res) + x_offset
    ycoords = torch.arange(0., y_length + grid_res, grid_res) + y_offset

    yy, xx = torch.meshgrid(ycoords, xcoords)
    return torch.stack([xx, yy, torch.full_like(xx, z_offset)], dim=-1)


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                 dtype=torch.long)

    # print("bev_resolution: ", bev_resolution)
    # print("bev_start_position: ", bev_start_position)
    # print("bev_dimension: ", bev_dimension)

    return bev_resolution, bev_start_position, bev_dimension


def resize_and_crop_image(img, resize_dims, crop):
    # Bilinear resizing followed by cropping
    img = img.resize(resize_dims, resample=PIL.Image.BILINEAR)
    img = img.crop(crop)
    return img


def update_intrinsics(intrinsics, top_crop=0.0, left_crop=0.0, scale_width=1.0, scale_height=1.0):
    """
    Parameters
    ----------
        intrinsics: torch.Tensor (3, 3)
        top_crop: float
        left_crop: float
        scale_width: float
        scale_height: float
    """
    updated_intrinsics = intrinsics.clone()
    # Adjust intrinsics scale due to resizing
    updated_intrinsics[0, 0] *= scale_width
    updated_intrinsics[0, 2] *= scale_width
    updated_intrinsics[1, 1] *= scale_height
    updated_intrinsics[1, 2] *= scale_height

    # Adjust principal point due to cropping
    updated_intrinsics[0, 2] -= left_crop
    updated_intrinsics[1, 2] -= top_crop

    return updated_intrinsics


def convert_egopose_to_matrix_numpy(egopose):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = Quaternion(egopose['rotation']).rotation_matrix
    translation = np.array(egopose['translation'])
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix


def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


def mat2pose_vec(matrix: torch.Tensor):
    """
    Converts a 4x4 pose matrix into a 6-dof pose vector
    Args:
        matrix (ndarray): 4x4 pose matrix
    Returns:
        vector (ndarray): 6-dof pose vector comprising translation components (tx, ty, tz) and
        rotation components (rx, ry, rz)
    """

    # M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    rotx = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])

    # M[0, 2] = +siny, M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    cosy = torch.sqrt(matrix[..., 1, 2] ** 2 + matrix[..., 2, 2] ** 2)
    roty = torch.atan2(matrix[..., 0, 2], cosy)

    # M[0, 0] = +cosy*cosz, M[0, 1] = -cosy*sinz
    rotz = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])

    rotation = torch.stack((rotx, roty, rotz), dim=-1)

    # Extract translation params
    translation = matrix[..., :3, 3]
    return torch.cat((translation, rotation), dim=-1)


def euler2mat(angle: torch.Tensor):
    """Convert euler angles to rotation matrix.
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) [Bx3]
    Returns:
        Rotation matrix corresponding to the euler angles [Bx3x3]
    """
    shape = angle.shape
    angle = angle.view(-1, 3)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = torch.zeros_like(z)
    ones = torch.ones_like(z)
    zmat = torch.stack([cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones], dim=1).view(-1, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], dim=1).view(-1, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx], dim=1).view(-1, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    rot_mat = rot_mat.view(*shape[:-1], 3, 3)
    return rot_mat


def pose_vec2mat(vec: torch.Tensor):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz [B,6]
    Returns:
        A transformation matrix [B,4,4]
    """
    translation = vec[..., :3].unsqueeze(-1)  # [...x3x1]
    rot = vec[..., 3:].contiguous()  # [...x3]
    rot_mat = euler2mat(rot)  # [...,3,3]
    transform_mat = torch.cat([rot_mat, translation], dim=-1)  # [...,3,4]
    transform_mat = torch.nn.functional.pad(transform_mat, [0, 0, 0, 1], value=0)  # [...,4,4]
    transform_mat[..., 3, 3] = 1.0
    return transform_mat


def invert_pose_matrix(x):
    """
    Parameters
    ----------
        x: [B, 4, 4] batch of pose matrices

    Returns
    -------
        out: [B, 4, 4] batch of inverse pose matrices
    """
    assert len(x.shape) == 3 and x.shape[1:] == (4, 4), 'Only works for batch of pose matrices.'

    transposed_rotation = torch.transpose(x[:, :3, :3], 1, 2)
    translation = x[:, :3, 3:]

    inverse_mat = torch.cat([transposed_rotation, -torch.bmm(transposed_rotation, translation)], dim=-1)  # [B,3,4]
    inverse_mat = torch.nn.functional.pad(inverse_mat, [0, 0, 0, 1], value=0)  # [B,4,4]
    inverse_mat[..., 3, 3] = 1.0
    return inverse_mat


def warp_features(x, flow, mode='nearest', spatial_extent=None):
    """ Applies a rotation and translation to feature map x.
        Args:
            x: (b, c, h, w) feature map
            flow: (b, 6) 6DoF vector (only uses the xy poriton)
            mode: use 'nearest' when dealing with categorical inputs
        Returns:
            in plane transformed feature map
        """
    if flow is None:
        return x
    b, c, h, w = x.shape
    # z-rotation
    angle = flow[:, 5].clone()  # torch.atan2(flow[:, 1, 0], flow[:, 0, 0])
    # x-y translation
    translation = flow[:, :2].clone()  # flow[:, :2, 3]

    # Normalise translation. Need to divide by how many meters is half of the image.
    # because translation of 1.0 correspond to translation of half of the image.
    translation[:, 0] /= spatial_extent[0]
    translation[:, 1] /= spatial_extent[1]
    # forward axis is inverted
    translation[:, 0] *= -1

    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)

    # output = Rot.input + translation
    # tx and ty are inverted as is the case when going from real coordinates to numpy coordinates
    # translation_pos_0 -> positive value makes the image move to the left
    # translation_pos_1 -> positive value makes the image move to the top
    # Angle -> positive value in rad makes the image move in the trigonometric way
    transformation = torch.stack([cos_theta, -sin_theta, translation[:, 1],
                                  sin_theta, cos_theta, translation[:, 0]], dim=-1).view(b, 2, 3)

    # Note that a rotation will preserve distances only if height = width. Otherwise there's
    # resizing going on. e.g. rotation of pi/2 of a 100x200 image will make what's in the center of the image
    # elongated.
    grid = torch.nn.functional.affine_grid(transformation, size=x.shape, align_corners=False)
    warped_x = torch.nn.functional.grid_sample(x, grid.float(), mode=mode, padding_mode='zeros', align_corners=False)

    return warped_x


def cumulative_warp_features(x, flow, mode='nearest', spatial_extent=None):
    """ Warps a sequence of feature maps by accumulating incremental 2d flow.

    x[:, -1] remains unchanged
    x[:, -2] is warped using flow[:, -2]
    x[:, -3] is warped using flow[:, -3] @ flow[:, -2]
    ...
    x[:, 0] is warped using flow[:, 0] @ ... @ flow[:, -3] @ flow[:, -2]

    Args:
        x: (b, t, c, h, w) sequence of feature maps
        flow: (b, t, 6) sequence of 6 DoF pose
            from t to t+1 (only uses the xy poriton)

    """
    sequence_length = x.shape[1]
    if sequence_length == 1:
        return x

    flow = pose_vec2mat(flow)

    out = [x[:, -1]]
    cum_flow = flow[:, -2]
    for t in reversed(range(sequence_length - 1)):
        out.append(warp_features(x[:, t], mat2pose_vec(cum_flow), mode=mode, spatial_extent=spatial_extent))
        # @ is the equivalent of torch.bmm
        cum_flow = flow[:, t - 1] @ cum_flow

    return torch.stack(out[::-1], 1)


def cumulative_warp_features_reverse(x, flow, mode='nearest', spatial_extent=None):
    """ Warps a sequence of feature maps by accumulating incremental 2d flow.

    x[:, 0] remains unchanged
    x[:, 1] is warped using flow[:, 0].inverse()
    x[:, 2] is warped using flow[:, 0].inverse() @ flow[:, 1].inverse()
    ...

    Args:
        x: (b, t, c, h, w) sequence of feature maps
        flow: (b, t, 6) sequence of 6 DoF pose
            from t to t+1 (only uses the xy poriton)

    """
    flow = pose_vec2mat(flow)

    out = [x[:, 0]]

    for i in range(1, x.shape[1]):
        if i == 1:
            cum_flow = invert_pose_matrix(flow[:, 0])
        else:
            cum_flow = cum_flow @ invert_pose_matrix(flow[:, i - 1])
        out.append(warp_features(x[:, i], mat2pose_vec(cum_flow), mode, spatial_extent=spatial_extent))
    return torch.stack(out, 1)


class VoxelsSumming(torch.autograd.Function):
    """Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/tools.py#L193"""
    @staticmethod
    def forward(ctx, x, geometry, ranks):
        """The features `x` and `geometry` are ranked by voxel positions."""
        # Cumulative sum of all features.
        x = x.cumsum(0)

        # Indicates the change of voxel.
        mask = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        mask[:-1] = ranks[1:] != ranks[:-1]

        x, geometry = x[mask], geometry[mask]
        # Calculate sum of features within a voxel.
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        ctx.save_for_backward(mask)
        ctx.mark_non_differentiable(geometry)

        return x, geometry

    @staticmethod
    def backward(ctx, grad_x, grad_geometry):
        (mask,) = ctx.saved_tensors
        # Since the operation is summing, we simply need to send gradient
        # to all elements that were part of the summation process.
        indices = torch.cumsum(mask, 0)
        indices[mask] -= 1

        output_grad = grad_x[indices]

        return output_grad, None, None
