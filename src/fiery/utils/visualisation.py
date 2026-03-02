import numpy as np
import torch
import matplotlib.pylab

from fiery.utils.instance import predict_instance_segmentation_and_trajectories

DEFAULT_COLORMAP = matplotlib.pylab.cm.jet


def flow_to_image(flow: np.ndarray, autoscale: bool = False) -> np.ndarray:
    """
    Applies colour map to flow which should be a 2 channel image tensor HxWx2. Returns a HxWx3 numpy image
    Code adapted from: https://github.com/liruoteng/FlowNet/blob/master/models/flownet/scripts/flowlib.py
    """
    u = flow[0, :, :]
    v = flow[1, :, :]

    # Convert to polar coordinates
    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = np.max(rad)

    # Normalise flow maps
    if autoscale:
        u /= maxrad + np.finfo(float).eps
        v /= maxrad + np.finfo(float).eps

    # visualise flow with cmap
    return np.uint8(compute_color(u, v) * 255)


def _normalise(image: np.ndarray) -> np.ndarray:
    lower = np.min(image)
    delta = np.max(image) - lower
    if delta == 0:
        delta = 1
    image = (image.astype(np.float32) - lower) / delta
    return image


def apply_colour_map(
    image: np.ndarray, cmap: matplotlib.colors.LinearSegmentedColormap = DEFAULT_COLORMAP, autoscale: bool = False
) -> np.ndarray:
    """
    Applies a colour map to the given 1 or 2 channel numpy image. if 2 channel, must be 2xHxW.
    Returns a HxWx3 numpy image
    """
    if image.ndim == 2 or (image.ndim == 3 and image.shape[0] == 1):
        if image.ndim == 3:
            image = image[0]
        # grayscale scalar image
        if autoscale:
            image = _normalise(image)
        return cmap(image)[:, :, :3]
    if image.shape[0] == 2:
        # 2 dimensional UV
        return flow_to_image(image, autoscale=autoscale)
    if image.shape[0] == 3:
        # normalise rgb channels
        if autoscale:
            image = _normalise(image)
        return np.transpose(image, axes=[1, 2, 0])
    raise Exception('Image must be 1, 2 or 3 channel to convert to colour_map (CxHxW)')


def heatmap_image(
    image: np.ndarray, cmap: matplotlib.colors.LinearSegmentedColormap = DEFAULT_COLORMAP, autoscale: bool = True
) -> np.ndarray:
    """Colorize an 1 or 2 channel image with a colourmap."""
    if not issubclass(image.dtype.type, np.floating):
        raise ValueError(f"Expected a ndarray of float type, but got dtype {image.dtype}")
    if not (image.ndim == 2 or (image.ndim == 3 and image.shape[0] in [1, 2])):
        raise ValueError(f"Expected a ndarray of shape [H, W] or [1, H, W] or [2, H, W], but got shape {image.shape}")
    heatmap_np = apply_colour_map(image, cmap=cmap, autoscale=autoscale)
    heatmap_np = np.uint8(heatmap_np * 255)
    return heatmap_np


def compute_color(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    assert u.shape == v.shape
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nan_mask = np.isnan(u) | np.isnan(v)
    u[nan_mask] = 0
    v[nan_mask] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    f_k = (a + 1) / 2 * (ncols - 1) + 1
    k_0 = np.floor(f_k).astype(int)
    k_1 = k_0 + 1
    k_1[k_1 == ncols + 1] = 1
    f = f_k - k_0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k_0 - 1] / 255
        col1 = tmp[k_1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = col * (1 - nan_mask)

    return img


def make_color_wheel() -> np.ndarray:
    """
    Create colour wheel.
    Code adapted from https://github.com/liruoteng/FlowNet/blob/master/models/flownet/scripts/flowlib.py
    """
    red_yellow = 15
    yellow_green = 6
    green_cyan = 4
    cyan_blue = 11
    blue_magenta = 13
    magenta_red = 6

    ncols = red_yellow + yellow_green + green_cyan + cyan_blue + blue_magenta + magenta_red
    colorwheel = np.zeros([ncols, 3])

    col = 0

    # red_yellow
    colorwheel[0:red_yellow, 0] = 255
    colorwheel[0:red_yellow, 1] = np.transpose(np.floor(255 * np.arange(0, red_yellow) / red_yellow))
    col += red_yellow

    # yellow_green
    colorwheel[col : col + yellow_green, 0] = 255 - np.transpose(
        np.floor(255 * np.arange(0, yellow_green) / yellow_green)
    )
    colorwheel[col : col + yellow_green, 1] = 255
    col += yellow_green

    # green_cyan
    colorwheel[col : col + green_cyan, 1] = 255
    colorwheel[col : col + green_cyan, 2] = np.transpose(np.floor(255 * np.arange(0, green_cyan) / green_cyan))
    col += green_cyan

    # cyan_blue
    colorwheel[col : col + cyan_blue, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, cyan_blue) / cyan_blue))
    colorwheel[col : col + cyan_blue, 2] = 255
    col += cyan_blue

    # blue_magenta
    colorwheel[col : col + blue_magenta, 2] = 255
    colorwheel[col : col + blue_magenta, 0] = np.transpose(np.floor(255 * np.arange(0, blue_magenta) / blue_magenta))
    col += +blue_magenta

    # magenta_red
    colorwheel[col : col + magenta_red, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, magenta_red) / magenta_red))
    colorwheel[col : col + magenta_red, 0] = 255

    return colorwheel


def make_contour(img, colour=[0, 0, 0], double_line=False):
    h, w = img.shape[:2]
    out = img.copy()
    # Vertical lines
    out[np.arange(h), np.repeat(0, h)] = colour
    out[np.arange(h), np.repeat(w - 1, h)] = colour

    # Horizontal lines
    out[np.repeat(0, w), np.arange(w)] = colour
    out[np.repeat(h - 1, w), np.arange(w)] = colour

    if double_line:
        out[np.arange(h), np.repeat(1, h)] = colour
        out[np.arange(h), np.repeat(w - 2, h)] = colour

        # Horizontal lines
        out[np.repeat(1, w), np.arange(w)] = colour
        out[np.repeat(h - 2, w), np.arange(w)] = colour
    return out


def plot_instance_map(instance_image, instance_map, instance_colours=None, bg_image=None):
    if isinstance(instance_image, torch.Tensor):
        instance_image = instance_image.cpu().numpy()
    assert isinstance(instance_image, np.ndarray)
    if instance_colours is None:
        instance_colours = generate_instance_colours(instance_map)
    if len(instance_image.shape) > 2:
        instance_image = instance_image.reshape((instance_image.shape[-2], instance_image.shape[-1]))

    if bg_image is None:
        plot_image = 255 * np.ones((instance_image.shape[0], instance_image.shape[1], 3), dtype=np.uint8)
    else:
        plot_image = bg_image

    for key, value in instance_colours.items():
        plot_image[instance_image == key] = value

    return plot_image


def visualise_output(labels, output, cfg):
    semantic_colours = np.array([[255, 255, 255], [0, 0, 0]], dtype=np.uint8)

    consistent_instance_seg = predict_instance_segmentation_and_trajectories(
        output, compute_matched_centers=False
    )

    sequence_length = consistent_instance_seg.shape[1]
    b = 0
    video = []
    for t in range(sequence_length):
        out_t = []
        # Ground truth
        unique_ids = torch.unique(labels['instance'][b, t]).cpu().numpy()[1:]
        instance_map = dict(zip(unique_ids, unique_ids))
        instance_plot = plot_instance_map(labels['instance'][b, t].cpu(), instance_map)[::-1, ::-1]
        instance_plot = make_contour(instance_plot)

        semantic_seg = labels['segmentation'].squeeze(2).cpu().numpy()
        semantic_plot = semantic_colours[semantic_seg[b, t][::-1, ::-1]]
        semantic_plot = make_contour(semantic_plot)

        if cfg.INSTANCE_FLOW.ENABLED:
            future_flow_plot = labels['flow'][b, t].cpu().numpy()
            future_flow_plot[:, semantic_seg[b, t] != 1] = 0
            future_flow_plot = flow_to_image(future_flow_plot)[::-1, ::-1]
            future_flow_plot = make_contour(future_flow_plot)
        else:
            future_flow_plot = np.zeros_like(semantic_plot)

        center_plot = heatmap_image(labels['centerness'][b, t, 0].cpu().numpy())[::-1, ::-1]
        center_plot = make_contour(center_plot)

        offset_plot = labels['offset'][b, t].cpu().numpy()
        offset_plot[:, semantic_seg[b, t] != 1] = 0
        offset_plot = flow_to_image(offset_plot)[::-1, ::-1]
        offset_plot = make_contour(offset_plot)

        out_t.append(np.concatenate([instance_plot, future_flow_plot,
                                     semantic_plot, center_plot, offset_plot], axis=0))

        # Predictions
        unique_ids = torch.unique(consistent_instance_seg[b, t]).cpu().numpy()[1:]
        instance_map = dict(zip(unique_ids, unique_ids))
        instance_plot = plot_instance_map(consistent_instance_seg[b, t].cpu(), instance_map)[::-1, ::-1]
        instance_plot = make_contour(instance_plot)

        semantic_seg = output['segmentation'].argmax(dim=2).detach().cpu().numpy()
        semantic_plot = semantic_colours[semantic_seg[b, t][::-1, ::-1]]
        semantic_plot = make_contour(semantic_plot)

        if cfg.INSTANCE_FLOW.ENABLED:
            future_flow_plot = output['instance_flow'][b, t].detach().cpu().numpy()
            future_flow_plot[:, semantic_seg[b, t] != 1] = 0
            future_flow_plot = flow_to_image(future_flow_plot)[::-1, ::-1]
            future_flow_plot = make_contour(future_flow_plot)
        else:
            future_flow_plot = np.zeros_like(semantic_plot)

        center_plot = heatmap_image(output['instance_center'][b, t, 0].detach().cpu().numpy())[::-1, ::-1]
        center_plot = make_contour(center_plot)

        offset_plot = output['instance_offset'][b, t].detach().cpu().numpy()
        offset_plot[:, semantic_seg[b, t] != 1] = 0
        offset_plot = flow_to_image(offset_plot)[::-1, ::-1]
        offset_plot = make_contour(offset_plot)

        out_t.append(np.concatenate([instance_plot, future_flow_plot,
                                     semantic_plot, center_plot, offset_plot], axis=0))
        out_t = np.concatenate(out_t, axis=1)
        # Shape (C, H, W)
        out_t = out_t.transpose((2, 0, 1))

        video.append(out_t)

    # Shape (B, T, C, H, W)
    video = np.stack(video)[None]
    return video


def convert_figure_numpy(figure):
    """ Convert figure to numpy image """
    figure_np = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    figure_np = figure_np.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    return figure_np


def generate_instance_colours(instance_map):
    # Most distinct 22 colors (kelly colors from https://stackoverflow.com/questions/470690/how-to-automatically-generate
    # -n-distinct-colors)
    # plus some colours from AD40k
    INSTANCE_COLOURS = np.asarray([
        [0, 0, 0],
        [255, 179, 0],
        [128, 62, 117],
        [255, 104, 0],
        [166, 189, 215],
        [193, 0, 32],
        [206, 162, 98],
        [129, 112, 102],
        [0, 125, 52],
        [246, 118, 142],
        [0, 83, 138],
        [255, 122, 92],
        [83, 55, 122],
        [255, 142, 0],
        [179, 40, 81],
        [244, 200, 0],
        [127, 24, 13],
        [147, 170, 0],
        [89, 51, 21],
        [241, 58, 19],
        [35, 44, 22],
        [112, 224, 255],
        [70, 184, 160],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [0, 255, 235],
        [255, 0, 235],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 255, 204],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [255, 214, 0],
        [25, 194, 194],
        [92, 0, 255],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
    ])

    return {instance_id: INSTANCE_COLOURS[global_instance_id % len(INSTANCE_COLOURS)] for
            instance_id, global_instance_id in instance_map.items()
            }
