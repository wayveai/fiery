import os
from argparse import ArgumentParser
from glob import glob

import cv2
import numpy as np
import torch
import torchvision
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from fiery.trainer import TrainingModule
from fiery.utils.network import NormalizeInverse
from fiery.utils.instance import predict_instance_segmentation_and_trajectories
from fiery.utils.visualisation import plot_instance_map, generate_instance_colours, make_contour, convert_figure_numpy

EXAMPLE_DATA_PATH = 'example_data'


def plot_prediction(image, output, cfg):
    # Process predictions
    consistent_instance_seg, matched_centers = predict_instance_segmentation_and_trajectories(
        output, compute_matched_centers=True
    )

    # Plot future trajectories
    unique_ids = torch.unique(consistent_instance_seg[0, 0]).cpu().long().numpy()[1:]
    instance_map = dict(zip(unique_ids, unique_ids))
    instance_colours = generate_instance_colours(instance_map)
    vis_image = plot_instance_map(consistent_instance_seg[0, 0].cpu().numpy(), instance_map)
    trajectory_img = np.zeros(vis_image.shape, dtype=np.uint8)
    for instance_id in unique_ids:
        path = matched_centers[instance_id]
        for t in range(len(path) - 1):
            color = instance_colours[instance_id].tolist()
            cv2.line(trajectory_img, tuple(path[t]), tuple(path[t + 1]),
                     color, 4)

    # Overlay arrows
    temp_img = cv2.addWeighted(vis_image, 0.7, trajectory_img, 0.3, 1.0)
    mask = ~ np.all(trajectory_img == 0, axis=2)
    vis_image[mask] = temp_img[mask]

    # Plot present RGB frames and predictions
    val_w = 2.99
    cameras = cfg.IMAGE.NAMES
    image_ratio = cfg.IMAGE.FINAL_DIM[0] / cfg.IMAGE.FINAL_DIM[1]
    val_h = val_w * image_ratio
    fig = plt.figure(figsize=(4 * val_w, 2 * val_h))
    width_ratios = (val_w, val_w, val_w, val_w)
    gs = mpl.gridspec.GridSpec(2, 4, width_ratios=width_ratios)
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    denormalise_img = torchvision.transforms.Compose(
        (NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         torchvision.transforms.ToPILImage(),)
    )
    for imgi, img in enumerate(image[0, -1]):
        ax = plt.subplot(gs[imgi // 3, imgi % 3])
        showimg = denormalise_img(img.cpu())
        if imgi > 2:
            showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)

        plt.annotate(cameras[imgi].replace('_', ' ').replace('CAM ', ''), (0.01, 0.87), c='white',
                     xycoords='axes fraction', fontsize=14)
        plt.imshow(showimg)
        plt.axis('off')

    ax = plt.subplot(gs[:, 3])
    plt.imshow(make_contour(vis_image[::-1, ::-1]))
    plt.axis('off')

    plt.draw()
    figure_numpy = convert_figure_numpy(fig)
    plt.close()
    return figure_numpy


def download_example_data():
    from requests import get

    def download(url, file_name):
        # open in binary mode
        with open(file_name, "wb") as file:
            # get request
            response = get(url)
            # write to file
            file.write(response.content)

    os.makedirs(EXAMPLE_DATA_PATH, exist_ok=True)
    url_list = ['https://github.com/wayveai/fiery/releases/download/v1.0/example_1.npz',
                'https://github.com/wayveai/fiery/releases/download/v1.0/example_2.npz',
                'https://github.com/wayveai/fiery/releases/download/v1.0/example_3.npz',
                'https://github.com/wayveai/fiery/releases/download/v1.0/example_4.npz'
                ]
    for url in url_list:
        download(url, os.path.join(EXAMPLE_DATA_PATH, os.path.basename(url)))


def visualise(checkpoint_path):
    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)

    device = torch.device('cuda:0')
    trainer = trainer.to(device)
    trainer.eval()

    # Download example data
    download_example_data()
    # Load data
    for data_path in sorted(glob(os.path.join(EXAMPLE_DATA_PATH, '*.npz'))):
        data = np.load(data_path)
        image = torch.from_numpy(data['image']).to(device)
        intrinsics = torch.from_numpy(data['intrinsics']).to(device)
        extrinsics = torch.from_numpy(data['extrinsics']).to(device)
        future_egomotions = torch.from_numpy(data['future_egomotion']).to(device)

        # Forward pass
        with torch.no_grad():
            output = trainer.model(image, intrinsics, extrinsics, future_egomotions)

        figure_numpy = plot_prediction(image, output, trainer.cfg)
        os.makedirs('./output_vis', exist_ok=True)
        output_filename = os.path.join('./output_vis', os.path.basename(data_path).split('.')[0]) + '.png'
        Image.fromarray(figure_numpy).save(output_filename)
        print(f'Saved output in {output_filename}')


if __name__ == '__main__':
    parser = ArgumentParser(description='Fiery visualisation')
    parser.add_argument('--checkpoint', default='./fiery.ckpt', type=str, help='path to checkpoint')

    args = parser.parse_args()

    visualise(args.checkpoint)
