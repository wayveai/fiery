# FIERY
This is the PyTorch implementation for inference and training of the future prediction bird's-eye view network as 
described in:

> **FIERY: Future Instance Segmentation in Bird's-Eye view from Surround Monocular Cameras**
>
> [Anthony Hu](https://anthonyhu.github.io/), [Zak Murez](http://zak.murez.com/), 
[Nikhil Mohan](https://uk.linkedin.com/in/nikhilmohan33), 
[Sofía Dudas](https://uk.linkedin.com/in/sof%C3%ADa-josefina-lago-dudas-2b0737132), 
[Jeffrey Hawke](https://uk.linkedin.com/in/jeffrey-hawke), 
[Vijay Badrinarayanan](https://sites.google.com/site/vijaybacademichomepage/home), 
[Roberto Cipolla](https://mi.eng.cam.ac.uk/~cipolla/index.htm) and [Alex Kendall](https://alexgkendall.com/)  
>
> [ICCV 2021 (Oral)](https://arxiv.org/abs/2104.10490)<br/>
> [Blog post](https://wayve.ai/blog/fiery-future-instance-prediction-birds-eye-view)

<p align="center">
     <img src="https://github.com/wayveai/fiery/releases/download/v1.0/predictions.gif" alt="FIERY future prediction">
     <br/>
     <sub><em>Multimodal future predictions by our bird’s-eye view network.<br/>
Top two rows: RGB camera inputs. The predicted future trajectories and segmentations are projected to the ground plane in the images.<br/>
Bottom row: future instance prediction in bird’s-eye view in a 100m×100m capture size around the ego-vehicle, which is indicated by a black rectangle in the center.
    </em></sub>
</p>

If you find our work useful, please consider citing:
```bibtex
@inproceedings{fiery2021,
  title     = {{FIERY}: Future Instance Segmentation in Bird's-Eye view from Surround Monocular Cameras},
  author    = {Anthony Hu and Zak Murez and Nikhil Mohan and Sofía Dudas and 
               Jeffrey Hawke and Vijay Badrinarayanan and Roberto Cipolla and Alex Kendall},
  booktitle = {Proceedings of the International Conference on Computer Vision ({ICCV})},
  year = {2021}
}
```

## ⚙ Setup
- Create the [conda](https://docs.conda.io/en/latest/miniconda.html) environment by running `conda env create`.
- Or install with pip: `pip install -r requirements.txt`.

## 🏄 Prediction
### Visualisation

In a colab notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12ahc3whI1RQZIVDi53grMWHzdA7WqIuo?usp=sharing)

Or locally:
- Download [pre-trained weights](https://github.com/wayveai/fiery/releases/download/v1.0/fiery.ckpt).
- Run `python visualise.py --checkpoint ${CHECKPOINT_PATH}`. This will render predictions from the network and save 
them to an `output_vis` folder.

### Evaluation
- Download the [NuScenes dataset](https://www.nuscenes.org/download). For detailed instructions, see [DATASET.md](DATASET.md).
- Download [pre-trained weights](https://github.com/wayveai/fiery/releases/download/v1.0/fiery.ckpt).
- Run `python evaluate.py --checkpoint ${CHECKPOINT_PATH} --dataroot ${NUSCENES_DATAROOT}`.
- Or with Hydra configs: `python eval.py checkpoint_path=${CHECKPOINT_PATH} fiery.DATASET.DATAROOT=${NUSCENES_DATAROOT}`.

## 🔥 Pre-trained models

All the configs are in the folder `configs/experiment`

| Config and weights      | Dataset | Past context | Future horizon | BEV size | IoU  | VPQ|
|--------------|---------|-----------------------|----------------|----------|------|----|
| [`baseline.yaml`](https://github.com/wayveai/fiery/releases/download/v1.0/fiery.ckpt) | NuScenes | 1.0s | 2.0s | 100mx100m (50cm res.) | 36.7 | 29.9 |
| [`lyft/baseline.yaml`](https://github.com/wayveai/fiery/releases/download/v1.0/lyft_fiery.ckpt) | Lyft | 0.8s | 2.0s| 100mx100m (50cm res.) | 36.3 | 29.2 |
| [`literature/static_pon_setting.yaml`](https://github.com/wayveai/fiery/releases/download/v1.0/static_pon_setting.ckpt) | NuScenes| 0.0s | 0.0s | 100mx50m (25cm res.) | 37.7| - |
| [`literature/pon_setting.yaml`](https://github.com/wayveai/fiery/releases/download/v1.0/pon_setting.ckpt) | NuScenes| 1.0s | 0.0s | 100mx50m (25cm res.) |39.9 | - |
| [`literature/static_lss_setting.yaml`](https://github.com/wayveai/fiery/releases/download/v1.0/static_lift_splat_setting.ckpt) | NuScenes | 0.0s | 0.0s | 100mx100m (50cm res.) | 35.8 | - |
| [`literature/lift_splat_setting.yaml`](https://github.com/wayveai/fiery/releases/download/v1.0/lift_splat_setting.ckpt) | NuScenes | 1.0s | 0.0s | 100mx100m (50cm res.) | 38.2 | - |
| [`literature/fishing_setting.yaml`](https://github.com/wayveai/fiery/releases/download/v1.0/fishing_setting.ckpt) | NuScenes | 1.0s | 2.0s | 32.0mx19.2m (10cm res.) | 57.6 | - |


## 🏊 Training
To train the model from scratch on NuScenes:
- Download the [NuScenes dataset](https://www.nuscenes.org/download). For detailed instructions, see [DATASET.md](DATASET.md).
- Run `python train.py experiment=baseline data=nuscenes paths.data_dir=${NUSCENES_DATAROOT_PARENT}`.
  (Expected layout: ${paths.data_dir}/nuscenes/{trainval,mini})

This will train the model on 4 GPUs, each with a batch of size 3. To train on single GPU use
`trainer.devices=[0]`, and to change the batch size use `fiery.BATCHSIZE=${DESIRED_BATCHSIZE}`.

## 🙌 Credits
Big thanks to Giulio D'Ippolito ([@gdippolito](https://github.com/gdippolito)) for the technical help on the gpu 
servers, Piotr Sokólski ([@pyetras](https://github.com/pyetras)) for the panoptic metric implementation, and to Hannes Liik ([@hannesliik](https://github.com/hannesliik)) 
for the awesome future trajectory visualisation on the ground plane.
