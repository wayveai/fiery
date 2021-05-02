# FIERY
This is the PyTorch implementation for inference and training of the future prediction bird's-eye view network as 
described in:

> **FIERY: Future Instance Segmentation in Bird's-Eye view from Surround Monocular Cameras**
>
> [Anthony Hu](https://anthonyhu.github.io/), [Zak Murez](http://zak.murez.com/), 
[Nikhil Mohan](https://uk.linkedin.com/in/nikhilmohan33), 
[Sofía Dudas](https://uk.linkedin.com/in/sof%C3%ADa-josefina-lago-dudas-2b0737132), 
[Jeffrey Hawke](https://uk.linkedin.com/in/jeffrey-hawke), 
[‪Vijay Badrinarayanan](https://sites.google.com/site/vijaybacademichomepage/home), 
[Roberto Cipolla](https://mi.eng.cam.ac.uk/~cipolla/index.htm) and [Alex Kendall](https://alexgkendall.com/)  
>
> [preprint (2021)](https://arxiv.org/abs/2104.10490)<br/>
> [Blog post](https://wayve.ai/blog/fiery-future-instance-prediction-birds-eye-view)

<p align="center">
     <img src="media/predictions.gif" alt="FIERY future prediction">
     <br/>
     <sub><em>Multimodal future predictions by our bird’s-eye view network.<br/>
Top two rows: RGB camera inputs. The predicted future trajectories and segmentations are projected to the ground plane in the images.<br/>
Bottom row: future instance prediction in bird’s-eye view in a 100m×100m capture size around the ego-vehicle, which is indicated by a black rectangle in the center.
    </em></sub>
</p>

If you find our work useful, please consider citing:
```
@inproceedings{fiery2021,
  title     = {{FIERY}: Future Instance Segmentation in Bird's-Eye view from Surround Monocular Cameras},
  author    = {Anthony Hu and Zak Murez and Nikhil Mohan and Sofía Dudas and 
               Jeffrey Hawke and Vijay Badrinarayanan and Roberto Cipolla and Alex Kendall},
  booktitle = {arXiv preprint},
  year = {2021}
}
```

## ⚙ Setup
- Create the [conda](https://docs.conda.io/en/latest/miniconda.html) environment by running `conda env create`.

## 🏄 Prediction

### Visualisation
- Download [pre-trained weights](https://drive.google.com/uc?export=download&id=10H8iZtsqeZukQHkVJl-sH3nSAdbW3d9M).
- Run `python visualise.py --checkpoint ${CHECKPOINT_PATH}`. This will render predictions from the network and save 
them to a `output_vis` folder.

### Evaluation
- Download the [NuScenes dataset](https://www.nuscenes.org/download).
- Download [pre-trained weights](https://drive.google.com/uc?export=download&id=10H8iZtsqeZukQHkVJl-sH3nSAdbW3d9M).
- Run `python evaluate.py --checkpoint ${CHECKPOINT_PATH} --dataroot ${NUSCENES_DATAROOT}`.

## 🔥 Pre-trained models

All the configs are in the folder `fiery/configs`

| Config       | Dataset | Past context | Future horizon | BEV size | IoU  | VPQ|
|--------------|---------|-----------------------|----------------|----------|------|----|
| [`baseline.yml`](https://drive.google.com/uc?export=download&id=10H8iZtsqeZukQHkVJl-sH3nSAdbW3d9M) | NuScenes | 1.0s | 2.0s | 100mx100m (50cm res.) | 37.0 | 29.5 |
| [`lyft/baseline.yml`]() | Lyft | 0.8s | 2.0s | 100mx100m (50cm res.) | 36.3 | 29.2 |
| [`literature/pon_setting.yml`]() | NuScenes | 0.0s | 0.0s | 100mx50m (25cm res.) | 39.9 | - |
| [`literature/lift_splat_setting.yml`]() | NuScenes | 0.0s | 0.0s | 100mx100m (50cm res.) | 36.7 | - |
| [`literature/fishing_setting.yml`]() | NuScenes | 1.0s | 2.0s | 32.0mx19.2m (10cm res.) | 58.5 | - |


## 🏊 Training
To train the model from scratch on NuScenes:
- Run `python train.py --config fiery/configs/baseline.yml DATASET.DATAROOT ${NUSCENES_DATAROOT}`.

To train on single GPU add the flag `GPUS [0]`, and to change the batch size use the flag `BATCHSIZE ${DESIRED_BATCHSIZE}`.

## 🙌 Credits
Big thanks to Piotr Sokólski ([@pyetras](https://github.com/pyetras)) for the panoptic metric implementation, and to 
Hannes Liik ([@hannesliik](https://github.com/hannesliik)) for the awesome future trajectory 
visualisation on the ground plane.
