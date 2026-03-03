from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

from fiery.config import get_cfg
from fiery.data import prepare_dataloaders
from fiery.metrics import IntersectionOverUnion, PanopticMetric
from fiery.trainer import TrainingModule
from fiery.utils.instance import predict_instance_segmentation_and_trajectories
from fiery.utils.network import preprocess_batch

# 30mx30m, 100mx100m
EVALUATION_RANGES = {
    "30x30": (70, 130),
    "100x100": (0, 200),
}


def _compose_fiery_cfg_dict(cfg: DictConfig):
    merged = OmegaConf.create({})

    for section in ("data", "experiment", "model"):
        section_cfg = cfg.get(section)
        if section_cfg is not None and "fiery" in section_cfg:
            merged = OmegaConf.merge(merged, section_cfg.fiery)

    if cfg.get("fiery") is not None:
        merged = OmegaConf.merge(merged, cfg.fiery)

    return OmegaConf.to_container(merged, resolve=True)


def evaluate(checkpoint_path, dataroot, version, device="cuda:0"):
    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True, weights_only=False)
    print(f"Loaded weights from \n {checkpoint_path}")
    trainer.eval()

    device = torch.device(device)
    trainer.to(device)
    model = trainer.model

    cfg = model.cfg
    cfg.BATCHSIZE = 1
    cfg.DATASET.DATAROOT = str(dataroot)
    cfg.DATASET.VERSION = version

    _, valloader = prepare_dataloaders(cfg)

    panoptic_metrics = {}
    iou_metrics = {}
    n_classes = len(cfg.SEMANTIC_SEG.WEIGHTS)
    for key in EVALUATION_RANGES.keys():
        panoptic_metrics[key] = PanopticMetric(n_classes=n_classes, temporally_consistent=True).to(device)
        iou_metrics[key] = IntersectionOverUnion(n_classes).to(device)

    for batch in tqdm(valloader):
        preprocess_batch(batch, device)
        image = batch["image"]
        intrinsics = batch["intrinsics"]
        extrinsics = batch["extrinsics"]
        future_egomotion = batch["future_egomotion"]

        batch_size = image.shape[0]

        labels, future_distribution_inputs = trainer.prepare_future_labels(batch)

        with torch.no_grad():
            # Evaluate with mean prediction
            noise = torch.zeros((batch_size, 1, model.latent_dim), device=device)
            output = model(image, intrinsics, extrinsics, future_egomotion, future_distribution_inputs, noise=noise)

        pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
            output, compute_matched_centers=False, make_consistent=True
        )

        segmentation_pred = output["segmentation"].detach()
        segmentation_pred = torch.argmax(segmentation_pred, dim=2, keepdims=True)

        for key, grid in EVALUATION_RANGES.items():
            limits = slice(grid[0], grid[1])
            panoptic_metrics[key](
                pred_consistent_instance_seg[..., limits, limits].contiguous().detach(),
                labels["instance"][..., limits, limits].contiguous(),
            )

            iou_metrics[key](
                segmentation_pred[..., limits, limits].contiguous(),
                labels["segmentation"][..., limits, limits].contiguous(),
            )

    results = {}
    for key in EVALUATION_RANGES.keys():
        panoptic_scores = panoptic_metrics[key].compute()
        for panoptic_key, value in panoptic_scores.items():
            results[f"{panoptic_key}"] = results.get(f"{panoptic_key}", []) + [100 * value[1].item()]

        iou_scores = iou_metrics[key].compute()
        results["iou"] = results.get("iou", []) + [100 * iou_scores[1].item()]

    for panoptic_key in ["iou", "pq", "sq", "rq"]:
        print(panoptic_key)
        print(" & ".join([f"{x:.1f}" for x in results[panoptic_key]]))


@hydra.main(version_base="1.3", config_path=str(Path(__file__).resolve().parent.parent / "configs"), config_name="eval.yaml")
def main(cfg: DictConfig):
    load_dotenv()

    fiery_cfg_dict = _compose_fiery_cfg_dict(cfg)
    fiery_cfg = get_cfg(cfg_dict=fiery_cfg_dict)

    evaluate(
        checkpoint_path=cfg.checkpoint_path,
        dataroot=fiery_cfg.DATASET.DATAROOT,
        version=fiery_cfg.DATASET.VERSION,
        device=cfg.device,
    )


if __name__ == "__main__":
    main()
