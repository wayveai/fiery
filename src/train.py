from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from fiery.config import get_cfg
from fiery.data import FieryDataModule
from fiery.trainer import TrainingModule
from fiery.utils.repro import seed_everything

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs"


def _build_callbacks(callbacks_cfg):
    callbacks = []
    if callbacks_cfg is None:
        return callbacks

    callbacks_dict = OmegaConf.to_container(callbacks_cfg, resolve=True)

    if "model_checkpoint" in callbacks_dict:
        callbacks.append(ModelCheckpoint(**callbacks_dict["model_checkpoint"]))
    if "learning_rate_monitor" in callbacks_dict:
        callbacks.append(LearningRateMonitor(**callbacks_dict["learning_rate_monitor"]))

    return callbacks


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="train.yaml")
def main(cfg: DictConfig):
    load_dotenv()

    if cfg.get("seed") is not None:
        seed_everything(cfg.seed, deterministic=bool(cfg.get("deterministic", False)))

    fiery_cfg_dict = OmegaConf.to_container(cfg.fiery, resolve=True)
    fiery_cfg = get_cfg(cfg_dict=fiery_cfg_dict)

    datamodule = FieryDataModule(fiery_cfg)
    model = TrainingModule(fiery_cfg.convert_to_dict())

    if fiery_cfg.PRETRAINED.LOAD_WEIGHTS:
        pretrained_model_weights = torch.load(
            Path(fiery_cfg.DATASET.DATAROOT) / fiery_cfg.PRETRAINED.PATH,
            map_location="cpu",
            weights_only=False,
        )["state_dict"]
        model.load_state_dict(pretrained_model_weights, strict=False)
        print(f"Loaded single-image model weights from {fiery_cfg.PRETRAINED.PATH}")

    logger_cfg = OmegaConf.to_container(cfg.logger, resolve=True) if cfg.get("logger") is not None else None
    logger = TensorBoardLogger(**logger_cfg) if logger_cfg is not None else None

    callbacks = _build_callbacks(cfg.get("callbacks"))
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)

    trainer = pl.Trainer(**trainer_kwargs, logger=logger, callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
