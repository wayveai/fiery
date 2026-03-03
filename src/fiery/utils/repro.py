import os

import pytorch_lightning as pl
import torch


def seed_everything(seed: int, deterministic: bool = False) -> int:
    pl.seed_everything(seed, workers=True)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    return seed
