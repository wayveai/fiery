import os
import time
import socket
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from fiery.config import get_parser, get_cfg
from fiery.data import prepare_dataloaders
from fiery.trainer import TrainingModule


def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    trainloader, valloader = prepare_dataloaders(cfg)
    model = TrainingModule(cfg.convert_to_dict())
    checkpoint_path = '/home/master/10/cytseng/fiery/tensorboard_logs/24November2021at11:16:14CST_cml26_lift_splat_setting/default/version_0/checkpoints/'+'epoch=23-step=165251.ckpt'
#     model = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    if cfg.PRETRAINED.LOAD_WEIGHTS:
        # Load single-image instance segmentation model.
        pretrained_model_weights = torch.load(
            os.path.join(cfg.DATASET.DATAROOT, cfg.PRETRAINED.PATH), map_location='cpu'
        )['state_dict']

        model.load_state_dict(pretrained_model_weights, strict=False)
        print(f'Loaded single-image model weights from {cfg.PRETRAINED.PATH}')

    save_dir = os.path.join(
        cfg.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
    )
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
    trainer = pl.Trainer(
        gpus=cfg.GPUS,
        accelerator='ddp',
        precision=cfg.PRECISION,
        sync_batchnorm=True,
        gradient_clip_val=cfg.GRAD_NORM_CLIP,
        max_epochs=cfg.EPOCHS,
        weights_summary='full',
        logger=tb_logger,
        # log_every_n_steps=cfg.LOGGING_INTERVAL,
        val_check_interval=0.25,
        num_sanity_val_steps=0,
        plugins=DDPPlugin(find_unused_parameters=True),
        profiler='simple',
        resume_from_checkpoint=checkpoint_path,
    )
   
    trainer.fit(model, trainloader, valloader)
#     trainer.fit(model, trainloader, valloader, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    main()
