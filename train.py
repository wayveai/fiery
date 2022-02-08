import os
# import time
# import socket
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.callbacks import ModelCheckpoint

from fiery.config import get_parser, get_cfg
from fiery.data import prepare_dataloaders
from fiery.trainer import TrainingModule


def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    trainloader, valloader, testloader = prepare_dataloaders(cfg)
    model = TrainingModule(cfg.convert_to_dict())

    if cfg.PRETRAINED.LOAD_WEIGHTS:
        # Load single-image instance segmentation model.
        pretrained_model_weights = torch.load(
            os.path.join(cfg.DATASET.DATAROOT, cfg.PRETRAINED.PATH), map_location='cpu'
        )['state_dict']

        model.load_state_dict(pretrained_model_weights, strict=False)
        print(f'Loaded single-image model weights from {cfg.PRETRAINED.PATH}')

    # save_dir = os.path.join(
    #     cfg.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
    # )
    save_dir_tags = [cfg.TAG, cfg.OBJ.HEAD_NAME, f'{cfg.IMAGE.N_CAMERA}_cam', f'imgSize_{cfg.IMAGE.FINAL_DIM[0]}_{cfg.IMAGE.FINAL_DIM[1]}']
    if cfg.LOSS.SEG_USE is True:
        save_dir_tags.append('segLoss')

    if cfg.DATASET.VERSION == 'v1.0-mini':
        save_dir_tags.append('mini')

    save_dir = os.path.join(cfg.LOG_DIR, '_'.join(save_dir_tags))
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=save_dir, name=None)

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=save_dir,
    # )

    trainer = pl.Trainer(
        gpus=cfg.GPUS,
        strategy=DDPPlugin(find_unused_parameters=True),
        precision=cfg.PRECISION,
        sync_batchnorm=True,
        gradient_clip_val=cfg.GRAD_NORM_CLIP,
        max_epochs=cfg.EPOCHS,
        weights_summary=cfg.WEIGHT_SUMMARY,
        logger=tb_logger,
        # log_every_n_steps=cfg.LOGGING_INTERVAL,
        val_check_interval=cfg.VALID_FREQ,
        num_sanity_val_steps=0,
        profiler='simple',

        # callbacks=checkpoint_callback,
        # resume_from_checkpoint=checkpoint_path,                                                              h,
    )

    if args.eval_path is None:
        trainer.fit(model, trainloader, valloader)
        trainer.test(dataloaders=testloader, ckpt_path='best', verbose=False)

    else:
        trainer.test(model=model, dataloaders=testloader, ckpt_path=args.eval_path, verbose=False)
    # trainer.save_checkpoint("final_epoch.ckpt")


if __name__ == "__main__":
    main()
