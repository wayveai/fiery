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

    save_dir_tags = [
        cfg.TAG,
        cfg.OBJ.HEAD_NAME,
        f'cam_{cfg.IMAGE.N_CAMERA}',
        f'imgSize_{cfg.IMAGE.FINAL_DIM[0]}_{cfg.IMAGE.FINAL_DIM[1]}',
        f'resolution_{cfg.LIFT.X_BOUND[2]}_{cfg.LIFT.Y_BOUND[2]}',
        f'img_encoder_ds_{cfg.MODEL.ENCODER.DOWNSAMPLE}',
        cfg.MODEL.MM.HEAD_MAPPING.get(cfg.MODEL.MM.BBOX_HEAD.type, cfg.MODEL.MM.BBOX_HEAD.type),
        f'bb_in_{cfg.MODEL.MM.BBOX_BACKBONE.in_channels}',
        f'bb_out_{cfg.MODEL.MM.BBOX_BACKBONE.out_channels[0]}_{cfg.MODEL.MM.BBOX_BACKBONE.out_channels[1]}_{cfg.MODEL.MM.BBOX_BACKBONE.out_channels[2]}',
        f'head_in_{cfg.MODEL.MM.BBOX_HEAD.in_channels}',

    ]

    additional_tags = model.model.detection_head.get_additional_tags()
    if additional_tags not in (None, '', [], ()):
        if isinstance(additional_tags, str):
            save_dir_tags.append(additional_tags)
        else:
            save_dir_tags.extend(additional_tags)

    if cfg.LOSS.SEG_USE is True:
        save_dir_tags.append('segLoss')

    if cfg.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE:
        save_dir_tags.append('ego')

    if cfg.MODEL.TEMPORAL_MODEL.NAME == 'temporal_block':
        save_dir_tags.append(f'tem_{cfg.TIME_RECEPTIVE_FIELD}')

    if cfg.SEMANTIC_SEG.NUSCENE_CLASS:
        save_dir_tags.append('semantic')

    if cfg.IMAGE.IMAGE_AUG:
        save_dir_tags.append('image_aug')

    if cfg.MODEL.MM.SEG_CAT_BACKBONE:
        save_dir_tags.append('seg_cat_backbone')

    if cfg.MODEL.MM.SEG_ADD_BACKBONE:
        save_dir_tags.append('seg_add_backbone')

    if cfg.DATASET.INCLUDE_VELOCITY:
        save_dir_tags.append('vel')

    if cfg.DATASET.VERSION == 'v1.0-mini':
        save_dir_tags.append('mini')

    if args.eval_path is not None:
        save_dir_tags.append('test')

    save_dir = os.path.join(cfg.LOG_DIR, '_'.join(save_dir_tags))
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=save_dir, name=None)

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=save_dir,
    # )

    trainer = pl.Trainer(
        gpus=cfg.GPUS,
        # you may want to set find_unused_parameters=True to avoid crashing
        strategy=DDPPlugin(find_unused_parameters=False),
        # strategy=DDPPlugin(find_unused_parameters=True),
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
    )

    # args.eval_path = '/home/master/10/cytseng/fiery/tensorboard_logs/lss_mm_cam_6_imgSize_224_480_segLoss_semantic/version_0/checkpoints/37.ckpt'
    # args.eval_path = '/home/master/10/cytseng/fiery/tensorboard_logs/lss_mm_cam_6_imgSize_336_720_segLoss/version_1/checkpoints/20.ckpt'

    if args.eval_path is None:
        trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader, ckpt_path=cfg.CKPT_PATH)
        trainer.test(dataloaders=testloader, ckpt_path='best', verbose=False)

    else:
        trainer.test(model=model, dataloaders=testloader, ckpt_path=args.eval_path, verbose=False)
    # trainer.save_checkpoint("final_epoch.ckpt")


if __name__ == "__main__":
    main()
