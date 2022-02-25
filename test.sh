DIR="$1"
CKPT="$2"
PY_ARGS=${@:3}

python train.py --config "$DIR/hparams.yaml" --eval-path "$DIR/checkpoints/$CKPT" BATCHSIZE 4 VAL_BATCHSIZE 4 DATASET.DATAROOT /home/master/10/cytseng/data/sets/nuscenes N_WORKERS 16 ${PY_ARGS}
