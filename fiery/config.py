import argparse
import logging
import os
from typing import Any, Dict
import yaml
from fvcore.common.config import CfgNode as _CfgNode


def convert_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary."""
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    if not isinstance(cfg_node, _CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print(
                'Key {} with value {} is not a valid type; valid types: {}'.format(
                    '.'.join(key_list), type(cfg_node), _VALID_TYPES
                ),
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


class CfgNode(_CfgNode):
    """Remove once https://github.com/rbgirshick/yacs/issues/19 is merged."""
    BASE_KEY = "_BASE_"

    @classmethod
    def load_yaml_with_base(cls, filename: str, allow_unsafe: bool = False) -> None:
        """
        Just like `yaml.load(open(filename))`, but inherit attributes from its
            `_BASE_`.

        Args:
            filename (str or file-like object): the file name or file of the current config.
                Will be used to find the base config file.
            allow_unsafe (bool): whether to allow loading the config file with
                `yaml.unsafe_load`.

        Returns:
            (dict): the loaded yaml
        """
        with cls._open_cfg(filename) as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                if not allow_unsafe:
                    raise
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Loading config {} with yaml.unsafe_load. Your machine may "
                    "be at risk if the file contains malicious content.".format(
                        filename
                    )
                )
                f.close()
                with cls._open_cfg(filename) as f:
                    cfg = yaml.unsafe_load(f)  # pyre-ignore

        # pyre-ignore
        def merge_a_into_b(a: Dict[Any, Any], b: Dict[Any, Any]) -> None:
            # merge dict a into dict b. values in a will overwrite b.
            for k, v in a.items():
                if isinstance(v, dict) and k in b:
                    assert isinstance(
                        b[k], dict
                    ), "Cannot inherit key '{}' from base!".format(k)
                    merge_a_into_b(v, b[k])
                else:
                    b[k] = v

        if cls.BASE_KEY in cfg:
            base_cfg_files = cfg[cls.BASE_KEY]
            del cfg[cls.BASE_KEY]
            if isinstance(base_cfg_files, str):
                base_cfg_files = [base_cfg_files]

            merged_cfg = CfgNode(new_allowed=True)

            for base_cfg_file in base_cfg_files:
                if base_cfg_file.startswith("~"):
                    base_cfg_file = os.path.expanduser(base_cfg_file)
                if not any(map(base_cfg_file.startswith, ["/", "https://", "http://"])):
                    # the path to base cfg is relative to the config file itself.
                    base_cfg_file = os.path.join(os.path.dirname(filename), base_cfg_file)
                base_cfg = cls.load_yaml_with_base(base_cfg_file, allow_unsafe=allow_unsafe)

                merge_a_into_b(base_cfg, merged_cfg)
            merge_a_into_b(cfg, merged_cfg)
            return merged_cfg
        return cfg

    def convert_to_dict(self):
        return convert_to_dict(self)


CN = CfgNode

_C = CN()
_C.LOG_DIR = 'tensorboard_logs'
_C.EVA_DIR = 'output_dir'

_C.TAG = 'lss'

_C.GPUS = [0]  # which gpus to use
_C.PRECISION = 32  # 16bit or 32bit
_C.BATCHSIZE = 3
_C.VAL_BATCHSIZE = 16
_C.EPOCHS = 100
_C.EVALUATION = False
_C.CKPT_PATH = None
_C.TEST_TRAINSET = False

_C.N_WORKERS = 5
_C.VIS_INTERVAL = 5000
_C.LOGGING_INTERVAL = 5000
_C.VALID_FREQ = 0.5
_C.WEIGHT_SUMMARY = "top"

_C.PRETRAINED = CN()
_C.PRETRAINED.LOAD_WEIGHTS = False
_C.PRETRAINED.PATH = ''

_C.DATASET = CN()
_C.DATASET.DATAROOT = '/home/master/10/cytseng/data/sets/nuscenes/'
_C.DATASET.VERSION = 'v1.0-trainval'
_C.DATASET.NAME = 'nuscenes'
_C.DATASET.IGNORE_INDEX = 255  # Ignore index when creating flow/offset labels
_C.DATASET.FILTER_INVISIBLE_VEHICLES = True  # Filter vehicles that are not visible from the cameras
_C.DATASET.TRAINING_SAMPLES = -1
_C.DATASET.VALIDATING_SAMPLES = -1
_C.DATASET.INCLUDE_VELOCITY = False

_C.TIME_RECEPTIVE_FIELD = 3  # how many frames of temporal context (1 for single timeframe)
_C.N_FUTURE_FRAMES = 4  # how many time steps into the future to predict

_C.IMAGE = CN()
_C.IMAGE.FINAL_DIM = [224, 480]
_C.IMAGE.RESIZE_SCALE = 0.3
_C.IMAGE.TOP_CROP = 46
_C.IMAGE.ORIGINAL_HEIGHT = 900  # Original input RGB camera height
_C.IMAGE.ORIGINAL_WIDTH = 1600  # Original input RGB camera width
_C.IMAGE.NAMES = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
_C.IMAGE.N_CAMERA = 6

_C.LIFT = CN()  # image to BEV lifting
_C.LIFT.X_BOUND = [-40.0, 40.0, 0.5]  #  Forward
_C.LIFT.Y_BOUND = [-40.0, 40.0, 0.5]  # Sides
_C.LIFT.Z_BOUND = [-10.0, 10.0, 20.0]  # Height
_C.LIFT.D_BOUND = [2.0, 50.0, 1.0]

_C.MODEL = CN()

_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.DOWNSAMPLE = 8
_C.MODEL.ENCODER.NAME = 'efficientnet-b4'
_C.MODEL.ENCODER.OUT_CHANNELS = 64
_C.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION = True

_C.MODEL.TEMPORAL_MODEL = CN()
_C.MODEL.TEMPORAL_MODEL.NAME = 'temporal_block'  # type of temporal model
_C.MODEL.TEMPORAL_MODEL.START_OUT_CHANNELS = 64
_C.MODEL.TEMPORAL_MODEL.EXTRA_IN_CHANNELS = 0
_C.MODEL.TEMPORAL_MODEL.INBETWEEN_LAYERS = 0
_C.MODEL.TEMPORAL_MODEL.PYRAMID_POOLING = True
_C.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE = True

_C.MODEL.DISTRIBUTION = CN()
_C.MODEL.DISTRIBUTION.LATENT_DIM = 32
_C.MODEL.DISTRIBUTION.MIN_LOG_SIGMA = -5.0
_C.MODEL.DISTRIBUTION.MAX_LOG_SIGMA = 5.0

_C.MODEL.FUTURE_PRED = CN()
_C.MODEL.FUTURE_PRED.N_GRU_BLOCKS = 3
_C.MODEL.FUTURE_PRED.N_RES_LAYERS = 3

_C.MODEL.DECODER = CN()

_C.MODEL.MM = CN()

_C.MODEL.MM.SEG_CAT_BACKBONE = False
_C.MODEL.MM.SEG_ADD_BACKBONE = False
_C.MODEL.MM.HEAD_MAPPING = CN(new_allowed=True)
_C.MODEL.MM.BBOX_BACKBONE = CN(new_allowed=True)
_C.MODEL.MM.BBOX_NECK = CN(new_allowed=True)
_C.MODEL.MM.BBOX_HEAD = CN(new_allowed=True)

_C.MODEL.BN_MOMENTUM = 0.1
_C.MODEL.SUBSAMPLE = False  # Subsample frames for Lyft

_C.SEMANTIC_SEG = CN()
_C.SEMANTIC_SEG.WEIGHTS = [1.0, 2.0]  # per class cross entropy weights (bg, dynamic, drivable, lane)
_C.SEMANTIC_SEG.USE_TOP_K = True  # backprop only top-k hardest pixels
_C.SEMANTIC_SEG.TOP_K_RATIO = 0.25
_C.SEMANTIC_SEG.NUSCENE_CLASS = False

_C.INSTANCE_SEG = CN()

_C.INSTANCE_FLOW = CN()
_C.INSTANCE_FLOW.ENABLED = True

_C.PROBABILISTIC = CN()
_C.PROBABILISTIC.ENABLED = True  # learn a distribution over futures
_C.PROBABILISTIC.WEIGHT = 100.0
_C.PROBABILISTIC.FUTURE_DIM = 6  # number of dimension added (future flow, future centerness, offset, seg)

_C.FUTURE_DISCOUNT = 0.95

_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = 'AdamW'
_C.OPTIMIZER.LR = 3e-4
_C.OPTIMIZER.WEIGHT_DECAY = 1e-7
_C.GRAD_NORM_CLIP = 5

_C.LOSS = CN()
_C.LOSS.SEG_USE = False

_C.LOSS.SEG_LOSS_WEIGHT = CN()
_C.LOSS.SEG_LOSS_WEIGHT.ALL = 0.5
_C.LOSS.SEG_LOSS_WEIGHT.SEG = 0.25
_C.LOSS.SEG_LOSS_WEIGHT.CENTER = 0.25
_C.LOSS.SEG_LOSS_WEIGHT.OFFSET = 0.25
_C.LOSS.SEG_LOSS_WEIGHT.PROBA = 0.25

_C.LOSS.OBJ_LOSS_WEIGHT = CN()
_C.LOSS.OBJ_LOSS_WEIGHT.ALL = 0.5
_C.LOSS.OBJ_LOSS_WEIGHT.SCORE = 0.25
_C.LOSS.OBJ_LOSS_WEIGHT.POS = 0.25
_C.LOSS.OBJ_LOSS_WEIGHT.DIM = 0.25
_C.LOSS.OBJ_LOSS_WEIGHT.ANG = 0.25

_C.OBJ = CN()
_C.OBJ.N_CLASSES = 1
_C.OBJ.HEAD_NAME = 'mm'


def get_parser():
    parser = argparse.ArgumentParser(description='Fiery training')
    # TODO: remove below?
    parser.add_argument('--config-file', default='', metavar='FILE', help='path to config file')
    parser.add_argument('--eval-path', help='eval model path')
    parser.add_argument(
        'opts', help='Modify config options using the command-line', default=None, nargs=argparse.REMAINDER,
    )
    return parser


def get_cfg(args=None, cfg_dict=None):
    """ First get default config. Then merge cfg_dict. Then merge according to args. """

    cfg = _C.clone()

    if cfg_dict is not None:
        cfg.merge_from_other_cfg(CfgNode(cfg_dict))
    cfg.MODEL.MM.HEAD_MAPPING.merge_from_other_cfg(CfgNode({'Anchor3DHeadWrapper': 'pp', 'CenterHeadWrapper': 'cp'}, new_allowed=True))
    if args is not None:
        if args.config_file:
            cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
    return cfg
