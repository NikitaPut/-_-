from yacs.config import CfgNode as CN

_CFG = CN()

_CFG.MODEL = CN()
_CFG.MODEL.META_ARCHITECTURE = 'MulticlassSegmentator'
_CFG.MODEL.DEVICE = "cpu"
# match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
# _CFG.MODEL.THRESHOLD = 0.5
# _CFG.MODEL.NUM_CLASSES = 2
# Hard negative mining
# _CFG.MODEL.NEG_POS_RATIO = 3
# _CFG.MODEL.CENTER_VARIANCE = 0.1
# _CFG.MODEL.SIZE_VARIANCE = 0.2

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
_CFG.MODEL.BACKBONE = CN()
_CFG.MODEL.BACKBONE.NAME = 'YOLOs'
_CFG.MODEL.BACKBONE.PRETRAINED = True
_CFG.MODEL.BACKBONE.FREEZE = True

# ---------------------------------------------------------------------------- #
# Head
# ---------------------------------------------------------------------------- #
_CFG.MODEL.HEAD = CN()
_CFG.MODEL.HEAD.NAME = 'DeepLabV3_Custom'
_CFG.MODEL.HEAD.INPUT_DEPTH = [64, 128, 256, 384] # YOLOs: [64, 128, 256, 384], MNETV2: [24, 32, 96, 320]
_CFG.MODEL.HEAD.HIDDEN_DEPTH = 64
_CFG.MODEL.HEAD.PRETRAINED = True
_CFG.MODEL.HEAD.FREEZE = True
_CFG.MODEL.HEAD.DROPOUT = 0.5
_CFG.MODEL.HEAD.CLASS_LABELS = []

# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_CFG.INPUT = CN()
_CFG.INPUT.IMAGE_SIZE = [512, 512]
_CFG.INPUT.PIXEL_MEAN = [123, 117, 104] # R-G-B
_CFG.INPUT.PIXEL_STD = [58, 57, 57] # R-G-B

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_CFG.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_CFG.DATASETS.TRAIN = CN()
_CFG.DATASETS.TRAIN.ROOT_DIR = []
# List of the dataset names for testing, as present in paths_catalog.py
_CFG.DATASETS.TEST = CN()
_CFG.DATASETS.TEST.ROOT_DIR = []

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
# _CFG.LOSS = CN()
# _CFG.LOSS.NAMES = ["Log"]
# _CFG.LOSS.WEIGHT = [1.0]
# _CFG.LOSS.CLEAN_WEIGHT = [0.9]
# _CFG.LOSS.UNC_WEIGHT = [0.9]
# _CFG.LOSS.STRONG_WEIGHT = [0.9]

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_CFG.DATA_LOADER = CN()
# Number of data loading threads
_CFG.DATA_LOADER.NUM_WORKERS = 1
_CFG.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_CFG.SOLVER = CN()
# train configs
_CFG.SOLVER.TYPE = 'Adam'
_CFG.SOLVER.MAX_ITER = 2048
# _CFG.SOLVER.LR_STEPS = [80000, 100000]
# _CFG.SOLVER.GAMMA = 0.1
_CFG.SOLVER.BATCH_SIZE = 32
_CFG.SOLVER.LR = 1e-3
# _CFG.SOLVER.MIN_LR_RATIO = 0.1
# _CFG.SOLVER.MOMENTUM = 0.9
# _CFG.SOLVER.WEIGHT_DECAY = 5e-4
# _CFG.SOLVER.WARMUP_FACTOR = 1e-2
# _CFG.SOLVER.WARMUP_ITERS = 500
_CFG.SOLVER.LR_LAMBDA = 0.95

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_CFG.TEST = CN()
_CFG.TEST.BATCH_SIZE = 8

# ---------------------------------------------------------------------------- #
# Output options
# ---------------------------------------------------------------------------- #
_CFG.OUTPUT_DIR = 'outputs/test'

# ---------------------------------------------------------------------------- #
# Tensorboard
# ---------------------------------------------------------------------------- #
_CFG.TENSORBOARD = CN()
_CFG.TENSORBOARD.BEST_SAMPLES_NUM = 32
_CFG.TENSORBOARD.WORST_SAMPLES_NUM = 32
_CFG.TENSORBOARD.METRICS_BIN_THRESHOLD = 0.85