import os
import torch
import argparse
import logging
from core.config import cfg
from core.utils.filesystem import getListOfFiles
from core.utils.logger import setup_logger
from core.utils import dist_util
from core.modelling.model import build_model
from core.data import make_data_loader
from core.solver import make_optimizer, make_lr_scheduler
from core.utils.checkpoint import CheckPointer
from core.engine.train import do_train
from core.data.datasets import build_dataset
from core.data.transforms import build_transforms


def train_model(cfg, args):
    logger = logging.getLogger('CORE')

    # Create device
    device = torch.device(cfg.MODEL.DEVICE)

    # Create model
    model = build_model(cfg)
    model.to(device)

    # Create training data
    data_loader = make_data_loader(cfg, is_train=True)

    # transforms = build_transforms(cfg.INPUT.IMAGE_SIZE, is_train=True, to_tensor=False)
    # dataset = build_dataset(cfg.DATASETS.TRAIN.ROOT_DIR[0], cfg.MODEL.HEAD.CLASS_LABELS, transforms)
    # dataset.visualize(500)

    # Create optimizer
    lr = cfg.SOLVER.LR * args.num_gpus  # scale by num gpus
    optimizer = make_optimizer(cfg, lr, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Create checkpointer
    arguments = {"epoch": 0}
    save_to_disk = dist_util.is_main_process()
    checkpointer = CheckPointer(model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger)
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    # Train model
    model = do_train(cfg, model, data_loader, optimizer, scheduler, checkpointer, device, arguments, args)

    return model


def str2bool(s):
    return s.lower() in ('true', '1')


def main() -> int:
    # Create argument parser
    parser = argparse.ArgumentParser(description='Semantic Segmentation Model Training With PyTorch')
    parser.add_argument("--config-file", dest="config_file", required=True, type=str, default="", metavar="FILE",
                        help="path to config file")
    parser.add_argument('--log-step', dest="log_step", required=False, type=int, default=1,
                        help='Print logs every log_step')
    parser.add_argument('--save-step', dest="save_step", required=False, type=int, default=1,
                        help='Save checkpoint every save_step')
    parser.add_argument('--eval-step', dest="eval_step", required=False, type=int, default=1,
                        help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--use-tensorboard', dest="use_tensorboard", required=False, default=False, type=str2bool,
                        help='Use tensorboard summary writer')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()
    NUM_GPUS = 1
    args.distributed = False
    args.num_gpus = NUM_GPUS

    # Enable cudnn auto-tuner to find the best algorithm to use for your hardware.
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Create config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Create output directory
    if os.path.exists(cfg.OUTPUT_DIR):
        files = getListOfFiles(cfg.OUTPUT_DIR)
        if files:
            print("Output path '{0}' already exists! Found {1} files.".format(cfg.OUTPUT_DIR, len(files)))
            # logger.error("Output path '{0}' already exists! Found {1} files.".format(cfg.OUTPUT_DIR, len(files)))
            # return -1
    else:
        os.makedirs(cfg.OUTPUT_DIR)

    # Create logger
    logger = setup_logger("CORE", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(NUM_GPUS))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))

    # Create config backup
    with open(os.path.join(cfg.OUTPUT_DIR, 'cfg.yaml'), "w") as cfg_dump:
        cfg_dump.write(str(cfg))

    # Train model
    model = train_model(cfg, args)

    return 0

if __name__ == '__main__':
    exit(main())