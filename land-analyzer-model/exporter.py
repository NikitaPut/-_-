import os
import torch
import argparse
import logging
import onnxruntime as ort
from core.config import cfg
from core.utils.logger import setup_logger
from core.utils import dist_util
from core.modelling.model import build_model
from core.data import make_data_loader
from core.utils.checkpoint import CheckPointer


def load_model(cfg, args):
    logger = logging.getLogger('DIRT')

    # Create device
    device = torch.device(cfg.MODEL.DEVICE)

    # Create model
    model = build_model(cfg)
    model.to(device)

    optimizer = None
    scheduler = None

    # Create checkpointer
    arguments = {"epoch": 0}
    save_to_disk = dist_util.get_rank() == 0
    checkpointer = CheckPointer(model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger)
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    return model


def main() -> int:
    # Create argument parser
    parser = argparse.ArgumentParser(description='Dirt Blockage Export With PyTorch')
    parser.add_argument('--config-file', dest='config_file', required=True, type=str, default="", metavar="FILE",
                        help="path to config file")
    parser.add_argument('--batch-size', dest='batch_size', required=False, type=int, default=1,
                        help="Batch size to export")
    parser.add_argument('--target', dest="target", choices=['ti', 'tensorrt'],
                        help='Target platform')
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
    cfg.DATA_LOADER.NUM_WORKERS = 1
    cfg.TEST.BATCH_SIZE = args.batch_size
    cfg.freeze()

    # Create logger
    logger = setup_logger("DIRT", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(NUM_GPUS))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))

    # Create output export dir
    folder_path = os.path.join(cfg.OUTPUT_DIR, "export/onnx/")
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    # Check target specific
    if (args.target == 'ti'):
        onnx_opset = 11
        print()
    elif (args.target == 'tensorrt'):
        onnx_opset = 10
        print()
    else:
        print("Unsupported target: {}".format(args.target))
        return -1

    # Do export
    with torch.no_grad():
        device = torch.device(cfg.MODEL.DEVICE)
        data_loaders_val = make_data_loader(cfg, is_train=False)

        model = load_model(cfg, args)
        model.eval()
        model.export_rebuild(args.target)

        torch.cuda.empty_cache()
        for id, data_entry in enumerate(data_loaders_val):
            images, coefs, flowmaps = data_entry

            print(images.shape)

            input_names = ["0"]
            output_names = ["out"]
    
            torch.onnx.export(model, images.to(device), os.path.join(folder_path, "model.onnx"), verbose=False, 
                    output_names=output_names, opset_version=onnx_opset, keep_initializers_as_inputs = False)
            break

        so = ort.SessionOptions()
        new_name = os.path.join(folder_path, "imp_model.onnx")
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.optimized_model_filepath = new_name
        session = ort.InferenceSession(os.path.join(folder_path, "model.onnx"), so)

    return 0

if __name__ == '__main__':
    exit(main())