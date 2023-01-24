import os
import torch
import logging
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
from core.config import cfg
from core.utils.logger import setup_logger
from core.utils import dist_util
from core.modelling.model import build_model
from core.utils.checkpoint import CheckPointer
from core.data.transforms import transforms2


def load_model(cfg, args):
    logger = logging.getLogger('INFER')

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
    parser.add_argument('--cfg', dest='config_file', required=True, type=str, default="", metavar="FILE",
                        help="path to config file")
    parser.add_argument('--i', dest='input_img', required=True, type=str, default="", metavar="FILE",
                        help="path to input image")
    parser.add_argument('--o', dest='output_img', required=True, type=str, default="", metavar="FILE",
                        help="path to output image")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()

    # Create config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Create logger
    logger = setup_logger("INFER", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))

    # Create model
    model = load_model(cfg, args)
    model.eval()

    # Create device
    device = torch.device(cfg.MODEL.DEVICE)

    # Read input image
    image = cv.imread(args.input_img)
    if image is None:
        logger.error("Failed to read input image file")

    # Prepare expanded sources
    tile_size = cfg.INPUT.IMAGE_SIZE[0] # TODO: w, h
    result_img_height = ((image.shape[0] // tile_size) + 1) * tile_size
    result_img_width = ((image.shape[1] // tile_size) + 1) * tile_size
    image_expanded = np.zeros((result_img_height, result_img_width, 3), np.uint8)
    image_expanded[:image.shape[0], :image.shape[1]] = image

    # Prepare result mask images
    result_masks = {}
    for cname in cfg.MODEL.HEAD.CLASS_LABELS:
        result_masks[cname] = np.zeros((result_img_height, result_img_width, 3), np.float64)

    # Infer model
    pbar = tqdm(total=((image.shape[0] // tile_size) + 1)*((image.shape[1] // tile_size) + 1))

    for y_offset in range(0, image_expanded.shape[0], tile_size):
        for x_offset in range(0, image_expanded.shape[1], tile_size):

            image_tile = image_expanded[y_offset:y_offset + tile_size, x_offset:x_offset + tile_size]

            with torch.no_grad():
                input, _, _ = transforms2.ConvertFromInts()(image_tile)
                input, _, _ = transforms2.Normalize()(input)
                input, _, _ = transforms2.ToTensor()(input)

                input = input.unsqueeze(0)
                outputs = model(input.to(device))
                outputs = torch.sigmoid(outputs)

                for cid, cname in enumerate(result_masks):
                    outmask = outputs[:, cid, :, :]
                    out_img = transforms2.ToCV2Image()(outmask)
                    result_tile = result_masks[cname][y_offset:y_offset + tile_size, x_offset:x_offset + tile_size]
                    result_tile += out_img * 255.0

                pbar.update(1)
    pbar.close()

    # Save result mask
    for cname, mask in result_masks.items():
        if mask is not None:
            mask = mask[0:image.shape[0], 0:image.shape[1]]
            mask = mask.astype(np.uint8)
            result = cv.addWeighted(image, 0.25, mask, 0.75, 1.0)
            filepath = os.path.abspath(args.output_img)
            filepath, ext = os.path.splitext(filepath)
            filepath = filepath + "_" + cname + ext
            print("Saving mask for '{0}' class to '{1}'...".format(cname, filepath))
            cv.imwrite(filepath, result)

    print("Done.")
    return 0


if __name__ == '__main__':
    exit(main())
