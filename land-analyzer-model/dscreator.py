import os
import sys
import shutil
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
from glob import glob


def convert_dataset(src_root:str, dst_root:str, tile_size:int):
    sample_paths = sorted(glob(os.path.join(src_root, '*')))

    for sample_path in tqdm(sample_paths):
        sample_name = os.path.basename(os.path.normpath(sample_path))

        # Read image
        img_paths = glob(os.path.join(sample_path, 'img/*'))
        if len(img_paths) != 1:
            print("Found more than 1 images in '{0}'. Skipped.".format(sample_path))
            continue
   
        image_orig = cv.imread(img_paths[0])
        if image_orig is None:
            print("Failed to read image '{0}'. Skipped.".format(img_paths[0]))
            continue

        # Read 'river' mask
        river_paths = glob(os.path.join(sample_path, 'mask_river/*'))
        if len(river_paths) != 1:
            print("Found more than 1 river masks in '{0}'. Skipped.".format(sample_path))
            continue

        river_orig = cv.imread(river_paths[0])
        if river_orig is None:
            print("Failed to read river mask '{0}'. Skipped.".format(river_paths[0]))
            continue

        # Read 'water' mask
        water_paths = glob(os.path.join(sample_path, 'mask_water/*'))
        if len(water_paths) != 1:
            print("Found more than 1 water masks in '{0}'. Skipped.".format(sample_path))
            continue

        water_orig = cv.imread(water_paths[0])
        if water_orig is None:
            print("Failed to read water mask '{0}'. Skipped.".format(water_paths[0]))
            continue

        # Check shapes
        if not (image_orig.shape == river_orig.shape == water_orig.shape):
            print("Shapes of images are inconsistent in '{0}'. Skipped.".format(sample_path))
            continue

        # Prepare expanded sources
        height_exp = ((image_orig.shape[0] // tile_size) + 1) * tile_size
        width_exp = ((image_orig.shape[1] // tile_size) + 1) * tile_size
        image_exp = np.zeros((height_exp, width_exp, 3), np.uint8)
        image_exp[:image_orig.shape[0], :image_orig.shape[1]] = image_orig
        river_exp = np.zeros((height_exp, width_exp, 3), np.uint8)
        river_exp[:river_orig.shape[0], :river_orig.shape[1]] = river_orig
        water_exp = np.zeros((height_exp, width_exp, 3), np.uint8)
        water_exp[:water_orig.shape[0], :water_orig.shape[1]] = water_orig

        # Prepare output dirs
        output_dir = os.path.join(dst_root, sample_name)
        
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

        output_dir_img = os.path.join(output_dir, 'img')
        output_dir_river = os.path.join(output_dir, 'river')
        output_dir_water = os.path.join(output_dir, 'water')
        os.makedirs(output_dir_img, exist_ok=True)
        os.makedirs(output_dir_river, exist_ok=True)
        os.makedirs(output_dir_water, exist_ok=True)

        # Split sources to tiles
        saved_counter = 0
        for tile_y_idx, y_offset in enumerate(range(0, height_exp, tile_size)):
            for tile_x_idx, x_offset in enumerate(range(0, width_exp, tile_size)):

                image_tile = image_exp[y_offset:y_offset + tile_size, x_offset:x_offset + tile_size]

                # Check if image empty
                black_count = np.count_nonzero(image_tile <= 0) / (image_tile.size)
                white_count = np.count_nonzero(image_tile >= 255) / (image_tile.size)
                if black_count > 0.5 or white_count > 0.5:
                    continue

                image_filename = os.path.join(output_dir_img, f'img_{tile_y_idx}_{tile_x_idx}.png')
                cv.imwrite(image_filename, image_tile)

                river_tile = river_exp[y_offset:y_offset + tile_size, x_offset:x_offset + tile_size]
                river_filename = os.path.join(output_dir_river, f'river_{tile_y_idx}_{tile_x_idx}.png')
                cv.imwrite(river_filename, river_tile)

                water_tile = water_exp[y_offset:y_offset + tile_size, x_offset:x_offset + tile_size]
                water_filename = os.path.join(output_dir_water, f'water_{tile_y_idx}_{tile_x_idx}.png')
                cv.imwrite(water_filename, water_tile)

                saved_counter += 1

        if saved_counter < 1:
            shutil.rmtree(output_dir)


def main() -> int:
    # Create argument parser
    parser = argparse.ArgumentParser(description='Dataset creator for training')
    parser.add_argument('--src-root', dest='src_root', required=True, type=str, default="",
                        help="Path to dataset root directory")
    parser.add_argument('--dst-root', dest='dst_root', required=True, type=str, default="",
                        help="Path where to save converted dataset")
    parser.add_argument('--tile-size', dest='tile_size', required=False, type=int, default=512,
                        help="Size of tile to split source image")
    args = parser.parse_args()

    # Check input directory
    if not os.path.isdir(args.src_root):
        print("Failed to check input '{0}' directory. Abort.".format(args.src_root))
        return -1

    # Convert dataset
    convert_dataset(args.src_root, args.dst_root, args.tile_size)

    print('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())