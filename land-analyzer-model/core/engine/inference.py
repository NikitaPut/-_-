import torch
import logging
import numpy as np
from tqdm import tqdm
from core.engine import losses as mylosses

def eval_dataset(cfg, model, data_loader, device, model_type='pytorch'):
    logger = logging.getLogger("CORE.inference")

    # Create losses
    criterion_logloss = mylosses.LogLoss(reduction=False)
    criterion_jaccard = mylosses.JaccardIndex(reduction=False)
    criterion_dice = mylosses.DiceLoss(reduction=False)

    stats = {
        'sample_count': 0.0,
        'loss': 0.0,
        'jaccard': 0.0,
        'dice': 0.0
    }

    for data_entry in tqdm(data_loader):
        images, labels, masks = data_entry

        # Forward images
        with torch.no_grad():
            # B,C,H,W = images.shape
            if model_type == 'onnx':
                images_np = images.numpy().astype(np.float32)
                outputs = model.forward(images_np, preprocess=False, postprocess=False)
                outputs = torch.from_numpy(outputs).to(device)
            elif model_type == 'tensorrt':
                images_np = images.numpy().astype(np.float32)
                outputs = model.forward(images_np, preprocess=False, postprocess=False)
                outputs = torch.from_numpy(outputs).to(device) 
            elif model_type == 'pytorch':
                images = images.to(device)
                outputs = model(images)
            else:
                logger.error("Unknown model type: %s. Aborting...".format(model_type))
                return -1

        # Calculate losses
        targets = labels.to(device)
        masks = masks.to(device)

        losses = criterion_logloss.forward(outputs, targets, masks)
        outputs_binarized = torch.threshold(outputs, cfg.TENSORBOARD.METRICS_BIN_THRESHOLD, 0.0)
        jaccard_losses = criterion_jaccard.forward(outputs_binarized, targets)
        dice_losses = criterion_dice.forward(outputs_binarized, targets)

        # Reduce loss (mean)
        stats['loss'] += torch.mean(losses).item()
        stats['jaccard'] += torch.mean(jaccard_losses).item()
        stats['dice'] += torch.mean(dice_losses).item()
        stats['sample_count'] += 1

    # Return results
    stats['loss'] /= stats['sample_count']
    stats['jaccard'] /= stats['sample_count']
    stats['dice'] /= stats['sample_count']

    result_dict = {
        'loss': stats['loss'],
        'jaccard': stats['jaccard'],
        'dice': stats['dice']
    }

    return result_dict