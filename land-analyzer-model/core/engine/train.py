import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from core.utils import dist_util
from torch.utils.tensorboard import SummaryWriter
from .inference import eval_dataset
from core.data import make_data_loader
from core.engine import losses as mylosses
from torchvision.utils import make_grid

def do_eval(cfg, model, distributed, **kwargs):
    torch.cuda.empty_cache()

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    data_loader = make_data_loader(cfg, False)

    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)

    result_dict = eval_dataset(cfg, model, data_loader, device, 'pytorch')

    torch.cuda.empty_cache()
    return result_dict


def do_train(cfg,
             model,
             data_loader,
             optimizer,
             scheduler,
             checkpointer,
             device,
             arguments,
             args):

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    logger = logging.getLogger("CORE")
    logger.info("Start training ...")

    # Set model to train mode
    model.train()

    # Create tensorboard writer
    save_to_disk = dist_util.is_main_process()
    if args.use_tensorboard and save_to_disk:
        summary_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    # Prepare to train
    iters_per_epoch = len(data_loader)
    total_steps = iters_per_epoch * cfg.SOLVER.MAX_ITER
    start_epoch = arguments["epoch"]
    logger.info("Iterations per epoch: {0}. Total steps: {1}. Start epoch: {2}".format(iters_per_epoch, total_steps, start_epoch))

    # Create losses
    criterion_logloss = mylosses.LogLoss(reduction=False)
    criterion_jaccard = mylosses.JaccardIndex(reduction=False)
    criterion_dice = mylosses.DiceLoss(reduction=False)

    # Epoch loop
    for epoch in range(start_epoch, cfg.SOLVER.MAX_ITER):
        arguments["epoch"] = epoch + 1

        # Create progress bar
        print(('\n' + '%10s' * 6) % ('Epoch', 'gpu_mem', 'lr', 'loss', 'jaccard', 'dice'))
        pbar = enumerate(data_loader)
        pbar = tqdm(pbar, total=len(data_loader))

        # Prepare data for tensorboard
        best_samples, worst_samples = {}, {}
        for cname in cfg.MODEL.HEAD.CLASS_LABELS:
            best_samples[cname] = []
            worst_samples[cname] = []

        # Iteration loop
        loss_sum, jaccard_loss_sum, dice_loss_sum = 0.0, 0.0, 0.0

        for iteration, data_entry in pbar:
            global_step = epoch * iters_per_epoch + iteration

            images, labels, masks = data_entry

            # Forward data to GPU
            images = images.to(device)
            targets = labels.to(device)
            masks = masks.to(device)

            # Do prection
            outputs = model(images)
            outputs = torch.sigmoid(outputs)

            # Calculate losses
            losses = criterion_logloss.forward(outputs, targets, masks)
            outputs_binarized = torch.threshold(outputs, cfg.TENSORBOARD.METRICS_BIN_THRESHOLD, 0.0)
            jaccard_losses = criterion_jaccard.forward(outputs_binarized, targets)
            dice_losses = criterion_dice.forward(outputs_binarized, targets)

            ################### Best images
            with torch.no_grad():
                losses_ = jaccard_losses.detach().clone()
                total_pix = torch.numel(labels[0, 0, :, :])

                for ch_id, cname in enumerate(best_samples.keys()):
                    # Select only non-empty labels
                    mask_area = torch.sum(labels[:, ch_id, :, :], dim=(1,2)) / (total_pix + 1e-6)
                    idxs = torch.tensor([i for i,v in enumerate(mask_area) if not torch.isclose(v, torch.tensor(0.0))])
                    if not torch.numel(idxs):
                        continue

                    losses_selected = torch.index_select(losses_[:, ch_id].to('cpu'), 0, idxs)
                    labels_selected = torch.index_select(labels[:, ch_id, :, :], 0, idxs)
                    outputs_selected = torch.index_select(outputs[:, ch_id, :, :].detach().to('cpu'), 0, idxs)
                    images_selected = torch.index_select(images.detach().to('cpu'), 0, idxs)

                    # Find maximum metric
                    losses_per_image = losses_selected
                    max_idx = torch.argmax(losses_per_image).item()

                    # Prepare sample images
                    best_loss = losses_per_image[max_idx].item()

                    best_label = labels_selected[max_idx, :, :]
                    best_label = torch.stack([best_label, best_label, best_label], dim=0) # to 3-channel image

                    best_output = outputs_selected[max_idx, :, :]
                    best_output = torch.stack([best_output, best_output, best_output], dim=0) # to 3-channel image

                    # Save image
                    save_img = torch.cat([images_selected[max_idx][0:3,:,:].detach().to('cpu'), best_label, best_output], dim=2)

                    if len(best_samples[cname]) >= cfg.TENSORBOARD.BEST_SAMPLES_NUM:
                        min_id = min(range(len(best_samples[cname])), key=lambda x : best_samples[cname][x][0])
                        min_loss = best_samples[cname][min_id][0]
                        if best_loss > min_loss:
                            del best_samples[cname][min_id]
                            best_samples[cname].append((best_loss, save_img))
                    else:
                        best_samples[cname].append((best_loss, save_img))
            ###############################

            ################### Worst images
            with torch.no_grad():
                losses_ = jaccard_losses.detach().clone()
                total_pix = torch.numel(labels[0, 0, :, :])

                for ch_id, cname in enumerate(best_samples.keys()):
                    # Select only non-empty labels
                    mask_area = torch.sum(labels[:, ch_id, :, :], dim=(1,2)) / (total_pix + 1e-6)
                    idxs = torch.tensor([i for i,v in enumerate(mask_area) if not torch.isclose(v, torch.tensor(0.0))])
                    if not torch.numel(idxs):
                        continue

                    losses_selected = torch.index_select(losses_[:, ch_id].to('cpu'), 0, idxs)
                    labels_selected = torch.index_select(labels[:, ch_id, :, :], 0, idxs)
                    outputs_selected = torch.index_select(outputs[:, ch_id, :, :].detach().to('cpu'), 0, idxs)
                    images_selected = torch.index_select(images.detach().to('cpu'), 0, idxs)

                    # Find minimum metric
                    losses_per_image = losses_selected
                    min_idx = torch.argmin(losses_per_image).item()

                    # Prepare sample images
                    worst_loss = losses_per_image[min_idx].item()

                    worst_label = labels_selected[min_idx, :, :]
                    worst_label = torch.stack([worst_label, worst_label, worst_label], dim=0) # to 3-channel image

                    worst_output = outputs_selected[min_idx, :, :]
                    worst_output = torch.stack([worst_output, worst_output, worst_output], dim=0) # to 3-channel image

                    # Save image
                    save_img = torch.cat([images_selected[min_idx][0:3,:,:].detach().to('cpu'), worst_label, worst_output], dim=2)

                    if len(worst_samples[cname]) >= cfg.TENSORBOARD.WORST_SAMPLES_NUM:
                        max_id = max(range(len(worst_samples[cname])), key=lambda x : worst_samples[cname][x][0])
                        max_loss = worst_samples[cname][max_id][0]
                        if worst_loss < max_loss:
                            del worst_samples[cname][max_id]
                            worst_samples[cname].append((worst_loss, save_img))
                    else:
                        worst_samples[cname].append((worst_loss, save_img))
            ###############################

            # Reduce loss (mean)
            loss = torch.mean(losses)
            loss_sum += loss.item()
            jaccard_loss_sum += torch.mean(jaccard_losses).item()
            dice_loss_sum += torch.mean(dice_losses).item()

            # Do optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # Update progress bar
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0) # (GB)
            s = ('%10s' * 2 + '%10.4g' * 4) % (
                                                '%g/%g' % (epoch, cfg.SOLVER.MAX_ITER - 1),
                                                mem,
                                                optimizer.param_groups[0]['lr'],
                                                loss_sum / (iteration + 1),
                                                jaccard_loss_sum / (iteration + 1),
                                                dice_loss_sum / (iteration + 1))

            pbar.set_description(s)

        # scheduler.step()

        # Do evaluation
        if args.eval_step > 0 and epoch % args.eval_step == 0:
            print('\nEvaluation ...')
            result_dict = do_eval(cfg, model, distributed=args.distributed, iteration=global_step)
            print(('\n' + 'Evaluation results:' + '%10s' * 3) % ('loss', 'jaccard', 'dice'))
            print('                   ' + '%10.4g%10.4g%10.4g' % (result_dict['loss'], result_dict['jaccard'], result_dict['dice']))

            if summary_writer:
                summary_writer.add_scalar('losses/validation_loss', result_dict['loss'], global_step=global_step)
                summary_writer.add_scalar('metrics/validation_jaccard', result_dict['jaccard'], global_step=global_step)
                summary_writer.add_scalar('metrics/validation_dice', result_dict['dice'], global_step=global_step)
                summary_writer.flush()

            model.train()

        # Save epoch results
        if epoch % args.save_step == 0:
            checkpointer.save("model_{:06d}".format(global_step), **arguments)

            if summary_writer:
                with torch.no_grad():
                    # Best samples
                    for cname, samples in best_samples.items():
                        _, images_ = zip(*samples)
                        image_grid = torch.stack(images_, dim=0)
                        image_grid = make_grid(image_grid, nrow=1)
                        summary_writer.add_image('images/{0}/train_best_samples'.format(cname), image_grid, global_step=global_step)

                    # Worst samples
                    for cname, samples in worst_samples.items():
                        _, images_ = zip(*samples)
                        image_grid = torch.stack(images_, dim=0)
                        image_grid = make_grid(image_grid, nrow=1)
                        summary_writer.add_image('images/{0}/train_worst_samples'.format(cname), image_grid, global_step=global_step)

                    summary_writer.add_scalar('losses/loss', loss_sum / (iteration + 1), global_step=global_step)
                    # summary_writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], global_step=global_step)
                    summary_writer.add_scalar('metrics/jaccard', jaccard_loss_sum / (iteration + 1), global_step=global_step)
                    summary_writer.add_scalar('metrics/dice', dice_loss_sum / (iteration + 1), global_step=global_step)
                    summary_writer.flush()

    # Save final model
    checkpointer.save("model_final", **arguments)

    return model