# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from util.infomae_utils import apply_adaptive_masking


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    surprisal_cache=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        # InfoMAE: Handle adaptive masking
        adaptive_mask = None
        current_surprisal = None

        if args.adaptive_masking and surprisal_cache is not None:
            # Get cached surprisal for adaptive masking
            cached_surprisal, available_mask = surprisal_cache.get_surprisal(targets)

            if cached_surprisal is not None:
                # Apply adaptive masking based on cached surprisal
                adaptive_mask = apply_adaptive_masking(
                    args.mask_ratio, cached_surprisal.to(device),
                    args.adaptive_alpha, args.adaptive_gamma
                )
                # For SWA, use cached surprisal as attention modulation
                if args.use_surprisal_attention:
                    current_surprisal = cached_surprisal.to(device)

        with torch.cuda.amp.autocast():
            if hasattr(model, 'forward') and 'adaptive_mask' in model.forward.__code__.co_varnames:
                # InfoMAE model with extended forward signature
                loss, pred, mask, surprisal = model(
                    samples,
                    mask_ratio=args.mask_ratio if adaptive_mask is None else 0.0,
                    adaptive_mask=adaptive_mask,
                    surprisal=current_surprisal,
                    beta_ib=args.beta_ib
                )
            else:
                # Standard MAE model
                loss, pred, mask = model(samples, mask_ratio=args.mask_ratio)
                surprisal = None

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # InfoMAE: Update surprisal cache (only if not read-only mode)
        if surprisal is not None and surprisal_cache is not None and (data_iter_step + 1) % accum_iter == 0 and not args.read_only_cache:
            surprisal_cache.update_surprisal(targets, surprisal.cpu(), mask.cpu())

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}