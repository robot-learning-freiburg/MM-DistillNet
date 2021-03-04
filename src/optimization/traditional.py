# -*- coding: utf-8 -*-
"""Master Project -- Multi Modal Object Detection

This file contains general training mechanism for traditional training

"""
# --------------------------------------------------------------------
#                                   Imports
# --------------------------------------------------------------------

# General Inputs
import logging

import numpy as np
import random

# Local Imports
from src.utils.utils import (
    custom_collate_factory,
    logits_to_ground_truth,
)


import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data.distributed


from tqdm import tqdm


# --------------------------------------------------------------------
#                           Logger Configuration
# --------------------------------------------------------------------
# Logging
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
#                                 Methods
# --------------------------------------------------------------------


def train_traditional(
    train_set,
    model,
    optimizer,
    epoch,
    config,
    writer
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the train generator
    train_sampler = None
    if config['engine'] == 'DistributedDataParallel':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set,
        )
        generator = DataLoader(
            train_set,
            batch_size=config.getint('batch_size'),
            shuffle=(train_sampler is None),
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_factory(config),
            num_workers=config.getint('num_workers'),
            sampler=train_sampler
        )
    else:
        generator = DataLoader(
            train_set,
            batch_size=config.getint('batch_size'),
            shuffle=True,
            drop_last=True,
            collate_fn=custom_collate_factory(config),
            num_workers=config.getint('num_workers'),
        )
    num_iter_per_epoch = len(generator)

    logger.info(f"Traditional Training for {num_iter_per_epoch} iters")

    model.module.teacher_models.eval()
    model.module.student_model.train()

    if train_sampler is not None:
        train_sampler.set_epoch(epoch)


    for iter, batch in enumerate(tqdm(generator, desc=f"Epoch={epoch+1}")):

        # ==================Forward=================

        rgb, thermal, depth, audio, label, id = batch
        rgb = Variable(
            rgb.to(device),
            requires_grad=True
        ).to(device)
        if config.getboolean('use_thermal'):
            thermal = Variable(
                thermal.to(device),
                requires_grad=True
            ).to(device)
        if config.getboolean('use_depth'):
            depth = Variable(
                depth.to(device),
                requires_grad=True
            ).to(device)

        # Add in like a new modality on what is called label
        # and then audio is the sumation of both label + the current batch
        augment = False
        if config['train_method'] == 'traditional_nms_kdlist_augmented' and random.random() > max(0.5, (0.5 + 0.5 * (1 - epoch/50))):
            augment = True
            label, audio = train_set.yield_batch(audio.shape[0], id)

        audio = Variable(
            audio.to(device),
            requires_grad=True
        ).to(device)

        # ==================Backward=================
        optimizer.zero_grad()

        if config['train_method'] == 'traditional_nms_augmented':
            augment = np.random.choice([True, False], p=[0.3, 0.7])

        result = model(
            rgb,
            thermal,
            depth,
            audio,
            label,
            augment=config.getboolean('audio_augmentation_merge'),
        )

        # For debug purposes
        if epoch == 0:
            with torch.no_grad():
                for i, item in enumerate(id):
                    logger.debug(f"\n{i}=> {item}")
                    logger.debug(f"rgb={torch.mean(rgb[i])}")
                    if config.getboolean('use_thermal'):
                        logger.debug(f"thermal={torch.mean(thermal[i])}")
                    if config.getboolean('use_depth'):
                        logger.debug(f"depth={torch.mean(depth[i])}")
                    if config.getboolean('use_audio'):
                        logger.debug(f"audio={torch.mean(audio[i])}")
                    logger.debug(f"label={label}")
                for j, modality in enumerate(model.module.teacher_models.keys()):
                    if modality == 'rgb':
                        logits_t, features_t = model.module.teacher_models['rgb'](rgb)
                    elif modality == 'thermal':
                        logits_t, features_t = model.module.teacher_models['thermal'](thermal)
                    elif modality == 'audio':
                        logits_t, features_t = model.module.teacher_models['audio'](audio)
                    elif modality == 'depth':
                        logits_t, features_t = model.module.teacher_models['depth'](depth)

                    annotations = logits_to_ground_truth(
                        logits=logits_t,
                        anchors=None,
                        valid_classes_dict=train_set.valid_classes_dict,
                        config=config,
                    )
                    logger.debug(f"GTs[{modality}]={annotations}")

        # Calculate the losses
        regression_losses, classification_losses, kd_losses, _, _, _ = result
        loss_regression = torch.mean(torch.stack(regression_losses))
        loss_cls = torch.mean(torch.stack(classification_losses))
        loss_main = loss_regression + loss_cls
        #loss_kd = torch.mean(torch.stack(kd_losses))
        loss_kd = torch.sum(torch.stack(kd_losses))
        loss_div = 0

        loss = config.getfloat('w_main') * loss_main
        loss += config.getfloat('w_div') * loss_div
        loss += config.getfloat('w_kd') * loss_kd
        loss.backward()

        if config.getfloat('grad_clip') > 0:
            torch.nn.utils.clip_grad_norm_(
                model.module.student_model.parameters(),
                config.getfloat('grad_clip')
            )

        optimizer.step()

        logger.info("="*40+"\n")
        logger.info(f"Epoch: {epoch + 1}/{config.getint('num_epoches')}")
        logger.info(f"Iteration: {iter+1}/{num_iter_per_epoch}")
        logger.info(f"Lr: {optimizer.param_groups[0]['lr']}")
        logger.info(f"Loss:{loss}")
        logger.info(f"Regression:{loss_regression}")
        logger.info(f"Cls:{loss_cls}")
        logger.info(f"KLDiv:{loss_div}")
        logger.info(f"KD:{loss_kd}")
        # Take the feedback from all modalities
        for i, modality in enumerate(model.module.teacher_models.keys()):
            logger.info(f"Regression_{modality}:{regression_losses}")
            logger.info(f"Cls_{modality}:{classification_losses}")
            logger.info(f"KLDiv_{modality}:{loss_div}")
            logger.info(f"KD_{modality}:{kd_losses}")
        logger.info("="*40+"\n")


        # Write to TensorBoard
        if writer:
            writer.add_scalar(
                f"Train/Total_loss",
                loss,
                epoch * num_iter_per_epoch + iter
            )
            writer.add_scalar(
                f"Train_/Regression_loss",
                loss_regression,
                epoch * num_iter_per_epoch + iter
            )
            writer.add_scalar(
                f"Train/Class_loss",
                loss_cls,
                epoch * num_iter_per_epoch + iter
            )
            writer.add_scalar(
                f"Train/KLDiv",
                loss_div,
                epoch * num_iter_per_epoch + iter
            )
            writer.add_scalar(
                f"Train/KD",
                loss_kd,
                epoch * num_iter_per_epoch + iter
            )

    return loss.item()
