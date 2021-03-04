# -*- coding: utf-8 -*-
"""Master Project -- Multi Modal Object Detection

This file contains general utilities for training models
"""
# -----------------------------------------------------------------
#                                Imports
# -----------------------------------------------------------------

# General Inputs
import logging
import os
import shutil

# Third Party
import pandas as pd
import numpy as np

# Local Imports
from src.optimization.traditional import train_traditional
from src.utils.utils import (
    custom_collate_factory,
    extract_criterions_from_config,
    logits_to_ground_truth,
    start_boardx_logger,
    weights_init,
    calc_gradient_penalty,
    isListEmpty,
)

from tabulate import tabulate

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.ops import nms

# -----------------------------------------------------------------
#                           Logger Configuration
# -----------------------------------------------------------------
# Logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
#                                 Methods
# -----------------------------------------------------------------
class ModelWithNMSKDListLossAugmented(nn.Module):
    def __init__(self, student_model, teacher_models, criterion_main, criterion_div, criterion_kd, config, valid_classes_dict):
        super().__init__()
        self.criterion_main = criterion_main
        self.criterion_div = criterion_div
        self.criterion_kd = criterion_kd
        self.student_model = student_model
        self.teacher_models = teacher_models
        self.config = config
        self.valid_classes_dict = valid_classes_dict

    def forward(self, rgb, thermal, depth, audio, label, validate=False, augment=False):

        logits_s, features_s = self.student_model(audio)

        # The annotations will be in batch labels
        batch_labels = [[] for i in range(rgb.shape[0])]

        # The features will be in kd_labels
        # Supposed to be a list of tensors
        kd_labels = []

        # for modality, teacher_model in self.teacher_models.items():
        modalities = list(self.teacher_models.keys())
        if augment:
            modalities.append('augmentation')
        for modality in modalities:
            if modality == 'augmentation':
                teacher_model = self.teacher_models['rgb']
            else:
                teacher_model = self.teacher_models[modality]
            with torch.no_grad():
                if modality == 'rgb':
                    prediction, features_t = teacher_model(rgb)
                elif modality == 'audio':
                    prediction, features_t = teacher_model(audio)
                elif modality == 'thermal':
                    prediction, features_t = teacher_model(thermal)
                elif modality == 'depth':
                    prediction, features_t = teacher_model(depth)
                elif modality == 'augmentation':
                    # Here label is a subset of rgb images that correspond to
                    # We have the original batch size of rgb, thermal depth
                    # the audio now is the original batch audio + each new audio
                    # taken from a thermal image
                    prediction, features_t = teacher_model(label)
                else:
                    raise ValueError('No valid modality to predict from teacher')
                # Detach for kd loss calculation
                if isinstance(features_t, tuple) or isinstance(features_t, list):
                    features_t = [f.detach() for f in features_t]
                else:
                    features_t = features_t.detach()

                kd_labels.append(features_t)

                this_batch_labels = logits_to_ground_truth(
                    logits=prediction,
                    anchors=None,
                    valid_classes_dict=self.valid_classes_dict,
                    config=self.config,
                    include_scores=True,
                )

            # Integrate all predictions
            for i in range(rgb.shape[0]):
                # No new prediction
                if isListEmpty(this_batch_labels[i]):
                    continue

                # A new prediction
                if len(np.shape(this_batch_labels[i])) == 1:
                    this_batch_labels[i] = np.expand_dims(this_batch_labels[i], axis=0)
                if isListEmpty(batch_labels[i]):
                    batch_labels[i] = this_batch_labels[i]
                    continue

                # If here, both them have it, so concat
                if len(np.shape(batch_labels[i])) == 1:
                    batch_labels[i] = np.expand_dims(batch_labels[i], axis=0)
                batch_labels[i] = np.concatenate(
                    (batch_labels[i], this_batch_labels[i]), axis=0)

        # Non-max suppress the prediction of multiple teachers
        for i in range(rgb.shape[0]):
            # No new prediction by any teacher
            if batch_labels[i] == []:
                continue
            # Else, do a maximum suppresion of the prediction
            idx = nms(
                boxes = torch.from_numpy(batch_labels[i][:,0:4]),
                scores= torch.from_numpy(batch_labels[i][:,4]),
                iou_threshold=0.5
            ).cpu().detach().numpy()

            # Remove scores
            batch_labels[i] = np.delete(batch_labels[i], 4, 1)

            # keep nms index
            batch_labels[i] = batch_labels[i][idx]

        # main loss function
        loss_regression, loss_cls = self.criterion_main(logits_s, batch_labels)

        loss_kd = torch.zeros(1)
        if self.criterion_kd is not None:
            # Due to parallel execution
            loss_kd = self.criterion_kd(
                features_s,
                kd_labels,
            )

        return [[loss_regression], [loss_cls], [loss_kd], torch.zeros(1).to(rgb.device), torch.zeros(1).to(rgb.device), torch.zeros(1).to(rgb.device)]


class ModelWithNMSKDListLoss(nn.Module):
    def __init__(self, student_model, teacher_models, criterion_main, criterion_div, criterion_kd, config, valid_classes_dict):
        super().__init__()
        self.criterion_main = criterion_main
        self.criterion_div = criterion_div
        self.criterion_kd = criterion_kd
        self.student_model = student_model
        self.teacher_models = teacher_models
        self.config = config
        self.valid_classes_dict = valid_classes_dict

    def forward(self, rgb, thermal, depth, audio, label, validate=False, augment=False):

        logits_s, features_s = self.student_model(audio)

        # The annotations will be in batch labels
        batch_labels = [[] for i in range(rgb.shape[0])]

        # The features will be in kd_labels
        # Supposed to be a list of tensors
        kd_labels = []

        for modality, teacher_model in self.teacher_models.items():
            with torch.no_grad():
                if modality == 'rgb':
                    prediction, features_t = teacher_model(rgb)
                elif modality == 'audio':
                    prediction, features_t = teacher_model(audio)
                elif modality == 'thermal':
                    prediction, features_t = teacher_model(thermal)
                elif modality == 'depth':
                    prediction, features_t = teacher_model(depth)
                else:
                    raise ValueError('No valid modality to predict from teacher')
                # Detach for kd loss calculation
                if isinstance(features_t, tuple) or isinstance(features_t, list):
                    features_t = [f.detach() for f in features_t]
                else:
                    features_t = features_t.detach()

                kd_labels.append(features_t)

                this_batch_labels = logits_to_ground_truth(
                    logits=prediction,
                    anchors=None,
                    valid_classes_dict=self.valid_classes_dict,
                    config=self.config,
                    include_scores=True,
                )

            # Integrate all predictions
            for i in range(rgb.shape[0]):
                # No new prediction
                if isListEmpty(this_batch_labels[i]):
                    continue

                # A new prediction
                if len(np.shape(this_batch_labels[i])) == 1:
                    this_batch_labels[i] = np.expand_dims(this_batch_labels[i], axis=0)
                if isListEmpty(batch_labels[i]):
                    batch_labels[i] = this_batch_labels[i]
                    continue

                # If here, both them have it, so concat
                if len(np.shape(batch_labels[i])) == 1:
                    batch_labels[i] = np.expand_dims(batch_labels[i], axis=0)
                batch_labels[i] = np.concatenate(
                    (batch_labels[i], this_batch_labels[i]), axis=0)

        # Non-max suppress the prediction of multiple teachers
        for i in range(rgb.shape[0]):
            # No new prediction by any teacher
            if batch_labels[i] == []:
                continue
            # Else, do a maximum suppresion of the prediction
            idx = nms(
                boxes = torch.from_numpy(batch_labels[i][:,0:4]),
                scores= torch.from_numpy(batch_labels[i][:,4]),
                iou_threshold=0.5
            ).cpu().detach().numpy()

            # Remove scores
            batch_labels[i] = np.delete(batch_labels[i], 4, 1)

            # keep nms index
            batch_labels[i] = batch_labels[i][idx]

        # main loss function
        loss_regression, loss_cls = self.criterion_main(logits_s, batch_labels)

        loss_kd = torch.zeros(1)
        if self.criterion_kd is not None:
            # Due to parallel execution
            loss_kd = self.criterion_kd(
                features_s,
                kd_labels,
            )
        return [[loss_regression], [loss_cls], [loss_kd], torch.zeros(1).to(rgb.device), torch.zeros(1).to(rgb.device), torch.zeros(1).to(rgb.device)]


class ModelWithNMSLossAugmented(nn.Module):
    def __init__(self, student_model, teacher_models, criterion_main, criterion_div, criterion_kd, config, valid_classes_dict):
        super().__init__()
        self.criterion_main = criterion_main
        self.criterion_div = criterion_div
        self.criterion_kd = criterion_kd
        self.student_model = student_model
        self.teacher_models = teacher_models
        self.config = config
        self.valid_classes_dict = valid_classes_dict

    def average_batch_0_1(self, features_t):

        # features_t is a list
        with torch.no_grad():
            for i in range(len(features_t)):
                features_t[i][1] = (features_t[i][0] + features_t[i][1])/2
                # Remove the first batch as it has been averaged with the second
                #features_t[i] = features_t[i][1:]
                # DOnt remove because of RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
                # we can stil use the batch 0, will be an extra image I guess

        return features_t

    def merge_batch_0_1(self, audio):

        # The spectrogram is the square of the complex magnitude of the STFT
        # spectrogram_librosa = np.abs(librosa.stft(
        #    y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hann'
        #)) ** 2
        # then we calculate the log10 of that
        # There is no way around the non linear square magnitue, only if the two
        # inputs are uncorrelated, this make sense. We assume that the likelihood of
        # taking a image from different drives is high so this term will be uncorrelated
        # https://stackoverflow.com/questions/36817236/spectrogram-of-two-audio-files-added-together
        # we approximate this via og((|stft(a) + stft(b)|)^2) = log(|stft(a)|^2) + log(|stft(b)|^2)
        with torch.no_grad():
            audio[1] = torch.pow(audio[0], 10) + torch.pow(audio[1], 10)
            eps = 1e-7
            audio[1][audio[1]<eps] = eps
            audio[1] = torch.log10(audio[1])

        #audio = audio[1:]
        return audio

    def forward(self, rgb, thermal, depth, audio, label, validate=False, augment=False):

        # TODO, check for rgb.shape greater than 2... but unlickely case

        kd_losses = []
        if augment:
            audio = self.merge_batch_0_1(audio)

        logits_s, features_s = self.student_model(audio)
        batch_labels = [[] for i in range(rgb.shape[0])]
        for modality, teacher_model in self.teacher_models.items():
            with torch.no_grad():
                if modality == 'rgb':
                    prediction, features_t = teacher_model(rgb)
                elif modality == 'audio':
                    prediction, features_t = teacher_model(audio)
                elif modality == 'thermal':
                    prediction, features_t = teacher_model(thermal)
                elif modality == 'depth':
                    prediction, features_t = teacher_model(depth)
                else:
                    raise ValueError('No valid modality to predict from teacher')
                # Detach for kd loss calculation
                if isinstance(features_t, tuple) or isinstance(features_t, list):
                    features_t = [f.detach() for f in features_t]
                else:
                    features_t = features_t.detach()

                # we have to average the feature 0 and feature 1
                # to comply with augmentation
                if augment:
                    features_t = self.average_batch_0_1(features_t)

                this_batch_labels = logits_to_ground_truth(
                    logits=prediction,
                    anchors=None,
                    valid_classes_dict=self.valid_classes_dict,
                    config=self.config,
                    include_scores=True,
                )

            loss_kd = torch.zeros(1)
            if self.criterion_kd is not None:
                # Due to parallel execution
                loss_kd = self.criterion_kd(
                    features_s,
                    features_t
                )
            kd_losses.append(loss_kd)

            # Integrate all predictions
            for i in range(rgb.shape[0]):
                # No new prediction
                if isListEmpty(this_batch_labels[i]):
                    continue

                # A new prediction
                if len(np.shape(this_batch_labels[i])) == 1:
                    this_batch_labels[i] = np.expand_dims(this_batch_labels[i], axis=0)
                if isListEmpty(batch_labels[i]):
                    batch_labels[i] = this_batch_labels[i]
                    continue

                # If here, both them have it, so concat
                if len(np.shape(batch_labels[i])) == 1:
                    batch_labels[i] = np.expand_dims(batch_labels[i], axis=0)
                batch_labels[i] = np.concatenate(
                    (batch_labels[i], this_batch_labels[i]), axis=0)

        # Here, merge labels [0] and [1]
        # Do so before NMS
        # If any one is [] then it is like adding noise
        # Notice we do not remove batch 0 and also use it as another
        # image for gradient
        if augment and batch_labels[1] != [] and batch_labels[0] != []:
            batch_labels[1] = np.concatenate(
                (batch_labels[0], batch_labels[1]), axis=0)
            #del batch_labels[0]


        # Non-max suppress the prediction of multiple teachers
        for i in range(rgb.shape[0]):
            # if augment, do go to the last non existant i
            # that is, we merged 0 and 1 so, there is one less to process
            #if augment and i == rgb.shape[0] - 1:
            #    break

            # No new prediction by any teacher
            if batch_labels[i] == []:
                continue
            # Else, do a maximum suppresion of the prediction
            idx = nms(
                boxes = torch.from_numpy(batch_labels[i][:,0:4]),
                scores= torch.from_numpy(batch_labels[i][:,4]),
                iou_threshold=0.5
            ).cpu().detach().numpy()

            # Remove scores
            batch_labels[i] = np.delete(batch_labels[i], 4, 1)

            # keep nms index
            batch_labels[i] = batch_labels[i][idx]


        #if not all([np.any(elem) for elem in batch_labels]):
        #if isListEmpty(batch_labels):
        loss_regression, loss_cls = self.criterion_main(logits_s, batch_labels)

        #print(f"regression_losses={regression_losses} {rgb.device}")
        #print(f"classification_losses={classification_losses}")
        #print(f"loss_div={loss_div}")
        #print(f"loss_kd={loss_kd}")
        return [[loss_regression], [loss_cls], kd_losses, torch.zeros(1).to(rgb.device), torch.zeros(1).to(rgb.device), torch.zeros(1).to(rgb.device)]


class ModelWithNMSLoss(nn.Module):
    def __init__(self, student_model, teacher_models, criterion_main, criterion_div, criterion_kd, config, valid_classes_dict):
        super().__init__()
        self.criterion_main = criterion_main
        self.criterion_div = criterion_div
        self.criterion_kd = criterion_kd
        self.student_model = student_model
        self.teacher_models = teacher_models
        self.config = config
        self.valid_classes_dict = valid_classes_dict

    def forward(self, rgb, thermal, depth, audio, label, validate=False, augment=False):

        kd_losses = []
        logits_s, features_s = self.student_model(audio)
        batch_labels = [[] for i in range(rgb.shape[0])]
        for modality, teacher_model in self.teacher_models.items():
            with torch.no_grad():
                if modality == 'rgb':
                    prediction, features_t = teacher_model(rgb)
                elif modality == 'audio':
                    prediction, features_t = teacher_model(audio)
                elif modality == 'thermal':
                    prediction, features_t = teacher_model(thermal)
                elif modality == 'depth':
                    prediction, features_t = teacher_model(depth)
                else:
                    raise ValueError('No valid modality to predict from teacher')
                # Detach for kd loss calculation
                if isinstance(features_t, tuple) or isinstance(features_t, list):
                    features_t = [f.detach() for f in features_t]
                else:
                    features_t = features_t.detach()

                this_batch_labels = logits_to_ground_truth(
                    logits=prediction,
                    anchors=None,
                    valid_classes_dict=self.valid_classes_dict,
                    config=self.config,
                    include_scores=True,
                )

            loss_kd = torch.zeros(1)
            if self.criterion_kd is not None:
                # Due to parallel execution
                loss_kd = self.criterion_kd(
                    features_s,
                    features_t
                )
            kd_losses.append(loss_kd)

            # Integrate all predictions
            for i in range(rgb.shape[0]):
                # No new prediction
                if isListEmpty(this_batch_labels[i]):
                    continue

                # A new prediction
                if len(np.shape(this_batch_labels[i])) == 1:
                    this_batch_labels[i] = np.expand_dims(this_batch_labels[i], axis=0)
                if isListEmpty(batch_labels[i]):
                    batch_labels[i] = this_batch_labels[i]
                    continue

                # If here, both them have it, so concat
                if len(np.shape(batch_labels[i])) == 1:
                    batch_labels[i] = np.expand_dims(batch_labels[i], axis=0)
                batch_labels[i] = np.concatenate(
                    (batch_labels[i], this_batch_labels[i]), axis=0)

        # Non-max suppress the prediction of multiple teachers
        for i in range(rgb.shape[0]):
            # No new prediction by any teacher
            if batch_labels[i] == []:
                continue
            # Else, do a maximum suppresion of the prediction
            idx = nms(
                boxes = torch.from_numpy(batch_labels[i][:,0:4]),
                scores= torch.from_numpy(batch_labels[i][:,4]),
                iou_threshold=0.5
            ).cpu().detach().numpy()

            # Remove scores
            batch_labels[i] = np.delete(batch_labels[i], 4, 1)

            # keep nms index
            batch_labels[i] = batch_labels[i][idx]

        #if not all([np.any(elem) for elem in batch_labels]):
        #if isListEmpty(batch_labels):
        loss_regression, loss_cls = self.criterion_main(logits_s, batch_labels)

        return [[loss_regression], [loss_cls], kd_losses, torch.zeros(1).to(rgb.device), torch.zeros(1).to(rgb.device), torch.zeros(1).to(rgb.device)]


class ModelWithLoss(nn.Module):
    def __init__(self, student_model, teacher_models, criterion_main, criterion_div, criterion_kd, config, valid_classes_dict):
        super().__init__()
        self.criterion_main = criterion_main
        self.criterion_div = criterion_div
        self.criterion_kd = criterion_kd
        self.student_model = student_model
        self.teacher_models = teacher_models
        self.config = config
        self.valid_classes_dict = valid_classes_dict

    def forward(self, rgb, thermal, depth, audio, label, validate=False):

        logits_s, features_s = self.student_model(audio)

        regression_losses = []
        classification_losses = []
        kd_losses = []
        for modality in self.teacher_models.keys():

            # generate the annotations
            with torch.no_grad():
                if modality == 'rgb':
                    logits_t, features_t = self.teacher_models['rgb'](rgb)
                elif modality == 'thermal':
                    logits_t, features_t = self.teacher_models['thermal'](thermal)
                elif modality == 'audio':
                    logits_t, features_t = self.teacher_models['audio'](audio)
                elif modality == 'depth':
                    logits_t, features_t = self.teacher_models['depth'](depth)

                # Detach for kd loss calculation
                if isinstance(features_t, tuple) or isinstance(features_t, list):
                    features_t = [f.detach() for f in features_t]
                else:
                    features_t = features_t.detach()

                if self.config.getboolean('use_labels'):
                    annotations = label
                else:
                    annotations = logits_to_ground_truth(
                        logits=logits_t,
                        anchors=None,
                        valid_classes_dict=self.valid_classes_dict,
                        config=self.config,
                    )

            loss_regression, loss_cls = self.criterion_main(logits_s, annotations)
            regression_losses.append(loss_regression)
            classification_losses.append(loss_cls)

            loss_div = torch.zeros_like(loss_regression)
            if self.criterion_div is not None:
                loss_div = self.criterion_div(logits_s, logits_t)

            loss_kd = torch.zeros_like(loss_regression)
            if self.criterion_kd is not None:
                # Due to parallel execution
                loss_kd = self.criterion_kd(
                    features_s,
                    features_t
                )
            kd_losses.append(loss_kd)

        return [regression_losses, classification_losses, kd_losses, torch.zeros_like(loss_regression), torch.zeros_like(loss_regression), torch.zeros_like(loss_regression)]


class ModelWithLossMultiHeadAdversarial(nn.Module):
    def __init__(self, student_model, teacher_models, discriminators, criterion_main, config, valid_classes_dict):
        super().__init__()
        # Loss function
        if config['adv_loss'] == 'BCELoss':
            self.criterion_adversarial = torch.nn.BCELoss()
        elif config['adv_loss'] == 'MultiBCELoss':
            self.criterion_adversarial = MultiBCELoss()
        else:
            raise Exception("Invalid adversarial loss provided")

        self.criterion_main = criterion_main
        self.student_model = student_model
        self.teacher_models = teacher_models
        self.discriminators = discriminators
        self.valid_classes_dict = valid_classes_dict
        self.config = config

    def forward(self, rgb, thermal, depth, audio, label, validate=False):

        # Adversarial ground truths
        Tensor = torch.cuda.FloatTensor
        valid = Variable(
            Tensor(rgb.shape[0], 1).fill_(1.0),
            requires_grad=False

        )
        fake = Variable(
            Tensor(rgb.shape[0], 1).fill_(0.0),
            requires_grad=False
        )

        logits_s, features_s = self.student_model(audio)

        regression_losses = []
        classification_losses = []
        kd_losses = []
        s_features = []
        losses_d = []
        for modality in self.teacher_models.keys():

            s_features.append(features_s[modality])
            # generate the annotations
            with torch.no_grad():
                if modality == 'rgb':
                    logits_t, features_t = self.teacher_models['rgb'](rgb)
                elif modality == 'thermal':
                    logits_t, features_t = self.teacher_models['thermal'](thermal)
                elif modality == 'audio':
                    logits_t, features_t = self.teacher_models['audio'](audio)
                elif modality == 'depth':
                    logits_t, features_t = self.teacher_models['depth'](depth)

                # Detach for kd loss calculation
                if isinstance(features_t, tuple) or isinstance(features_t, list):
                    features_t = [f.detach() for f in features_t]
                else:
                    features_t = features_t.detach()

                annotations = logits_to_ground_truth(
                    logits=logits_t,
                    anchors=None,
                    valid_classes_dict=self.valid_classes_dict,
                    config=self.config,
                )

            loss_regression, loss_cls = self.criterion_main(logits_s[modality], annotations)
            regression_losses.append(loss_regression)
            classification_losses.append(loss_cls)

            #loss_div = torch.zeros_like(loss_regression)
            #if self.criterion_div is not None:
            #    loss_div = self.criterion_div(logits_s, logits_t)

            #loss_kd = torch.zeros_like(loss_regression)
            #if self.criterion_kd is not None:
            #    # Due to parallel execution
            #    loss_kd = self.criterion_kd(
            #        features_s,
            #        features_t
            #    )
            #kd_losses.append(loss_kd)

            real_loss = self.criterion_adversarial(
                self.discriminators[modality](features_t),
                valid
            )

            if isinstance(features_s[modality], tuple) or isinstance(features_s[modality], list):
                features_s_detached = [f.detach() for f in features_s[modality]]
            else:
                features_s_detached = features_s[modality].detach()
            fake_loss = self.criterion_adversarial(
                self.discriminators[modality](features_s_detached),
                fake
            )
            # d_loss = (real_loss + fake_loss) / 2
            # Someone adviced to do If the generator minimizes criterion,
            # then you probably want it to be real_loss - fake_loss,
            # so that the discriminators push up the criterion for fake images.
            d_loss = real_loss + fake_loss
            losses_d.append(d_loss)

        return regression_losses, classification_losses, losses_d, s_features, [], []


class ModelWithLossMultiHeadAdversarialWGANGP(nn.Module):
    def __init__(self, student_model, teacher_models, discriminators, criterion_main, config, valid_classes_dict):
        super().__init__()
        self.criterion_main = criterion_main
        self.student_model = student_model
        self.teacher_models = teacher_models
        self.discriminators = discriminators
        self.valid_classes_dict = valid_classes_dict
        self.config = config

    def forward(self, rgb, thermal, depth, audio, label, validate=False):

        logits_s, features_s = self.student_model(audio)

        regression_losses = []
        classification_losses = []
        kd_losses = []
        s_features = []
        reals = []
        fakes = []
        gps = []
        for modality in self.teacher_models.keys():

            s_features.append(features_s[modality])
            # generate the annotations
            with torch.no_grad():
                if modality == 'rgb':
                    logits_t, features_t = self.teacher_models['rgb'](rgb)
                elif modality == 'thermal':
                    logits_t, features_t = self.teacher_models['thermal'](thermal)
                elif modality == 'audio':
                    logits_t, features_t = self.teacher_models['audio'](audio)
                elif modality == 'depth':
                    logits_t, features_t = self.teacher_models['depth'](depth)

                # Detach for kd loss calculation
                if isinstance(features_t, tuple) or isinstance(features_t, list):
                    features_t = [f.detach() for f in features_t]
                else:
                    features_t = features_t.detach()

                annotations = logits_to_ground_truth(
                    logits=logits_t,
                    anchors=None,
                    valid_classes_dict=self.valid_classes_dict,
                    config=self.config,
                )

            loss_regression, loss_cls = self.criterion_main(logits_s[modality], annotations)
            regression_losses.append(loss_regression)
            classification_losses.append(loss_cls)

            D_real = self.discriminators[modality](features_t)
            reals.append(D_real)
            if isinstance(features_s[modality], tuple) or isinstance(features_s[modality], list):
                features_s_detached = [f.detach() for f in features_s[modality]]
            else:
                features_s_detached = features_s[modality].detach()
            D_fake = self.discriminators[modality](features_s_detached)
            fakes.append(D_fake)
            gradient_penalty = []
            if not validate:
                gradient_penalty = calc_gradient_penalty(
                    self.discriminators[modality],
                    features_t,
                    features_s_detached
                )
            gps.append(gradient_penalty)

        return regression_losses, classification_losses, reals, fakes, gps, s_features


def train(
    teacher_models,
    student_model,
    config,
    train_set,
    val_set,
    method="traditional",
    no_validation=False,
    writer=True,
):
    """
    Trains a model in different traditional/adversarial training
    Args:
            teacher_model: A pretrained model.
            student_model: The model to be trained
            config: A parsed configuration file
            train_set: A dataset object from where to draw inputs
            val_set: A dataset object used for validation
            method: traditional or adversarial training
            no_validation: Skip any validation to speed up time

    Returns:
            Student model with weights improved
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.warn(f"Begin training with following configuration:")
    logger.warn("\n" + tabulate(pd.DataFrame(
        config.items()),
        headers='keys',
        tablefmt='psql'
        )
    )

    # Configure training helpers

    # Loss
    criterion_main, criterion_div, criterion_kd = extract_criterions_from_config(
        config,
        train_set
    )

    # What to optimize
    trainable_list = nn.ModuleList([])
    trainable_list.append(student_model)
    if config['kd_loss'] == 'CRDLoss':
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif config['kd_loss'] == 'MultiTeacherContrastiveAttentionLoss':
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)

    # How to optimize
    if config['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            trainable_list.parameters(),
            lr=config.getfloat('lr'),
            momentum=config.getfloat('momentum'),
            weight_decay=config.getfloat('weight_decay')
        )
    elif config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(
            trainable_list.parameters(),
            lr=config.getfloat("lr"),
            betas=(
                config.getfloat("b1"),
                config.getfloat("b2")
            )
        )
    elif config['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            trainable_list.parameters(),
            lr=config.getfloat("lr"),
            betas=(
                config.getfloat("b1"),
                config.getfloat("b2")
            )
        )
    else:
        raise Exception(f"Unsupported optimizer {config['optimizer']}")

    discriminators = None
    optimizer_D = None
    if 'adversarial' in method:
        discriminators = torch.nn.ModuleDict()
        for modality in teacher_models.keys():
            discriminators[modality] = Discriminator().to(device)
            discriminators[modality].apply(weights_init)
        optimizer_D = torch.optim.Adam(
            discriminators.parameters(),
            lr=config.getfloat("lr"),
            betas=(config.getfloat("b1"), config.getfloat("b2"))
        )

    # Scheduler
    if config['scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.getint('step_size'),
            gamma=config.getfloat('gamma'),
        )
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=3,
            verbose=True
        )
    elif config['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
        )
    else:
        raise Exception(f"Unsupported scheduler {config['scheduler']}")

    # Check if we should resume
    start_epoch, best_loss, best_epoch = resume_from_checkpoint(
        config=config,
        student_model=student_model,
        discriminators=discriminators,
        optimizer=optimizer,
        optimizer_D=optimizer_D,
        scheduler=scheduler,
    )

    # Fix weights of the teacher and unfix the weights of the student
    teacher_models.eval()
    for param in teacher_models.parameters():
        param.requires_grad = False

    # Make sure student is up for training
    student_model.train()

    # Better parallelization support
    if 'traditional' in method:
        func = ModelWithLoss
        if method == 'traditional_nms':
            func = ModelWithNMSLoss
        elif method == 'traditional_nms_kdlist':
            func = ModelWithNMSKDListLoss
        elif method == 'traditional_nms_augmented':
            func = ModelWithNMSLossAugmented
        elif method == 'traditional_nms_kdlist_augmented':
            func = ModelWithNMSKDListLossAugmented
        else:
            # Notice, traditional is deprecated in favor of traditional_nms
            raise ValueError(f"unsupported train method={method}")
        model = func(
            student_model,
            teacher_models,
            criterion_main,
            criterion_div,
            criterion_kd,
            config,
            train_set.valid_classes_dict
        )
    elif 'train_adversarial_multiteacher_wgangp' in method:
        assert config['features_from'] == 'header'
        model = ModelWithLossMultiHeadAdversarialWGANGP(
            student_model,
            teacher_models,
            discriminators,
            criterion_main,
            config,
            train_set.valid_classes_dict
        )
    elif 'adversarial' in method:
        assert config['features_from'] == 'header'
        model = ModelWithLossMultiHeadAdversarial(
            student_model,
            teacher_models,
            discriminators,
            criterion_main,
            config,
            train_set.valid_classes_dict
        )
    else:
        raise ValueError(f"unsupported train method={method}")
    # Support parallel execution
    if (device.type == 'cuda') and (config.getint('ngpu') > 1):
        logger.info(f"Parallel exec for {config.getint('ngpu')} GPUs")
        if config['engine'] == 'DataParallel':
            # On Resume, prevent a teacher model of teacher model
            if not type(model) == torch.nn.DataParallel:
                model = nn.DataParallel(
                    model,
                    list(range(config.getint('ngpu')))
                )
        elif config['engine'] == 'DistributedDataParallel':
            model = model.cuda()
            # On Resume, prevent a teacher model of teacher model
            if not type(model) == torch.nn.parallel.DistributedDataParallel:
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[int(config['local_rank'])],
                    output_device=int(config['local_rank'])
                )

    # Log train Information
    if writer:
        writer = start_boardx_logger(config)

    for epoch in range(start_epoch, config.getint('num_epoches')):

        # Training
        if 'traditional' in method:
            # Traditional training with direct likelihood opt
            loss = train_traditional(
                train_set,
                model,
                optimizer,
                epoch,
                config,
                writer
            )
        elif 'adversarial' in method:
            if method == 'adversarial':
                func = train_adversarial
            elif method == 'train_adversarial_multiteacher':
                func = train_adversarial_multiteacher
            elif method == 'train_adversarial_multiteacher_wgangp':
                func = train_adversarial_multiteacher_wgangp
            else:
                raise NotImplementedError()
            loss = func(
                train_set,
                model,
                discriminators,
                optimizer,
                optimizer_D,
                epoch,
                config,
                writer,
            )
        else:
            raise Exception(f"Unsupported train method {method} provided")

        # Take a scheduler step
        if config['scheduler'] == 'StepLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(loss)

        # Skip anything further as no validation was requested
        if no_validation:
            continue

        # Validation
        is_best = False
        if epoch % config.getint('val_interval') == 0:
            val_loss = validate(
                val_set,
                model,
                epoch,
                config,
                writer,
            )

            # Save best run so far
            is_best = val_loss < best_loss
            if is_best:
                logger.debug(f"Epoch={epoch+1} Best{best_loss}=>{val_loss}")
                if type(model) in [torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel]:
                    torch.save(
                        model.module.student_model.state_dict(),
                        f"{config['exp_name']}/only_parameters_student_best.{config['rank']}")
                else:
                    torch.save(
                        model.student_model.state_dict(),
                        f"{config['exp_name']}/only_parameters_student_best.{config['rank']}")
                best_loss = val_loss
                best_epoch = epoch + 1
            # Early stopping
            if epoch - best_epoch > config.getint('es_patience') > 0:
                if not method == 'adversarial':
                    logger.info(f"ES Epoch{epoch}. Lowest loss is {val_loss}")
                    break
                else:
                    logger.debug(f"adversarial not ES on loss {val_loss}")

        # Only store runs that are the best in fast mode
        if config.getboolean('fast_run') and not is_best:
            continue

        state = {
            'epoch': epoch + 1,
            'state_dict': model.module.student_model.state_dict(),
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        if 'adversarial' in method:
            state['optimizer_D'] = optimizer_D.state_dict()
            state['discriminators'] = discriminators.state_dict()
        save_checkpoint(
            state,
            is_best=is_best,
            config=config,
        )

    if writer:
        writer.export_scalars_to_json(f"{config['exp_name']}/all_logs.{config['rank']}.json")
        writer.close()

    # Only validation at the end
    if no_validation:
        val_loss = validate(
            val_set,
            model,
            epoch=config.getint('num_epoches'),
            config=config,
            writer=writer,
        )

    return val_loss


def validate(
    val_set,
    model,
    epoch,
    config,
    writer
):
    """
    Validate a student model
    Args:
            val_set: Add dataset object used for validation
            teacher_models: A pretrained model.
            student_model: The model to be trained
            config: A parsed configuration file
            epoch: The current epoch for loggin
            writer: A writer object to store tensorboard info
    Returns:
            The validation loss of the model
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the training validator
    val_generator = DataLoader(
        val_set,
        batch_size=min(config.getint('batch_size'), len(val_set)),
        shuffle=False,
        drop_last=True,
        collate_fn=custom_collate_factory(config),
        num_workers=config.getint('num_workers'),
    )

    model.module.student_model.eval()
    loss_ls = []
    loss_regression_ls = []
    loss_cls_ls = []
    loss_div_ls = []
    loss_kd_ls = []
    for val_iter, val_batch in enumerate(val_generator):
        rgb, thermal, depth, audio, label, id = val_batch
        modalities = {}
        num_sample = len(rgb)
        rgb = rgb.to(device)
        audio = audio.to(device)
        if config.getboolean('use_thermal'):
            thermal = thermal.to(device)
        if config.getboolean('use_depth'):
            depth = depth.to(device)

        with torch.no_grad():

            # =================Forward=================
            regression_losses, classification_losses, kd_losses, _, _, _ = model(
                rgb,
                thermal,
                depth,
                audio,
                label,
                validate=True,
            )
            #b_loss_regression = torch.mean(torch.stack(regression_losses))
            b_loss_regression = torch.sum(torch.stack(regression_losses))
            #b_loss_cls = torch.mean(torch.stack(classification_losses))
            b_loss_cls = torch.sum(torch.stack(classification_losses))
            b_loss_div = 0
            #b_loss_kd = torch.mean(torch.stack(kd_losses))
            b_loss_kd = torch.sum(torch.stack(kd_losses))
            b_loss_main = b_loss_regression + b_loss_cls

            b_loss = config.getfloat('w_main') * b_loss_main
            b_loss += config.getfloat('w_div') * b_loss_div
            b_loss += config.getfloat('w_kd') * b_loss_kd

        loss_ls.append(b_loss * num_sample)
        loss_regression_ls.append(b_loss_regression * num_sample)
        loss_cls_ls.append(b_loss_cls * num_sample)
        loss_div_ls.append(b_loss_div * num_sample)
        loss_kd_ls.append(b_loss_kd * config.getfloat('w_kd') * num_sample)
    val_loss = sum(loss_ls) / val_set.__len__()
    val_regression_loss = sum(loss_regression_ls) / val_set.__len__()
    val_cls_loss = sum(loss_cls_ls) / val_set.__len__()
    val_div_loss = sum(loss_div_ls) / val_set.__len__()
    val_kd_loss = sum(loss_kd_ls) / val_set.__len__()

    # ===================meters=====================

    logger.warn("="*15+"VAL"+"="*15+"\n")
    logger.warn(f"Epoch: {epoch + 1}/{config.getint('num_epoches')}")
    logger.warn(f"Loss:{val_loss}")
    logger.warn(f"Regression:{val_regression_loss}")
    logger.warn(f"Cls:{val_cls_loss}")
    logger.warn(f"KLDiv:{val_div_loss}")
    logger.warn(f"KD:{val_kd_loss}")
    logger.warn("="*34+"\n")

    if writer:
        writer.add_scalar('Test/Total_loss', val_loss, epoch)
        writer.add_scalar('Test/Regression_loss', val_regression_loss, epoch)
        writer.add_scalar('Test/Class_loss', val_cls_loss, epoch)
        writer.add_scalar('Test/KLDiv', val_div_loss, epoch)
        writer.add_scalar('Test/KD', val_kd_loss, epoch)

    return val_loss


def resume_from_checkpoint(
        config,
        student_model,
        discriminators,
        optimizer,
        optimizer_D,
        scheduler
):
    """
    Resume from a checkpoint
    Args:
            config: A parsed configuration file
            student_model: The model to be trained
            discriminator: The discriminator for adversarial training
            optimizer: The optimizer of the student model
            optimizer_D: The optimizer of the discriminator model
            scheduler: The scheduler state

    Returns:
            start_epoch: From where to start training
            best_loss: Magnitud of the best loss
            best_epoch: When did the best loss appeared
    """
    start_epoch = 0
    best_loss = 1e10
    best_epoch = 0

    if config.getboolean('resume'):
        if os.path.exists(f"{config['exp_name']}/checkpoint.{config['rank']}.pth.tar"):
            map_location = {'cuda:%d' % 0: f"cuda:{config['rank']}"}
            checkpoint = torch.load(
                f"{config['exp_name']}/checkpoint.{config['rank']}.pth.tar",
                map_location=map_location
            )
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            best_epoch = checkpoint['best_epoch']
            start_epoch = checkpoint['epoch']
            student_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'optimizer_D' in checkpoint:
                optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            if 'discriminators' in checkpoint:
                discriminators.load_state_dict(checkpoint['discriminators'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logging.info(f"Starting from epoch={start_epoch}")
            logging.info(f"Load {config['exp_name']}/checkpoint.{config['rank']}.pth.tar")

    return start_epoch, best_loss, best_epoch


def save_checkpoint(state, is_best, config):
    """
    Save the checkpoint for resuming
    Args:
            state: State to be saved
            is_best: Whether or not to save the state as best
            config: A parsed configuration file

    Returns:
            None
    """
    filename = f"{config['exp_name']}/checkpoint.{config['rank']}.pth.tar"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f"{config['exp_name']}/best.{config['rank']}.pth.tar")
    return


def weights_init_normal(m):
    """
    Applies a weight initialization to conv/batchnorm
    Args:
            m: The class name give by apply

    Returns:
            None
    """
    # generator.apply(weights_init_normal)
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    return
