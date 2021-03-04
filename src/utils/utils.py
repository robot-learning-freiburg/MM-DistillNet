# -*- coding: utf-8 -*-
"""Master Project -- Multi Modal Object Detection

General utilities for object detection

"""

# General Inputs
import operator
import logging
import os
import copy
import pickle
import time
import math
import random
import shutil
import re
from datetime import datetime

# Third Party
import cv2

import librosa
import librosa.display

from google_drive_downloader import GoogleDriveDownloader as gdd

# Plotting
import matplotlib.pyplot as plt

import numpy as np

from matplotlib import cm
from PIL import Image

import pandas as pd

from src.fullcnn_net import FullCNNNet
from src.datasets.transformations import (
    HSVAdjust,
    Normalizer,
    Resize,
    Resizer,
    ThermalAugmenter,
    AudioAugmenter,
)
from src.StereoSoundNet import StereoSoundNet

from src.loss.AttentionLoss import AttentionLoss
from src.loss.ABLoss import ABLoss
from src.loss.MTALoss import MTALoss
from src.loss.KLLoss import KLLoss
from src.loss.GroupAttentionLoss import GroupAttentionLoss
from src.loss.MultiTeacherPairWiseSimilarityLoss import MultiTeacherPairWiseSimilarityLoss
from src.loss.MultiTeacherContrastiveAttentionLoss import MultiTeacherContrastiveAttentionLoss
from src.loss.MultiTeacherTrippletAttentionLoss import MultiTeacherTrippletAttentionLoss
from src.loss.CRDLoss import CRDLoss
from src.loss.DistillKL import DistillKL
from src.loss.FocalLoss import FocalLoss
from src.loss.YetAnotherFocalLoss import YetAnotherFocalLoss
from src.loss.NSTLoss import NSTLoss
from src.loss.PKTLoss import PKTLoss
from src.loss.SimilarityLoss import SimilarityLoss
from src.loss.RankingLoss import RankingLoss


from tabulate import tabulate

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_

from tqdm import tqdm

import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis

# No support for efficientdet
from torchvision.ops import nms
from src.YetAnotherEfficientDet import YetAnotherEfficientDet, YetAnotherEfficientDetBBoxTransform
from src.YetAnotherEfficientDet_generator import YetAnotherEfficientDetGenerator
from src.YetAnotherEfficientDetMultiHeaded import YetAnotherEfficientDetMultiHeaded
from torchvision.ops.boxes import batched_nms


# -------------------------------------------------------------
#                           Logger Configuration
# -------------------------------------------------------------
# Logging
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
#                                 Methods
# -------------------------------------------------------------
def custom_collate_factory(config):
    """
    Factory of collate functions to handle tensors and list
    Args:
            config: A parsed configuration file.

    Returns:
            A function to collate multiple modalities.

    """
    def custom_collate_fn(batch):
        items = list(zip(*batch))
        # rgb is always available
        items[0] = default_collate(items[0])
        if config.getboolean('use_thermal'):
            items[1] = default_collate(items[1])
        else:
            items[1] = list(items[1])
        if config.getboolean('use_depth'):
            items[2] = default_collate(items[2])
        else:
            items[2] = list(items[2])
        # if config.getboolean('use_audio'):
        if True:
            items[3] = default_collate(items[3])
        else:
            items[3] = list(items[3])
        items[4] = list(items[4])
        items[5] = list(items[5])
        return items
    return custom_collate_fn


class ClipBoxes(nn.Module):
    """
    Taken from https://github.com/toandaominh1997/EfficientDet.Pytorch.git
    Clips predictions that go out of the image boundary
    """

    def __init__(self, rgb_size=None):
        super(ClipBoxes, self).__init__()
        self.rgb_size = rgb_size

    def forward(self, boxes):

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=self.rgb_size)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=self.rgb_size)

        return boxes


def EfficientDet_post_processing(
    logits,
    valid_classes_dict,
    anchors,
    config,
    text_classes=True,
    regressBoxes = None,
    clipBoxes = None,
):
    """

    #https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/f2bf780ccd84771435f6e15af3fc1d85c0f1309d/utils/utils.py#L91
    """

    conf_threshold=config.getfloat('conf_threshold')
    nms_threshold=config.getfloat('nms_threshold')
    image_size = config.getint('image_size')

    classification, regression, anchors = logits

    # If running with parallel, anchors would be concat.
    # Code expects 1, 49104,4 anchor but in parallel process
    # a batch is split in 2 process, and when joined a 2, 49104, 4
    # Object is created. Take one one anchor
    anchors = anchors[[0]]

    if regressBoxes is None:
        raise Exception("User must provide the type of efficientdet regressor")

    if clipBoxes is None:
        clipBoxes = ClipBoxes(rgb_size=image_size)

    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > conf_threshold)[:, :, 0]
    final_boxes = list()
    ignore_labels = []
    if 'ignore_labels' in config:
        ignore_labels = config['ignore_labels'].split(',')
        ignore_labels = [int(x) for x in ignore_labels]
    for i in range(scores.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            final_boxes.append([])
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)

        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)

        # Just process the classes we care
        mask = (
            classes_[..., None] == torch.tensor(
                list(valid_classes_dict['predictions_txt2i'].values())).to(classes_.device).type(classes_.type())
                ).any(-1).type(torch.BoolTensor)

        transformed_anchors_per = transformed_anchors_per[mask]
        classes_ = classes_[mask]
        scores_per = scores_per[mask]
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=nms_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            for label in ignore_labels:
                boxes_ = boxes_[classes_ != label]
                scores_ = scores_[classes_ != label]
                classes_ = classes_[classes_ != label]

            predictions = np.hstack((
                boxes_.cpu().numpy(),
                scores_.cpu().numpy().reshape(-1,1),
                classes_.cpu().numpy().reshape(-1,1)
            )).tolist()

            if text_classes:
                for i in range(len(predictions)):
                    predictions[i][5] = valid_classes_dict['predictions_i2txt'][int(predictions[i][5])]

            final_boxes.append(predictions)
        else:
            final_boxes.append([])

    return final_boxes


def logits_to_ground_truth(
        logits,
        anchors,
        valid_classes_dict,
        config,
        include_scores=False,
        crash_if_no_pred=False,
        text_classes=False,
        regressBoxes=None,
        clipBoxes = None,
):
    """
    Take the logits from a model and build the actual object detections
    Args:
            logits: the predictions of a Yolo Model
            anchors: Yolo anchors fro prediction processing
            classes: The ordered classes list
            id: id of image, used for debug
            config: A config object to know how to run
            include_scores: If we should include score in the returned object
            crash_if_no_pred: Crashes if no prediction is made. In theory teacher
                should have capability to at least predict something

    Returns:
            [[[xmin, ymin, xmax, ymax, <score>, label], ...] ,... ]
    """

    # Get the correct post processing engine
    if 'EfficientDet' in config['student']:
        post_processing = EfficientDet_post_processing
    else:
        raise Exception("Unsupported student={config['student']}")

    # The bbox regressor depends on the type of efficiendet also!
    if regressBoxes is None and 'EfficientDet' in config['student']:
        if 'YetAnotherEfficientDet' in config['student']:
            regressBoxes = YetAnotherEfficientDetBBoxTransform()
        elif 'EfficientDet' in config['student']:
            regressBoxes = EfficientDetBBoxTransform()

    image_size = config.getint('image_size')
    batch_predictions = post_processing(
        logits=logits,
        valid_classes_dict=valid_classes_dict,
        anchors=anchors,
        config=config,
        regressBoxes=regressBoxes,
        clipBoxes=clipBoxes,
        text_classes=text_classes,
    )

    ground_truth = []
    for i, predictions in enumerate(batch_predictions):
        image_predictions = []
        for pred in predictions:
            xmin = int(max(pred[0], 0))
            ymin = int(max(pred[1], 0))
            xmax = int(min(pred[2], image_size))
            ymax = int(min(pred[3], image_size))
            label = pred[5]
            # Remap from voc/coc if needed
            if not text_classes:
                label = valid_classes_dict['labels_txt2i'][valid_classes_dict['predictions_i2txt'][label]]

            if include_scores:
                image_predictions.append([
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    pred[4],
                    label
                ])
            else:
                image_predictions.append([
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    label
                ])

        # Ignore images where no info was extracted
        if crash_if_no_pred:
            assert len(image_predictions) > 0

        if text_classes:
            ground_truth.append(image_predictions)
        else:
            ground_truth.append(np.array(image_predictions, dtype=np.float32))
    return ground_truth


def filter_model_dict(model, pretrained_dict):
    """
    Purges a model dictionary to read it in the future
    Args:
            model: Model pytorch
            pretrained_dict: A pretrained dictionary from which to extract weights

    Returns:
            A filtered dictionary. Dummy things like parallel module are deleted
    """

    model_dict = model.state_dict()
    for k, v in model_dict.items():
        logger.debug(f"Model dict {k}->{v.shape}")

    mapping = {
        'backbone': 'model_backbones',
        'neck': 'model_necks',
        'backbone_net': 'model_backbones',
        'bifpn': 'model_necks',
        'regressor': 'model_regressor',
        'classifier': 'model_classifier',

        # allow to use a pretrained audio teacher
        'model_backbones.audio': 'backbone_net',
        'model_necks.audio': 'bifpn',
        'model_backbones.thermal': 'backbone_net',
        'model_necks.thermal': 'bifpn',
        'model_backbones.depth': 'backbone_net',
        'model_necks.depth': 'bifpn',
    }

    # 1. filter out unnecessary keys
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        logger.debug(f"Pretrained dict {k}->{v.shape}")

        # ================================================================
        # EfficientDet
        # ================================================================
        # For generator support
        for k_m, v_m in mapping.items():
            # Don't waste time if no mapping is needed
            if k_m not in k:
                continue

            # Empty also to allow for mapping of no modal components
            for modality in ['.audio', '.thermal', '.rgb', '.depth', '']:
                k_gen = k.replace(k_m, f"{v_m}{modality}")
                # EfficientDet Generator
                if k_gen in model_dict:
                    if v.size() == model_dict[k_gen].size():
                        filtered_dict[k_gen] = v
                # EfficientDet Generator
                if f"module.{k_gen}" in model_dict:
                    if v.size() == model_dict[f"module.{k_gen}"].size():
                        filtered_dict[f"module.{k_gen}"] = v

        # ================================================================
        # Network Independent
        # ================================================================
        # For parallel support
        if f"module.{k}" in model_dict:
            if v.size() == model_dict[f"module.{k}"].size():
                filtered_dict[f"module.{k}"] = v

        # For parallel support
        if 'module' in k:
            k_no_module = k[7:]
            if k_no_module in model_dict:
                if v.size() == model_dict[k_no_module].size():
                    filtered_dict[k_no_module] = v

        if k in model_dict:
            if v.size() == model_dict[k].size():
                filtered_dict[k] = v

    logger.debug(f"ModelDict Update:{len(filtered_dict.keys())}/{len(model_dict.keys())}")
    missing_keys = np.setdiff1d(list(model_dict.keys()), list(filtered_dict.keys()))
    logger.debug(f"Missing:{missing_keys}")

    # 2. overwrite entries in the existing state dict
    model_dict.update(filtered_dict)

    return model_dict


def get_data_dim_from_config(config):
    """
    Returns a dictionary with the data dimensions that are needed
    to build a yolo generator model
    Args:
            config: A configuration file from which to probe modalities
    Returns:
            dict({modality:#inputs})
    """
    return_dict = dict()

    # If no modalities it means user rgb
    if config.getboolean('use_thermal'):
        return_dict['thermal'] = 1
    if config.getboolean('use_depth'):
        return_dict['depth'] = 3
    if config.getboolean('use_audio'):
        return_dict['audio'] = 8
    if config.getboolean('use_rgb'):
        return_dict['rgb'] = 3

    if not return_dict:
        return_dict['rgb'] = 3

    return return_dict


def load_model(model_type, config, modality=None):
    """
    Load a pytorch model based on desired type
    Args:
            model_type: The string of the model to load

    Returns:
            A loaded model of type model_type
    """
    model_dict = {
        'EfficientDet' : {
            'path': "trained_models/only_params_trained_EfficientDet",
            'id': None,
            'class': EfficientDet
        },
        'EfficientDet_kaist' : {
            'path': "trained_models/only_params_trained_EfficientDet_kaist",
            'id': None,
            'class': EfficientDet
        },
        'EfficientDet_generator' : {
            'path': "trained_models/only_params_trained_EfficientDet",
            'id': None,
            'class': EfficientDetGenerator
        },
        'YetAnotherEfficientDet_D2' : {
            'path': "trained_models/yet-another-efficientdet-d2.pth",
            #'path': "trained_models/efficientdet-d2.pth",
            'id': None,
            'class': YetAnotherEfficientDet
        },
        'YetAnotherEfficientDet_D2_embedding' : {
            'path': "trained_models/yet-another-efficientdet-d2.pth",
            #'path': "trained_models/efficientdet-d2.pth",
            'id': None,
            'class': YetAnotherEfficientDet
        },
        'YetAnotherEfficientDetGenerator_D2' : {
            'path': "trained_models/yet-another-efficientdet-d2.pth",
            #'path': "trained_models/efficientdet-d2.pth",
            'id': None,
            'class': YetAnotherEfficientDetGenerator
        },
        'YetAnotherEfficientDet_D2_input8' : {
            'path': "trained_models/yet-another-efficientdet-d2.pth",
            #'path': "trained_models/efficientdet-d2.pth",
            'id': None,
            'class': YetAnotherEfficientDet
        },
        'YetAnotherEfficientDet_D2_input1' : {
            'path': "trained_models/yet-another-efficientdet-d2.pth",
            #'path': "trained_models/efficientdet-d2.pth",
            'id': None,
            'class': YetAnotherEfficientDet
        },

        # GOLDEN MODELS
        # -----------------------------------------------------------
        'YetAnotherEfficientDet_D2_individual_student-audio_teacher-rgb_baseline' : {
            'path': "trained_models/individual_student-audio_teacher-rgb_baseline.pth",
            'id': None,
            'class': YetAnotherEfficientDet
        },
        'YetAnotherEfficientDet_D2_individual_student-depth_teacher-rgb' : {
            'path': "trained_models/individual_student-depth_teacher-rgb.pth",
            'id': None,
            'class': YetAnotherEfficientDet
        },
        'YetAnotherEfficientDet_D2_multiteacher_student-audio_teacher-all_pairwisenobohb' : {
            'path': "trained_models/multiteacher_student-audio_teacher-all_pairwisenobohb.pth",
            'id': None,
            'class': YetAnotherEfficientDet
        },
        'YetAnotherEfficientDet_D2_individual_student-audio_teacher-rgb_pairwise' : {
            'path': "trained_models/individual_student-audio_teacher-rgb_pairwise.pth",
            'id': None,
            'class': YetAnotherEfficientDet
        },
        'YetAnotherEfficientDet_D2_individual_student-thermal_teacher-rgb' : {
            'path': "trained_models/individual_student-thermal_teacher-rgb.pth",
            'id': None,
            'class': YetAnotherEfficientDet
        },
        # -----------------------------------------------------------
        'YetAnotherEfficientDet_D2_audio' : {
            'path': "trained_models/yet-another-efficientdet-d2-audio.pth",
            #'path': "trained_models/efficientdet-d2.pth",
            'id': None,
            'class': YetAnotherEfficientDetGenerator
        },
        'YetAnotherEfficientDetMultiHeaded_D2' : {
            'path': "trained_models/yet-another-efficientdet-d2.pth",
            #'path': "trained_models/efficientdet-d2.pth",
            'id': None,
            'class': YetAnotherEfficientDetMultiHeaded
        },
        'YetAnotherEfficientDetGenerator_D2_STATIC' : {
            'path': "trained_models/yet-another-efficientdet-d2-audio-static.pth",
            #'path': "trained_models/efficientdet-d2.pth",
            'id': None,
            'class': YetAnotherEfficientDet
        },
    }

    if model_type not in model_dict:
        logger.exception(f"Unsupported model type {model_type} provided")
        raise Exception(f"Unsupported model type {model_type} provided")

    # To generate a model, one needs to know how many inputs
    # and the type of inputs to be handled
    input_data_config = get_data_dim_from_config(config)

    # In case of multiheaded, we need to create classifier/regressor
    # for each modality
    output_data_config = get_data_dim_from_config(config)

    # If path is None, train from scratch
    if model_dict[model_type]['path'] is None:
        model = model_dict[model_type]['class'](
            input_data_config=input_data_config,
            integration_mode=config['integration_mode']
        )
        return model

    # Make the path modality dependent
    path = model_dict[model_type]['path']
    in_channels = 3
    if modality is not None:
        if modality == 'rgb':
            path = "trained_models/yet-another-efficientdet-d2-rgb.pth"
            in_channels = 3
        elif modality == 'audio_static':
            path = "trained_models/yet-another-efficientdet-d2-audio.pth"
            in_channels = 8
        elif modality == 'audio_student':
            # We don't want to provide the pretained path here
            #path = "trained_models/yet-another-efficientdet-d2-audio.pth"
            in_channels = 8
        elif modality == 'depth':
            path = "trained_models/yet-another-efficientdet-d2-depth.pth"
            in_channels = 3
        elif modality == 'thermal':
            path = "trained_models/yet-another-efficientdet-d2-thermal.pth"
            in_channels = 1
        else:
            raise Exception(f"Unsupported modality={modality} on load model")

    if not os.path.exists(path):
        # download the file
        gdd.download_file_from_google_drive(
            file_id=model_dict[model_type]['id'],
            dest_path=path,
            unzip=False
        )
    logger.debug(f"using path={path}")
    if 'YetAnotherEfficientDet' in model_type:
        model = model_dict[model_type]['class'](
            compound_coef=2,
            input_data_config=input_data_config,
            output_data_config=output_data_config,
            integration_mode=config['integration_mode'],
            in_channels=in_channels,
            features_from=config['features_from'],
        )
    else:
        model = model_dict[model_type]['class'](
            input_data_config=input_data_config,
            integration_mode=config['integration_mode']
        )
    map_location = {'cuda:%d' % 0: f"cuda:{config['rank']}"}
    state_dict = filter_model_dict(model, torch.load(path, map_location=map_location))
    model.load_state_dict(state_dict)
    if 'YetAnotherEfficientDet_D2_embedding' in model_type:
        path = "trained_models/yet-another-efficientdet-d2-embedding.pth"
        logger.debug(f"Incrementally updating the weights with embedding")
        state_dict = filter_model_dict(model, torch.load(path, map_location=map_location))
        model.load_state_dict(state_dict)
    return model


def make_reproducible_run(seed):
    """
    A bunch of recommendations to make a run reproducible
    If a negative seed is provided, nothing is done
    Args:
            seed: the seed value to be able to reproduce run

    Returns:
            Nothing
    """
    if seed < 1:
        logger.warn(f"No valid seed provided: {seed}")
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    logger.warn(f"Using seed {seed}")

def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = str(re.sub('[^\w\s-]', '_', value).strip().lower())
    value = str(re.sub('[-\s]+', '-', value))
    # ...
    return value

def plot_image_predictions(
        source,
        config,
        model=None,
        id=None,
):
    """
    Plots a nice image with predicitons for visual debug
    Args:
            source: From where to get the data to plot
            config: A config object to know how to run
            anchors: Yolo anchors fro prediction processing
            model: Optional model to predict

    Returns:
            None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Print options
    np.set_printoptions(suppress=True, precision=3)

    # Read input image
    rgb_size = config.getint('image_size')
    if id is None:
        id = np.random.randint(len(source))
    else:
        id = source.ids.index(id)

    # Extract labels always
    rgb, thermal, depth, audio, label, id = source[id]
    rgb_path, thermal_path, depth_path, audio_paths, rgb_xml_path = source.get_paths(id)

    # Keep originals
    orig_image = cv2.imread(rgb_path)
    print(f"orig_image={orig_image.shape}")
    orig_labels = None
    if label is not None:
        orig_labels = source.get_annotations(id)
        print(f"orig_labels={orig_labels}")
    height, width, channels = orig_image.shape

    modalities = {}

    rgb_image = cv2.imread(rgb_path)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    rgb = rgb[None, :, :, :]
    rgb = Variable(torch.FloatTensor(rgb)).to(device)
    modalities['rgb'] = rgb
    print(f"Using rgb={rgb.shape}")

    thermal_image = None
    if config.getboolean('use_thermal'):
        thermal_image = np.copy(thermal.squeeze())
        thermal_image = thermal_image *255.
        thermal_image = thermal.astype(np.int16)
        height_thermal, width_thermal = thermal_image.shape
        thermal = thermal[None, :, :]
        thermal = Variable(torch.FloatTensor(thermal)).to(device)
        modalities['thermal'] = thermal
        print(f"Using thermal={thermal.shape}")

    depth_image = None
    if config.getboolean('use_depth'):
        depth_image = readPmf(depth_path)
        depth_image = applyLogJetColorMap(depth)
        height_depth, width_depth, _ = depth_image.shape
        #depth = depth[None, :, :]
        depth = depth[None, :, :, :]
        depth = Variable(torch.FloatTensor(depth)).to(device)
        modalities['depth'] = depth
        print(f"Using depth={depth.shape}")

    audio_image = None
    if config.getboolean('use_audio'):
        audio = audio[None, :, :, :]
        audio = Variable(torch.FloatTensor(audio)).to(device)
        modalities['audio'] = audio
        print(f"Using audio={audio.shape}")

    # Resize and other transformations
    rgb_image, thermal_image, depth_image, audio_image, _, id = source.transformations(
        (
            rgb_image, thermal_image, depth_image, audio_image, None, id
        )
    )

    # Convert the label to xmax/ymax for mat
    images_to_plot = []

    anchors = None
    if 'yolov2' in config['teacher']:
        if type(model) == torch.nn.DataParallel:
            anchors = model.module.anchors
        else:
            anchors = model.anchors

    if label is None:
        with torch.no_grad():
            teacher_models = torch.nn.ModuleDict()
            if config.getboolean('use_rgb'):
                teacher_models['rgb'] = load_model(config['teacher'], config, 'rgb').to(device)
            if config.getboolean('use_audio'):
                teacher_models['audio'] = load_model(config['teacher'], config, 'audio_static').to(device)
            if config.getboolean('use_depth'):
                teacher_models['depth'] = load_model(config['teacher'], config, 'depth').to(device)
            if config.getboolean('use_thermal'):
                teacher_models['thermal'] = load_model(config['teacher'], config, 'thermal').to(device)

            for modality, teacher_model in teacher_models.items():
                teacher_model.eval()
                logits, features = teacher_model(modalities[modality])

                label[modality] = logits_to_ground_truth(
                    logits=logits,
                    anchors=anchors,
                    valid_classes_dict=source.valid_classes_dict,
                    config=config,
                    include_scores=True,
                    crash_if_no_pred=False,
                    text_classes=True,
                )[0]
    else:
        label_new = {}
        for modality in teacher_models.keys():
            label_new[modality] = label
        label = label_new

    if orig_labels is not None:
        for orig in orig_labels:
            print(f"({id}): Expected orig label={orig}")

    for expected in label:
        print(f"({id}): Expected transformed label={expected}")

    # Do the teacher predictions just once
    if model is not None:
        with torch.no_grad():
            model.eval()
            logits_s, features_s = model(modalities)

    for modality in teacher_models.keys():
        common_size = rgb.shape[-1]
        if height > width:
            scale = common_size / height
            resized_height = common_size
            resized_width = int(width * scale)
        else:
            scale = common_size / width
            resized_height = int(height * scale)
            resized_width = common_size
        images_to_plot.append({
            'data': orig_image,
            'bbox': orig_labels if orig_labels is not None else label[modality],
            'title': "RGBGT",
            'type': 'image',
            'width_ratio':  1 if orig_labels is not None else float(resized_width) / width,
            'height_ratio': 1 if orig_labels is not None else float(resized_height) / height,
        })
        if config.getboolean('use_thermal'):
            common_size_thermal = thermal.shape[-1]
            if height_thermal > width_thermal:
                scale_thermal = common_size_thermal / height_thermal
                resized_height_thermal = common_size_thermal
                resized_width_thermal = int(width_thermal * scale_thermal)
            else:
                scale_thermal = common_size_thermal / width_thermal
                resized_height_thermal = int(height_thermal * scale_thermal)
                resized_width_thermal = common_size_thermal
            images_to_plot.append({
                'data': thermal_image,
                'bbox': label[modality],
                'title': "Thermal",
                'type': 'image',
                'width_ratio':  1,
                'height_ratio': 1,
            })
        if config.getboolean('use_depth'):
            common_size_depth = depth.shape[-1]
            if height_depth > width_depth:
                scale_depth = common_size_depth / height_depth
                resized_height_depth = common_size_depth
                resized_width_depth = int(width_depth * scale_depth)
            else:
                scale_depth = common_size_depth / width_depth
                resized_height_depth = int(height_depth * scale_depth)
                resized_width_depth = common_size_depth
            images_to_plot.append({
                'data': depth_image,
                'bbox': label[modality],
                'title': "Depth",
                'type': 'image',
                'width_ratio':  1 if orig_labels is not None else float(resized_width_depth) / width_depth,
                'height_ratio': 1 if orig_labels is not None else float(resized_height_depth) / height_depth,
            })
        if config.getboolean('use_audio'):
            for audio_path in audio_paths:
                y, sr = librosa.load(audio_path.replace('pkl', 'wav'), sr=44100)
                images_to_plot.append({
                    'data': y,
                    'bbox': None,
                    'title': "Audio",
                    'type': 'waveplot'
                })

        # Predict the output of the model instead of GT
        if model is not None:
            prediction = logits_to_ground_truth(
                logits=logits_s[modality],
                anchors=anchors,
                valid_classes_dict=source.valid_classes_dict,
                config=config,
                include_scores=True,
                crash_if_no_pred=False,
                text_classes=True,
            )[0]
            for pred in prediction:
                print(f"Predicted {modality }={pred}")
            if len(prediction) > 0:
                images_to_plot.append({
                    'data': rgb_image,
                    'bbox': prediction,
                    'title': "Prediction_Resized",
                    'type': 'image',
                    'width_ratio': 1,
                    'height_ratio': 1,
                })
                if orig_labels is not None:
                    prediction_scaled = []
                    for pred in prediction:
                        prediction_scaled.append([
                            pred[0]/scale,
                            pred[1]/scale,
                            pred[2]/scale,
                            pred[3]/scale,
                            pred[4],
                            pred[5],
                        ])
                    print(f"\nprediction_scaled {modality} ({scale}) =")
                    for pred in prediction_scaled:
                        print(f"Predicted={pred}")
                    images_to_plot.append({
                        'data': orig_image,
                        'bbox': prediction_scaled,
                        'title': "Prediction",
                        'type': 'image',
                        'width_ratio': 1,
                        'height_ratio': 1,
                    })

        # Plot the image
        id = slugify(id)
        color = colors[source.classes.index(class_name)]
        color = {
            'rgb': 'Blue',
            'thermal': 'Red',
            'audio': 'Green',
            'depth': 'Yellow'
        }
        for i, modality_image in enumerate(images_to_plot):

            # Add bbox into the image
            if modality_image['bbox'] is not None:
                for pred in modality_image['bbox']:
                    # Change to str if required
                    if not isinstance(pred[-1], str):
                        class_name = source.valid_classes_dict['labels_i2txt'][int(pred[-1])]
                    else:
                        class_name = pred[-1]

                    # If from prediction, scale to orig size
                    # Labels from GT are 0,1 from pred are 2,3 if any

                    #  Do not disturbe orig labels
                    width_ratio  = modality_image['width_ratio']
                    height_ratio = modality_image['height_ratio']


                    xmin = int(max(pred[0] / width_ratio, 0))
                    ymin = int(max(pred[1] / height_ratio, 0))
                    xmax = int(min(pred[2] / width_ratio, width))
                    ymax = int(min(pred[3] / height_ratio, height))

                    colors = pickle.load(open("src/utils/pallete", "rb"))
                    cv2.rectangle(modality_image['data'], (xmin, ymin), (xmax, ymax), color[modality], 2)
                    text_size = cv2.getTextSize(
                        class_name,
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        1
                    )[0]
                    cv2.rectangle(
                        modality_image['data'],
                        (xmin, ymin),
                        (xmin + text_size[0] + 3, ymin + text_size[1] + 4),
                        color[modality],
                        -1
                    )
                    cv2.putText(
                        modality_image['data'], class_name,
                        (xmin, ymin + text_size[1] + 4),
                        cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255),
                        1
                    )

    # Regardless of the teacher modality print the image
    for i, modality in enumerate(images_to_plot):
        # Plot the modality
        if modality_image['type'] == 'image':
            cv2.imwrite(
                f"{id}_image_{ modality['title'] }.jpg",
                modality['data'].astype('float32'),
            )
        if modality['type'] == 'waveplot':
            librosa.display.waveplot(modality['data'], sr=44100, alpha=0.25)
            plt.savefig(f"{id}_waveplot_{ modality['title']  }.jpg")
            plt.close()
        if modality['type'] == 'specshow':
            librosa.display.specshow(
                modality['data'],
                x_axis='time',
                y_axis='mel',
                sr=44100,
                fmax=8000
            )
            plt.savefig(f"{id}_specshow_{ modality['title']  }.jpg")
            plt.close()

    return


def start_boardx_logger(config):
    """
    Starts a logger to track loss and other metricss
    Args:
            config: A config object to know how to run

    Returns:
            A writer object for tensorboardx
    """
    log_path = os.path.join(
        config['exp_name'],
        config['rank'],
        datetime.now().strftime('mylogfile_%H_%M_%d_%m_%Y.log'),
    )
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    return writer


def closest_point(node, nodes):
    """
    Find which is the clossest point from a list of points
    Args:
            node: value to search
            nodes: List of values in which to search

    Returns:
            Clossest location of object
    """
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)


def get_batch_central_distances(outputs, targets, width, height):
    """
    IMplementation of central distances as stated in
    https://arxiv.org/pdf/1910.11760.pdf
    Args:
            Outputs: predictions of the model
            targets: The expected prediction of the model
            width and height of the images

    Returns:
            The error of not predicting the target, how far away the
            regression object is from the goal
    """
    cd_x, cd_y = [], []
    for sample_i in range(len(outputs)):

        target = np.array(targets[sample_i])
        # Maybe this is correct, if not target, nothing to check right?
        if len(target) < 1:
            continue
        target_point = target[:, 2:4] - target[:, 0:2]
        target_labels = target[:, -1]

        # If no valid predictions, dont produce anything
        # the AP calculation would handle this
        output = np.array(outputs[sample_i])
        if len(output) < 1:
            logger.debug(f"No output for this sample {sample_i} in {output}")
            pred_labels = np.zeros_like(target_labels)
            output_point = np.zeros_like(target_point)
        else:
            pred_labels = output[:, -1]
            output_point = output[:, 2:4] - output[:, 0:2]

        distance_x = []
        distance_y = []
        for i in range(len(target_point)):

            # Only compare desired target
            label = target_labels[i]
            valid_points = output_point[pred_labels == label]
            orig_indexes = np.arange(len(pred_labels))[pred_labels == label]
            if len(valid_points) < 1:
                # if no prediction, assume 0 point prediction
                distance_x.append(target_point[i, 0])
                distance_y.append(target_point[i, 1])
            else:
                # Search for the closes point
                index_closest = closest_point(target_point[i], valid_points)

                # If this point was used, take it out of the picture
                pred_labels[orig_indexes[index_closest]] = -1

                distance_x.append(
                    np.abs(target_point[i, 0] - valid_points[index_closest, 0])
                )
                distance_y.append(
                    np.abs(target_point[i, 1] - valid_points[index_closest, 1])
                )
        cd_x.append(np.mean(distance_x) / width)
        cd_y.append(np.mean(distance_y) / height)

    return cd_x, cd_y


def get_batch_statistics(outputs, targets, iou_threshold, add_detected = False):
    """
    Compute true positives, predicted scores and predicted labels per sample
    Modified version from https://github.com/eriklindernoren/PyTorch-YOLOv3
    @article{yolov3,
      title={YOLOv3: An Incremental Improvement},
        author={Redmon, Joseph and Farhadi, Ali},
          journal = {arXiv},
            year={2018}
            }
    Args:
            outputs: Prediciton of the model
            targets: the ground truth
            iou_threshold: Target IOU that makes a prediction correct

    Returns:
            The true positives and labels from the prediction
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_metrics = []
    for sample_i in range(len(outputs)):

        output = np.array(outputs[sample_i])
        output = torch.from_numpy(output).float().to(device)

        # If no valid predictions, dont produce anything
        # the AP calculation would handle this
        if len(output) < 1:
            continue

        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        target = np.array(targets[sample_i])
        # Maybe this is correct, if no target, then
        # there is nothing to check really
        if len(target) < 1:
            continue
        annotations = torch.from_numpy(target).float().to(device)
        target_boxes = annotations[:, :4]
        target_labels = annotations[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        detected_boxes = []

        for pred_i, (box, label) in enumerate(zip(pred_boxes, pred_labels)):

            # If targets are found break
            if len(detected_boxes) == len(annotations):
                break

            # Ignore if label is not one of the target labels
            if label not in target_labels:
                continue

            iou, box_index = bbox_iou(box.unsqueeze(0), target_boxes).max(0)
            if iou >= iou_threshold and box_index not in detected_boxes:
                true_positives[pred_i] = 1
                detected_boxes += [box_index]
        if add_detected:
            mask_array = np.zeros(target_boxes.shape[0])
            for i in range(len(detected_boxes)):
                mask_array[detected_boxes[i].cpu().numpy().item(0)] = 1

            batch_metrics.append([
                true_positives,
                mask_array,
                pred_scores.cpu().numpy(),
                pred_labels.cpu().numpy()
            ])
        else:
            batch_metrics.append([
                true_positives,
                pred_scores.cpu().numpy(),
                pred_labels.cpu().numpy()
            ])
    return batch_metrics


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Computes the IOU given 2 boxes
    Modified version from https://github.com/eriklindernoren/PyTorch-YOLOv3
    @article{yolov3,
      title={YOLOv3: An Incremental Improvement},
        author={Redmon, Joseph and Farhadi, Ali},
          journal = {arXiv},
            year={2018}
            }
    Args:
            box1: candidate to obtain IOU
            box2: baseline for the IOU comparisson
            x1y1x2y2= Transformation from center to width

    Returns:
            IOU score withing 2 boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(
        inter_rect_x2 - inter_rect_x1 + 1, min=0
    ) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []

    # We need a metric to know if we are making progress
    # If we treat all classes same, we can take the total number of
    # objects vs total predictions
    total_gt, total_p = 0.0, 0.0
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        total_gt += n_gt
        total_p += n_p

        logger.debug(f"For class {c} n_gt={n_gt} and n_p={n_p}")

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            logger.debug(f"For class {c} fpc={fpc} tpc={tpc}")

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32"), total_p/total_gt


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_multiteacher(
    teacher_models,
    student_model,
    test_set,
    config,
):
    """
    Standart Evaluation Metrics
    Args:
            teacher_model: GT generator on prediciton
            student_model: The model that is being evaluated
            test_set: A generator of data
            config: guides how the run was configurated

    Returns:
            AVG Score of the prediction
    """

    logger.warn(f"\nBeginning evaluation of student model performance")

    # Calculate AP table
    ap_table = []

    # Get the prediction
    start_time = time.time()
    all_predictions, all_labels, labels = get_predictions(
        teacher_models,
        student_model,
        test_set,
        config,
    )

    # account for resources in the model quality
    elapsed_time = time.time() - start_time
    pytorch_total_params = sum(p.numel() for p in student_model.parameters())
    pytorch_total_params_trainable = sum(
        p.numel() for p in student_model.parameters() if p.requires_grad
    )
    resources = pd.DataFrame([{
        'model' : config['student'],
        'Time2Predict' : elapsed_time,
        'TotalParams' : pytorch_total_params,
        'TrainParams' :pytorch_total_params_trainable,
    }])
    logger.warn("\n" + tabulate(resources, headers='keys', tablefmt='psql'))
    if os.path.exists(f"{config['exp_name']}"):
        resources.to_csv(f"{config['exp_name']}/resources.{config['rank']}.csv", index=False)

    for modality in teacher_models.keys():
        logger.debug(f"Working on modality={modality}")
        ap_modality = {
            'exp_name': config['exp_name'],
            'modality': modality,
            'AP@Ave': [0.],
            'AP@0.5': [0.],
            'AP@0.75': [0.],
            'CDx': [0.],
            'CDy': [0.]
        }
        # Update the ap table per prediction
        ap_record = []
        for IOU in np.arange(0.5, 0.95, 0.05):
            # Make sure no extra decimal due to div
            IOU = np.around(IOU, decimals=2)

            sample_metrics = []  # List of tuples (TP, confs, pred)
            cd_x, cd_y = [], []
            for batch_predictions, batch_labels in tqdm(zip(all_predictions[modality], all_labels[modality]), desc=f"Ap@{IOU}", total=len(all_labels[modality])):
                logger.debug(f"batch_predictions={np.array(batch_predictions).shape}")
                logger.debug(f"batch_labels={np.array(batch_labels).shape}")
                sample_metrics += get_batch_statistics(
                    batch_predictions,
                    batch_labels,
                    IOU,
                )
                # Get the cd_x and cd_y distances
                cdx, cdy = get_batch_central_distances(
                    batch_predictions,
                    batch_labels,
                    config.getint('image_size'),
                    config.getint('image_size'),
                )
                cd_x.extend(cdx)
                cd_y.extend(cdy)

            # Concatenate sample statistics
            if not any(sample_metrics):
                logger.error(f"No valid prediction was made!!")
                precision = [0.]
                recall = [0.]
                AP = [0.]
                f1 = [0.]
                ap_class = [0]
                score = [0.]
                cd_x = [100.]
                cd_y = [100.]
            else:
                true_positives, pred_scores,  pred_labels = [
                    np.concatenate(x, 0) for x in list(zip(*sample_metrics))
                ]
                precision, recall, AP, f1, ap_class, score = ap_per_class(
                    true_positives,
                    pred_scores,
                    pred_labels,
                    labels[modality]
                )

            mean = 0.0
            if hasattr(AP, 'mean'):
                mean = AP.mean()
            if IOU == 0.5:
                ap_class_txt = np.frompyfunc(lambda s: test_set.classes[s], 1, 1)(ap_class)
                df = pd.DataFrame(
                    np.stack((ap_class_txt, precision, recall, AP, f1), axis=-1),
                    index=ap_class,
                    columns=['class', 'precision', 'recal', 'AP', 'F1']
                )
                logger.warn(f"AP per class {modality} for IOU=0.5")
                logger.warn("\n" + tabulate(df, headers='keys', tablefmt='psql'))

                ap_modality['AP@0.5'] = mean * 100

                # CD should not be IOU dependent but add here to make it efficient
                ap_modality['CDx'] = np.mean(cd_x) * 100
                ap_modality['CDy'] = np.mean(cd_y) * 100
            if IOU == 0.75:
                ap_modality['AP@0.75'] = mean * 100
            logger.debug(f"IOU={IOU} AP.mean()={mean}")
            ap_record.append(mean)

        ap_modality['AP@Ave'] = np.mean(ap_record) * 100
        ap_table.append(ap_modality)
    ap_table = pd.DataFrame(ap_table)
    logger.warn("\n" + tabulate(ap_table, headers='keys', tablefmt='psql'))
    if os.path.exists(f"{config['exp_name']}"):
        ap_table.to_csv(f"{config['exp_name']}/results.{config['rank']}.csv", index=False)

    return


def extract_transformations(
    config,
    mode,
    rgb_size,
    thermal_size,
    depth_size,
    audio_size,
):
    """
    Extract what transformation to use from a config file
    Args:
            config: A config object to know how to run
            mode: train or validation

    Returns:
            List of transformations
    """
    if mode == "train":
        config_trans = config['train_transformations']
    elif mode == "val":
        config_trans = config['val_transformations']
    else:
        raise Exception(f"No valid mode provided")
    transformations = []
    for trans in config_trans.split(','):
        logger.info(f"Adding {mode} trans={trans}")
        if trans == 'HSVAdjust':
            transformations.append(HSVAdjust())
        elif trans == 'Resize':
            transformations.append(Resize(
                rgb_size,
                thermal_size,
                depth_size,
                audio_size,
            ))
        elif trans == 'Resizer':
            transformations.append(Resizer(
                common_size=rgb_size,
            ))
        elif trans == 'Normalizer':
            transformations.append(Normalizer())
        elif trans == 'ThermalAugmenter':
            transformations.append(ThermalAugmenter())
        elif trans == 'AudioAugmenter':
            transformations.append(AudioAugmenter())
        else:
            raise Exception(f"No valid transformation {trans} provided")
    return transformations


def weights_init(m):
    """
    Applies a weight initializaiton to a model
    Args:
            m: Model into which apply weights init

    Returns:
            None
    """
    classname = m.__class__.__name__
    if classname.find('ConvModule') == -1 and classname.find('Conv2dStaticSamePadding') ==-1 and classname.find('SeparableConvBlock') ==-1 :
        if classname.find('Conv') != -1:
            print(classname)
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


def readPmf(filepath, max_value=192):

    """
    Taken from https://github.com/pat676/MLPDepthMap/blob/master/src/MiddleburyUtil.py
    Reads the middlebury pmf files and returns a numpy array
    Args:
        Filepath(String): The path of the pmf file
    """

    with open(filepath, "rb") as f:

        # Only works for grayscale
        imgType = f.readline().decode('utf-8').rstrip()
        assert imgType == "Pf", "pmf file not grayscale"

        # Read header
        width, height = f.readline().split()
        width, height = int(width), int(height)
        scaleFactor = float(f.readline().rstrip())

        # Determine endian
        endian = '<' if scaleFactor < 0 else '>'

        data = np.fromfile(f, endian + 'f')

    img = np.reshape(data, (height, width))
    img[img == np.inf] = 0
    img[img >= max_value] = max_value
    #img = np.flip(img, 0)
    return img


def applyLogJetColorMap(img, logScale=False, maxValue=0):
    """
    Taken from https://github.com/pat676/MLPDepthMap/blob/master/src/MiddleburyUtil.py
    Converts image to logaritmic color map
    Args:
        img([[Double]])  : The image
        logScale(Bool)   : Performs a logaritmic scaling of the colormap
                           if true
        maxValue(float)  : The maximum value for the colormap, will be set
                           to img.max() if value is zero
    """

    if(maxValue == 0):
        maxValue = img.max()

    colorMapImg = img/maxValue
    colorMapImg *= 255

    if(logScale):
        colorMapImg[colorMapImg < 1] = 1
        colorMapImg = np.log(colorMapImg)
        colorMapImg = colorMapImg/colorMapImg.max()
        colorMapImg *= 255

    colorMapImg = colorMapImg.astype(np.uint8)
    colorMapImg = cv2.applyColorMap(colorMapImg, cv2.COLORMAP_JET)

    return colorMapImg


def extract_criterions_from_config(config, train_set):
    """
    Parses a config file and extract the criterions to be used
    Args:
            config: A parsed configuration file.

    Returns:
            The main, divergence and alignment criterion to be used
    """
    anchors = None
    if 'yolov2' in config['teacher']:
        #if type(teacher_model) == torch.nn.DataParallel:
        #    anchors = teacher_model.module.anchors
        #else:
        #    anchors = teacher_model.anchors
        raise Exception('YOLO IS NO LONGER SUPPORTED')

    if config['main_loss'] == 'YoloLoss':
        criterion_main = YoloLoss(
            train_set.num_classes,
            anchors,
            reduction=32,
        )
    elif config['main_loss'] == 'FocalLoss':
        criterion_main = FocalLoss()
    elif config['main_loss'] == 'YetAnotherFocalLoss':
        criterion_main = YetAnotherFocalLoss()
    else:
        raise Exception(f"Unsupported Main Loss {config['main_loss']}")

    # KD at output is traditionally KL divergence
    if config['div_loss'] == 'DistillKL':
        criterion_div = DistillKL()
    elif config['div_loss'] == 'None':
        criterion_div = None
    else:
        raise Exception(f"Unsupported DIV Loss {config['div_loss']} ")

    # Specialized Losses for KD
    if config['kd_loss'] == 'SimilarityLoss':
        criterion_kd = SimilarityLoss()
    elif config['kd_loss'] == 'MultiTeacherPairWiseSimilarityLoss':
        criterion_kd = MultiTeacherPairWiseSimilarityLoss()
    elif config['kd_loss'] == 'AttentionLoss':
        criterion_kd = AttentionLoss()
    elif config['kd_loss'] == 'ABLoss':
        criterion_kd = ABLoss()
    elif config['kd_loss'] == 'MTALoss':
        criterion_kd = MTALoss(T=config['T'], p=config['p'])
    elif config['kd_loss'] == 'KLLoss':
        criterion_kd = KLLoss(T=config['T'], p=config['p'])
    elif config['kd_loss'] == 'MultiTeacherTrippletAttentionLoss':
        criterion_kd = MultiTeacherTrippletAttentionLoss()
    elif config['kd_loss'] == 'NSTLoss':
        criterion_kd = NSTLoss()
    elif config['kd_loss'] == 'PKTLoss':
        criterion_kd = PKTLoss()
    elif config['kd_loss'] == 'RankingLoss':
        criterion_kd = RankingLoss(delta=0.2)
    elif config['kd_loss'] == 'CRDLoss':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rgb, thermal, label, idx = train_set[0]
        rgb = rgb[None, :, :, :]
        rgb = Variable(
            torch.from_numpy(rgb).float().to(device),
            requires_grad=False
        ).to(device)
        with torch.no_grad():
            logits_t, features_t = teacher_model(rgb)
        f_shape = features_t.view(features_t.shape[0], -1).shape[1]
        criterion_kd = CRDLoss(
            s_dim=f_shape,
            # s_dim: the dimension of student's feature
            t_dim=f_shape,
            # t_dim: the dimension of teacher's feature
            n_data=len(train_set),
        )
        criterion_kd.to(device)
        criterion_kd.embed_s.to(device)
        criterion_kd.embed_s.train()
        criterion_kd.embed_t.to(device)
        criterion_kd.embed_t.train()
    elif config['kd_loss'] == 'MultiTeacherContrastiveAttentionLoss':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rgb, thermal, label, idx = train_set[0]
        rgb = rgb[None, :, :, :]
        rgb = Variable(
            torch.from_numpy(rgb).float().to(device),
            requires_grad=False
        ).to(device)
        with torch.no_grad():
            logits_t, features_t = teacher_model(rgb)
        f_shape = features_t.view(features_t.shape[0], -1).shape[1]
        criterion_kd = MultiTeacherContrastiveAttentionLoss(
            s_dim=f_shape,
            # s_dim: the dimension of student's feature
            t_dim=f_shape,
            # t_dim: the dimension of teacher's feature
            n_data=len(train_set),
        )
        criterion_kd.to(device)
        criterion_kd.embed_s.to(device)
        criterion_kd.embed_s.train()
        criterion_kd.embed_t.to(device)
        criterion_kd.embed_t.train()
    elif config['kd_loss'] == 'MSELoss':
        criterion_kd = nn.MSELoss()
    elif config['kd_loss'] == 'None':
        criterion_kd = None
    else:
        raise Exception(f"Unsupported KD Loss {config['kd_loss']} provided")

    return criterion_main, criterion_div, criterion_kd


def str2bool(v):
    """Utility to parse boolean from command line"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_weights(model):
    for name, module in model.named_modules():
        is_conv_layer = isinstance(module, nn.Conv2d)

        if is_conv_layer:
            if "conv_list" or "header" in name:
                variance_scaling_(module.weight.data)
            else:
                nn.init.kaiming_uniform_(module.weight.data)

            if module.bias is not None:
                if "classifier.header" in name:
                    bias_value = -np.log((1 - 0.01) / 0.01)
                    torch.nn.init.constant_(module.bias, bias_value)
                else:
                    module.bias.data.zero_()


def variance_scaling_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""
    initializer for SeparableConv in Regressor/Classifier
    reference: https://keras.io/zh/initializers/  VarianceScaling
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(gain / float(fan_in))

    return _no_grad_normal_(tensor, 0., std)


def isListEmpty(inList):
    if isinstance(inList, list):  # Is a list
        return all(map(isListEmpty, inList))
    if hasattr(inList, 'size') and inList.size == 0:
        return True
    return False  # Not a list


def get_predictions_multiteacher(
    teacher_models,
    student_model,
    test_set,
    config,
):
    """
    Gets the prediciton of a student model per batch
    Args:
            teacher_models: Teacher model dict to define ground truth
            student_model: Student to actually do the predictions
            test_set: the source that yields data
            config: A config object that dictates how run should go

    Returns:
            All predictions from the model. High memory but for speed
            of evaluation!!
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    anchors = None

    student_model.eval()

    # Get the training validator
    generator = DataLoader(
        test_set,
        batch_size=config.getint('batch_size'),
        shuffle=False,
        drop_last=False,
        collate_fn=custom_collate_factory(config),
        num_workers=config.getint('num_workers'),
        pin_memory=True,
    )
    """Evaluates how good a model was trained"""
    all_labels, all_predictions, labels = [], [], []
    for test_iter, test_batch in enumerate(tqdm(generator, desc=f"Get Pred ALL")):

        start_time = time.time()
        rgb, thermal, depth, audio, label, id = test_batch

        rgb = rgb.to(device)
        audio = audio.to(device)
        if config.getboolean('use_thermal'):
            thermal = thermal.to(device)
        if config.getboolean('use_depth'):
            depth = depth.to(device)

        #print(f"Time to get the batch={time.time() - start_time}")
        start_time = time.time()

        with torch.no_grad():
            if config['student_modality'] == 'audio':
                logits_s, features_s = student_model(audio)
            elif config['student_modality'] == 'depth':
                logits_s, features_s = student_model(depth)
            elif config['student_modality'] == 'thermal':
                logits_s, features_s = student_model(thermal)
            batch_labels = [[] for i in range(rgb.shape[0])]
            for modality, teacher_model in teacher_models.items():
                if modality == 'rgb':
                    prediction, features = teacher_model(rgb)
                elif modality == 'audio':
                    prediction, features = teacher_model(audio)
                elif modality == 'thermal':
                    prediction, features = teacher_model(thermal)
                elif modality == 'depth':
                    prediction, features = teacher_model(depth)
                else:
                    raise ValueError('No valid modality to predict from teacher')
                if config.getboolean('use_labels'):
                    this_batch_labels = label
                else:
                    this_batch_labels = logits_to_ground_truth(
                        logits=prediction,
                        anchors=anchors,
                        valid_classes_dict=test_set.valid_classes_dict,
                        config=config,
                        include_scores=True,
                    )

                # Integrate
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
                    #print(f"batch_labels[i] ({np.shape(batch_labels[i])})={batch_labels[i]}")
                    batch_labels[i] = np.concatenate((batch_labels[i], this_batch_labels[i]), axis=0)

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

                #keep nms index
                batch_labels[i] = batch_labels[i][idx]

            #if not all([np.any(elem) for elem in batch_labels]):
            if isListEmpty(batch_labels):
                logger.debug(f"{test_iter}: predictions={[]}")
                all_predictions.append([])
                logger.debug(f"{test_iter}: ground_truth={batch_labels}")
                all_labels.append(batch_labels)

                continue

            for label in batch_labels:
                if isListEmpty(label):
                    labels += []
                else:
                    labels += label[:, 4].tolist()

            #logits_s, features_s = student_model(audio)

            batch_predictions = logits_to_ground_truth(
                logits=logits_s,
                anchors=anchors,
                valid_classes_dict=test_set.valid_classes_dict,
                config=config,
                include_scores=True,
                crash_if_no_pred=False,
            )

            # The label of the dataset and the label from the ground truth
            # Differ in the resize step. that is, label
            # from dataset is [xmin, ymin, xmax, ymax, class]
            # but the model predicts and should be trained
            # with [xmin, ymin, width, height, class]

            # I need to add an IOU metric
            logger.debug(f"{test_iter}: id={id}")
            logger.debug(f"{test_iter}: predictions={batch_predictions}")
            all_predictions.append(batch_predictions)
            logger.debug(f"{test_iter}: ground_truth={batch_labels}")
            all_labels.append(batch_labels)

            # Save the prediction to a file
            for i in range(rgb.shape[0]):
                drive, rgb_timestamp = id[i].split('/')
                directory = os.path.join(test_set.data_path,drive,'annotations')
                if not os.path.exists(directory):
                    os.mkdir(directory)
                path = os.path.join(
                    directory,
                    f"{rgb_timestamp}.all.txt",
                )
                if not os.path.exists(path):
                    np.savetxt(path, batch_predictions[i], delimiter=',')

    return all_predictions, all_labels, labels


def get_predictions(
    teacher_model,
    student_model,
    test_set,
    config,
    modality,
):
    """
    Gets the prediciton of a student model per batch
    Args:
            teacher_model: Teacher model to define ground truth
            student_model: Student to actually do the predictions
            test_set: the source that yields data
            config: A config object that dictates how run should go

    Returns:
            All predictions from the model. High memory but for speed
            of evaluation!!
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    anchors = None
    if 'yolov2' in config['teacher']:
        if type(teacher_model) == torch.nn.DataParallel:
            anchors = teacher_model.module.anchors
        else:
            anchors = teacher_model.anchors

    student_model.eval()

    # Get the training validator
    generator = DataLoader(
        test_set,
        batch_size=config.getint('batch_size'),
        shuffle=False,
        drop_last=False,
        collate_fn=custom_collate_factory(config),
        num_workers=config.getint('num_workers'),
        pin_memory=True,
    )
    """Evaluates how good a model was trained"""
    all_labels, all_predictions, labels = [], [], []
    for test_iter, test_batch in enumerate(tqdm(generator, desc=f"Get Pred {modality}")):

        start_time = time.time()
        rgb, thermal, depth, audio, label, id = test_batch

        rgb = rgb.to(device)
        audio = audio.to(device)
        if config.getboolean('use_thermal'):
            thermal = thermal.to(device)
        if config.getboolean('use_depth'):
            depth = depth.to(device)

        #print(f"Time to get the batch={time.time() - start_time}")
        start_time = time.time()

        with torch.no_grad():
            if config['student_modality'] == 'audio':
                logits_s, features_s = student_model(audio)
            elif config['student_modality'] == 'depth':
                logits_s, features_s = student_model(depth)
            elif config['student_modality'] == 'thermal':
                logits_s, features_s = student_model(thermal)


            if modality == 'rgb':
                prediction, features = teacher_model(rgb)
            elif modality == 'audio':
                prediction, features = teacher_model(audio)
            elif modality == 'thermal':
                prediction, features = teacher_model(thermal)
            elif modality == 'depth':
                prediction, features = teacher_model(depth)
            else:
                raise ValueError('No valid modality to predict from teacher')
            if config.getboolean('use_labels'):
                batch_labels = label
            else:
                batch_labels = logits_to_ground_truth(
                    logits=prediction,
                    anchors=anchors,
                    valid_classes_dict=test_set.valid_classes_dict,
                    config=config,
                )
            if isListEmpty(batch_labels):
                logger.debug(f"{test_iter}: predictions={[]}")
                all_predictions.append([])
                logger.debug(f"{test_iter}: ground_truth={batch_labels}")
                all_labels.append(batch_labels)
                continue

            for label in batch_labels:
                if isListEmpty(label):
                    labels += []
                else:
                    labels += label[:, 4].tolist()

            batch_predictions = logits_to_ground_truth(
                logits=logits_s,
                anchors=anchors,
                valid_classes_dict=test_set.valid_classes_dict,
                config=config,
                include_scores=True,
                crash_if_no_pred=False,
            )
            #print(f"to get student labels  = {time.time()-start_time}")
            #print(f"Time post processing ={time.time() - start_time}")
            # The label of the dataset and the label from the ground truth
            # Differ in the resize step. that is, label
            # from dataset is [xmin, ymin, xmax, ymax, class]
            # but the model predicts and should be trained
            # with [xmin, ymin, width, height, class]

            # I need to add an IOU metric
            logger.debug(f"{test_iter}: id={id}")
            logger.debug(f"{test_iter}: predictions={batch_predictions}")
            all_predictions.append(batch_predictions)
            logger.debug(f"{test_iter}: ground_truth={batch_labels}")
            all_labels.append(batch_labels)
            #print(f"total  = {time.time()-start_time} \n\n")

    return all_predictions, all_labels, labels


def evaluate(
    teacher_models,
    student_model,
    test_set,
    config,
):
    """
    Standart Evaluation Metrics
    Args:
            teacher_model: GT generator on prediciton
            student_model: The model that is being evaluated
            test_set: A generator of data
            config: guides how the run was configurated

    Returns:
            AVG Score of the prediction
    """

    logger.warn(f"\nBeginning evaluation of student model performance")

    # Calculate AP table
    ap_table = []

    # ALL stands for the fact that we want to compare against all predictions
    # This is under the idea that all
    if type(teacher_models) == torch.nn.DataParallel:
        modalities_keys = teacher_models.module.keys()
    else:
        modalities_keys = teacher_models.keys()
    testing_points = list(modalities_keys)
    if config.getboolean('use_thermal') and config.getboolean('use_depth') and config.getboolean('use_rgb'):
        testing_points = ['ALL']

    for modality in testing_points:
        logger.debug(f"Working on modality={modality}")
        ap_modality = {
            'exp_name': config['exp_name'],
            'modality': modality,
            'AP@Ave': [0.],
            'AP@0.5': [0.],
            'AP@0.75': [0.],
            'CDx': [0.],
            'CDy': [0.]
        }

        # Get the prediction
        start_time = time.time()
        if modality != 'ALL':
            all_predictions, all_labels, labels = get_predictions(
                teacher_models[modality],
                student_model,
                test_set,
                config,
                modality
            )
        else:
            all_predictions, all_labels, labels = get_predictions_multiteacher(
                teacher_models,
                student_model,
                test_set,
                config,
            )

        # account for resources in the model quality
        elapsed_time = time.time() - start_time
        pytorch_total_params = sum(p.numel() for p in student_model.parameters())
        pytorch_total_params_trainable = sum(
            p.numel() for p in student_model.parameters() if p.requires_grad
        )
        resources = pd.DataFrame([{
            'model' : config['student'],
            'Time2Predict' : elapsed_time,
            'TotalParams' : pytorch_total_params,
            'TrainParams' :pytorch_total_params_trainable,
        }])
        logger.warn("\n" + tabulate(resources, headers='keys', tablefmt='psql'))
        if os.path.exists(f"{config['exp_name']}"):
            resources.to_csv(f"{config['exp_name']}/resources.{config['rank']}.csv", index=False)

        # Update the ap table per prediction
        ap_record = []
        for IOU in np.arange(0.5, 0.95, 0.05):

            # Make sure no extra decimal due to div
            IOU = np.around(IOU, decimals=2)

            sample_metrics = []  # List of tuples (TP, confs, pred)
            cd_x, cd_y = [], []
            for batch_predictions, batch_labels in tqdm(zip(all_predictions, all_labels), desc=f"Ap@{IOU}", total=len(all_labels)):
                logger.debug(f"batch_predictions={np.array(batch_predictions).shape}")
                logger.debug(f"batch_labels={np.array(batch_labels).shape}")
                sample_metrics += get_batch_statistics(
                    batch_predictions,
                    batch_labels,
                    IOU,
                )
                # Get the cd_x and cd_y distances
                cdx, cdy = get_batch_central_distances(
                    batch_predictions,
                    batch_labels,
                    config.getint('image_size'),
                    config.getint('image_size'),
                )
                cd_x.extend(cdx)
                cd_y.extend(cdy)

            # Concatenate sample statistics
            if not any(sample_metrics):
                logger.error(f"No valid prediction was made!!")
                precision = [0.]
                recall = [0.]
                AP = [0.]
                f1 = [0.]
                ap_class = [0]
                score = [0.]
                cd_x = [100.]
                cd_y = [100.]
            else:
                true_positives, pred_scores,  pred_labels = [
                    np.concatenate(x, 0) for x in list(zip(*sample_metrics))
                ]
                precision, recall, AP, f1, ap_class, score = ap_per_class(
                    true_positives,
                    pred_scores,
                    pred_labels,
                    labels
                )

            mean = 0.0
            if hasattr(AP, 'mean'):
                mean = AP.mean()
            if IOU == 0.5:
                ap_class_txt = np.frompyfunc(lambda s: test_set.classes[s], 1, 1)(ap_class)
                df = pd.DataFrame(
                    np.stack((ap_class_txt, precision, recall, AP, f1), axis=-1),
                    index=ap_class,
                    columns=['class', 'precision', 'recal', 'AP', 'F1']
                )
                logger.warn("AP per class for IOU=0.5")
                logger.warn("\n" + tabulate(df, headers='keys', tablefmt='psql'))

                #ap_table.at[0, 'AP@0.5'] = mean * 100
                ap_modality['AP@0.5'] = mean * 100

                ## CD should not be IOU dependent but add here to make it efficient
                #ap_table.at[0, 'CDx'] = np.mean(cd_x) * 100
                #ap_table.at[0, 'CDy'] = np.mean(cd_y) * 100
                # CD should not be IOU dependent but add here to make it efficient
                ap_modality['CDx'] = np.mean(cd_x) * 100
                ap_modality['CDy'] = np.mean(cd_y) * 100
            if IOU == 0.75:
                #ap_table.at[0, 'AP@0.75'] = mean * 100
                ap_modality['AP@0.75'] = mean * 100
            logger.debug(f"IOU={IOU} AP.mean()={mean}")
            ap_record.append(mean)

        #ap_table.at[0, 'AP@Ave'] = np.mean(ap_record) * 100
        ap_modality['AP@Ave'] = np.mean(ap_record) * 100
        ap_table.append(ap_modality)
    ap_table = pd.DataFrame(ap_table)
    logger.warn("\n" + tabulate(ap_table, headers='keys', tablefmt='psql'))
    if os.path.exists(f"{config['exp_name']}"):
        ap_table.to_csv(f"{config['exp_name']}/results.{config['rank']}.csv", index=False)
    return

def calc_gradient_penalty(netD, real_data, fake_data, lambda_gp=10.0):
    #print real_data.size()
    #alpha = torch.rand(BATCH_SIZE, 1)
    #alpha = alpha.expand(real_data.size())
    #alpha = alpha.cuda(gpu) if use_cuda else alpha
    alpha = torch.rand(real_data.size(0), 1, 1, 1).cuda().expand_as(real_data)

    #interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(alpha * real_data.data + (1 - alpha) * fake_data.data,                   requires_grad=True).cuda()

    disc_interpolates = netD(interpolates)

    grad = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    #gradient_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    grad = grad.view(grad.size(0), -1)
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

    # Backward + Optimize
    gradient_penalty = lambda_gp * d_loss_gp
    return gradient_penalty

def plot_audio_predictions(
        source,
        config,
        model=None,
        id=None,
):
    """
    Plots a nice image with predicitons for visual debug
    Args:
            source: From where to get the data to plot
            config: A config object to know how to run
            anchors: Yolo anchors fro prediction processing
            model: Optional model to predict

    Returns:
            None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Print options
    np.set_printoptions(suppress=True, precision=3)

    # Read input image
    rgb_size = config.getint('image_size')
    if id is None:
        id = np.random.randint(len(source))
    else:
        id = source.ids.index(id)

    # First Predict
    rgb, thermal, depth, audio, label, id_ = source[id]
    rgb_image, thermal_image, depth_image, audio_image, label_image, id = source.get_clean_data(id)
    #thermal_image[thermal_image > 140] = 140
    if thermal_image is not None:
        thermal_image[thermal_image < 90] = 90
        thermal_image =  cv2.normalize(thermal_image,  np.zeros_like(thermal_image), 0, 255, cv2.NORM_MINMAX)
    rgb_path, thermal_path, depth_path, audio_paths, rgb_xml_path = source.get_paths(id)
    id = slugify(id)

    # Scale the predictions to the original size
    # Keep originals
    orig_image = cv2.imread(rgb_path)
    print(f"orig_image={orig_image.shape}")
    height, width, channels = orig_image.shape
    common_size = rgb.shape[-1]
    if height > width:
        scale = common_size / height
        resized_height = common_size
        resized_width = int(width * scale)
    else:
        scale = common_size / width
        resized_height = int(height * scale)
        resized_width = common_size

    # Audio is always the input modality
    audio = audio[None, :, :, :]
    audio = Variable(torch.FloatTensor(audio)).to(device)
    print(f"Using audio={audio.shape}")
    with torch.no_grad():
        model.eval()
        logits_s, features_s = model(audio)
        for feature in features_s:
            feature_at = torch.nn.functional.normalize(
                feature.pow(2).mean(1).view(feature.size(0), -1)
            ).view(feature.shape[-1], feature.shape[-1]).cpu().numpy()
            plt.matshow(feature_at, cmap='viridis')
            plt.savefig(f"{id}_activation_{feature.shape[-1]}.jpg")
            plt.close()

    # Plot the image in case no prediction
    plt.imshow(rgb_image)
    plt.savefig(f"{id}_image.jpg")
    plt.close()

    # Predict the output of the model instead of GT
    prediction = logits_to_ground_truth(
        logits=logits_s,
        anchors=None,
        valid_classes_dict=source.valid_classes_dict,
        config=config,
        include_scores=True,
        crash_if_no_pred=False,
        text_classes=True,
    )[0]
    for pred in prediction:
        print(f"Predicted {pred}")
    images_to_plot = []
    if len(prediction) > 0:
        images_to_plot.append({
            'data': rgb_image,
            'bbox': prediction,
            'title': "Prediction_Resized",
            'type': 'image',
            'width_ratio': 1,
            'height_ratio': 1,
        })
        if thermal_image is not None:
            images_to_plot.append({
                'data': cv2.applyColorMap(thermal_image, cv2.COLORMAP_HOT),
                'bbox': prediction,
                'title': "thermal",
                'type': 'image',
                'width_ratio': 1,
                'height_ratio': 1,
            })
        if depth_image is not None:
            images_to_plot.append({
                'data': depth_image,
                'bbox': prediction,
                'title': "depth",
                'type': 'image',
                'width_ratio': 1,
                'height_ratio': 1,
            })
        prediction_scaled = []
        for pred in prediction:
            prediction_scaled.append([
                pred[0]/scale,
                pred[1]/scale,
                pred[2]/scale,
                pred[3]/scale,
                pred[4],
                pred[5],
            ])
        print(f"\nprediction_scaled  ({scale}) =")
        for pred in prediction_scaled:
            print(f"Predicted={pred}")
        images_to_plot.append({
            'data': orig_image,
            'bbox': prediction_scaled,
            'title': "Prediction",
            'type': 'image',
            'width_ratio': 1,
            'height_ratio': 1,
        })
        images_to_plot.append({
            'data': audio_image,
            'bbox': None,
            'title': "spectogram",
            'type': 'specshow',
            'width_ratio': 1,
            'height_ratio': 1,
        })

    # Plot the image
    colors = pickle.load(open("src/utils/pallete", "rb"))
    color = {
        'rgb': 'Blue',
        'thermal': 'Red',
        'audio': 'Green',
        'depth': 'Yellow'
    }
    for i, modality_image in enumerate(images_to_plot):

        # Add bbox into the image
        if modality_image['bbox'] is not None:
            for pred in modality_image['bbox']:
                # Change to str if required
                if not isinstance(pred[-1], str):
                    class_name = source.valid_classes_dict['labels_i2txt'][int(pred[-1])]
                else:
                    class_name = pred[-1]

                # If from prediction, scale to orig size
                # Labels from GT are 0,1 from pred are 2,3 if any

                #  Do not disturbe orig labels
                width_ratio  = modality_image['width_ratio']
                height_ratio = modality_image['height_ratio']

                xmin = int(max(pred[0] / width_ratio, 0))
                ymin = int(max(pred[1] / height_ratio, 0))
                xmax = int(min(pred[2] / width_ratio, width))
                ymax = int(min(pred[3] / height_ratio, height))

                cv2.rectangle(modality_image['data'], (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

        # Plot the modality
        if modality_image['type'] == 'image':
            cv2.imwrite(
                f"{id}_image_{ modality_image['title'] }.jpg",
                modality_image['data'].astype('float32'),
            )
        if modality_image['type'] == 'waveplot':
            librosa.display.waveplot(modality_image['data'], sr=44100, alpha=0.25)
            plt.savefig(f"{id}_waveplot_{ modality_image['title']  }.jpg")
            plt.close()
        if modality_image['type'] == 'specshow':
            for i in range(len(modality_image['data'])):
                librosa.display.specshow(
                    modality_image['data'][i],
                    x_axis='time',
                    y_axis='mel',
                    sr=44100,
                    fmax=8000
                )
                plt.savefig(f"{id}_specshow_{ modality_image['title']  }_{i}.jpg", dpi=1000, quality=95)
                plt.close()

    return


def prediction_frame_to_dict(frame, shape=6):
    """
    Converts a data frame to a csv for easy comparisson

    Args:
        frame (pd.DataFrame): The frame with predictions
        shape (int): The shape of the np array. Teacher doesn;t have score, so it should be 5
                     and student prediction should be shape 6 [xmi, ymin, xmax, ymax, score, label]
    """
    predictions = {}
    for index, row in tqdm(frame.iterrows(), desc='prediction_frame_to_dict', total=frame.shape[0]):
        bboxes = np.array(eval(
            #row['batch_labels'].to_numpy()[0].replace('\n', '').replace(' ', ', ')
            row['batch_labels'].replace('\n', '').replace(' ', ', ')
        ))
        if bboxes.size == 0:
            continue
        if len(bboxes.shape) == 1:
            bboxes = np.expand_dims(bboxes, axis=0)
        assert bboxes.shape[1] == shape, f"row={row} shape={bboxes.shape}"
        #predictions[row['id'].to_string()] = bboxes
        predictions[row['id']] = bboxes
    return predictions


def bboxes_to_area(bboxes):
    list_of_areas = []
    for i in range(bboxes.shape[0]):
        if len(bboxes[i]) == 6:
            xmin, ymin, xmax, ymax, score, clase = bboxes[i]
        else:
            xmin, ymin, xmax, ymax, clase = bboxes[i]

        area = (xmax-xmin) * (ymax-ymin)
        list_of_areas.append(area)
    return list_of_areas


def get_bbox_location(missing_bboxes):
    predominance = 'ALL'

    mapping = {
        'border_left': 0,
        'border_right': 0,
        'border': 0,
        'TL': 0,
        'TR': 0,
        'BL': 0,
        'BL': 0,
    }

    for prediction in missing_bboxes:
        if prediction[0] < 10:
            mapping['border_left'] += 1
            mapping['border'] += 1
        if prediction[2] > 750:
            mapping['border_right'] += 1
            mapping['border'] += 1
        if prediction[0] < 768//2 and prediction[1] < 768//2:
            mapping['BL'] += 1
        if prediction[0] > 768//2 and prediction[1] > 768//2:
            mapping['BR'] += 1
        if prediction[0] < 768//2 and prediction[1] > 768//2:
            mapping['TL'] += 1
        if prediction[0] > 768//2 and prediction[1] > 768//2:
            mapping['TR'] += 1

    if mapping['border'] >= max(mapping.values()):
        return 'border'

    return max(mapping.items(), key=operator.itemgetter(1))[0]


def collect_prediction_statistics(student_prediction_csv, teacher_prediction_csv):

    # Get a dict with the bboxes of each prediction
    student_frame = pd.read_csv(student_prediction_csv)
    student_predictions = prediction_frame_to_dict(student_frame, shape=6)
    teacher_frame = pd.read_csv(teacher_prediction_csv)
    teacher_predictions = prediction_frame_to_dict(teacher_frame, shape=5)

    # the idea is to create a pandas frame with statistics about why we
    # were not able to predict
    statistics = []
    total_excess_predictions = 0

    for teacher_id, teacher_bboxes in tqdm(teacher_predictions.items(), desc='Frame'):
        # we do not care about about teacher with no bboxes
        if teacher_bboxes.size == 0:
            continue

        drive, code = teacher_id.split('/')
        drive_type = DRIVES[drive]

        # If the student was not able to predict this, assume worst case scenario
        if teacher_id not in student_predictions or student_predictions[teacher_id].size == 0:
            print(f"ALL={teacher_id}")
            statistics.append({
                'id': teacher_id,
                'expected_bboxes': teacher_bboxes.shape[0],
                'predicted_bboxes': 0,
                'missing_bboxes': teacher_bboxes.shape[0],
                'excess_bboxes': 0,
                'smallest_bbox_missed': np.min(bboxes_to_area(teacher_bboxes)),
                'biggest_bbox_missed': np.max(bboxes_to_area(teacher_bboxes)),
                'avg_bbox_missed': np.mean(bboxes_to_area(teacher_bboxes)),
                'is_day': 'day' in drive_type,
                'is_night': 'night' in drive_type,
                'is_static': 'static' in drive_type,
                'is_driving': 'driving' in drive_type,
                'predominating_area_missing': 'ALL',
            })
            continue
        student_bboxes = student_predictions[teacher_id]
        true_positives, detected, pred_scores,  pred_labels = get_batch_statistics(
            #student_bboxes,
            np.expand_dims(student_bboxes, axis=0),
            #teacher_bboxes,
            np.expand_dims(teacher_bboxes, axis=0),
            iou_threshold=0.5,
            add_detected=True
        )[0]
        missing_bboxes = teacher_bboxes[detected == 0]
        excess_predictions = min(0, (teacher_bboxes.shape[0] - student_bboxes.shape[0]))
        # TODO: are excess prediction
        if excess_predictions > 1:
            total_excess_predictions += 1
        if missing_bboxes.size == 0:
            continue

        statistics.append({
            'id': teacher_id,
            'expected_bboxes': teacher_bboxes.shape[0],
            'predicted_bboxes': student_bboxes.shape[0],
            'missing_bboxes': missing_bboxes.shape[0],
            'excess_bboxes': excess_predictions,
            'smallest_bbox_missing': min(bboxes_to_area(missing_bboxes)),
            'biggest_bbox_missing': max(bboxes_to_area(missing_bboxes)),
            'avg_bbox_missed': np.mean(bboxes_to_area(missing_bboxes)),
            'is_day': 'day' in drive_type,
            'is_night': 'night' in drive_type,
            'is_static': 'static' in drive_type,
            'is_driving': 'driving' in drive_type,
            'predominating_area_missing': get_bbox_location(missing_bboxes),
        })

    print(f"total_excess_predictions={total_excess_predictions}")
    return pd.DataFrame(statistics)



def overlay_mask(img, mask, colormap='jet', alpha=0.7):
    """Overlay a colormapped mask on a background image
    Args:
        img (PIL.Image.Image): background image
        mask (PIL.Image.Image): mask to be overlayed in grayscale
        colormap (str, optional): colormap to be applied on the mask
        alpha (float, optional): transparency of the background image
    Returns:
        PIL.Image.Image: overlayed image
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError('img and mask arguments need to be PIL.Image')

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError('alpha argument is expected to be of type float between 0 and 1')

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, 1:]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


