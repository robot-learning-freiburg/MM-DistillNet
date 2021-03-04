# -*- coding: utf-8 -*-
"""Master Project -- Multi Modal Object Detection

FreiburgSmallPassingCars RGB/thermal/Depth Dataset

"""

# -------------------------------------------------------------------------------
#                                   Imports
# -------------------------------------------------------------------------------
# General Inputs
import logging
import os
import pickle
import re
import glob

# Third Party
import cv2

import numpy as np

import pandas as pd

import librosa

# Local Imports
from .BaseDataset import BaseDataset
from src.utils.utils import (
    readPmf,
    applyLogJetColorMap
)

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

# -------------------------------------------------------------------------------
#                           Logger Configuration
# -------------------------------------------------------------------------------
# Logging
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------
#                                 Methods
# -------------------------------------------------------------------------------


class CityScapesDataset(BaseDataset):
    """
    Dataset to read multimodal detection data
    """
    def __init__(
        self,
        config,
        mode,
    ):
        """
        Creates a Dataset to yield multiple modalities and Labels
        if they are available
        Args:
                config: A configuration file
                mode: Can be train, val or test

        Returns:
                A yielder object of images that ingerits from dataset
        """


        self.classes = [
            'aeroplane',
            'bicycle',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'diningtable',
            'dog',
            'horse',
            'motorbike',
            'person',
            'pottedplant',
            'sheep',
            'sofa',
            'train',
            'tvmonitor'
        ]

        super().__init__(config=config, mode=mode,classes=self.classes)

        #self.crop_left = 150
        #self.crop_right = 850

        # Empirically found with noise filtering histogram
        self.depth_max = 192

        logger.info(f"MultimodalDetection {mode} with {self.num_images} images")

    def get_id_list(self):

        self.ids = []

        for path in glob.glob(os.path.join(
            self.data_path,
            "annotations",
            self.mode,
            '*',
            '*',
        )):
            dirname = os.path.basename(os.path.dirname(path))
            id_name, ext = os.path.splitext(os.path.basename(path))
            id_name = id_name.replace('_annotations', '')
            self.ids.append(f"{dirname}/{id_name}")
        self.num_images = len(self.ids)
        return self.ids

    def get_paths(self, id):
        """
        Returns the path of the modalities to load given an ID
        """
        city, name = id.split('/')

        rgb_path = os.path.join(
            self.data_path,
            'leftImg8bit',
            self.mode,
            city,
            f"{name}_leftImg8bit.png"
        )
        thermal_path = None
        depth_path = os.path.join(
            self.data_path,
            'disparity',
            self.mode,
            city,
            f"{name}_disparity.png"
        )
        audio_paths = None
        annotation_path = os.path.join(
            self.data_path,
            'annotations',
            self.mode,
            city,
            f"{name}_annotations.txt"
        )

        return rgb_path, thermal_path, depth_path, audio_paths, annotation_path

    def get_annotations(self, id):
        rgb_path, thermal_path, depth_path,  audio_paths, label_path = self.get_paths(id)
        label = np.loadtxt(label_path)
        if label.ndim < 2:
            label = label.reshape(1,5)
        return label

    def filter_labels(self, labels):
        return labels

    def __getitem__(self, item):

        id = self.ids[item]

        # Obtain the images to process
        rgb_path, thermal_path, depth_path,  audio_paths, label_path = self.get_paths(id)
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        thermal = None

        depth = None
        if self.use_depth:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth[depth > 0] = (depth[depth > 0] - 1) / 256
            depth[depth > self.depth_max] = self.depth_max
            # depth = applyLogJetColorMap(depth)

        audio = None

        # Normalize to 1
        if self.normalize:
            rgb = rgb.astype(np.float32)/255.

        label = self.get_annotations(id)


        if self.transformations is not None:
            rgb, thermal, depth, audio, label, id = self.transformations((
                rgb, thermal, depth, audio, label, id
            ))

        rgb = np.transpose(np.array(rgb, dtype=np.float32), (2, 0, 1))
        if self.use_depth:
            depth = depth[:, :, None]
            depth = np.transpose(np.array(depth, dtype=np.float32), (2, 0, 1))

        label = np.array(label, dtype=np.float32)

        return rgb, thermal, depth, audio, label, id
