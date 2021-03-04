# -*- coding: utf-8 -*-
"""Master Project -- Multi Modal Object Detection

FLIR RGB and thermal Dataset

"""

# -------------------------------------------------------------------
#                                   Imports
# -------------------------------------------------------------------
# General Inputs
import glob
import json
import logging
import os
import pickle
import tarfile

# Third Party
import cv2

import numpy as np

import pandas as pd

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


class ArgoverseDataset(BaseDataset):
    """
    Custom dataset to read FLIR thermal and RGB images
    """
    def __init__(
        self,
        config,
        mode,
    ):

        # Check if there is data to load, else die
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

    def get_id_list(self):
        # For the label, we create an easier format for our purposes
        id_list_path = glob.glob(f"{self.data_path}/{self.mode}/*/stereo_front_left/*.resized.jpg")
        if len(id_list_path) < 1:
            raise Exception(f"No data on {self.data_path}!")
        self.ids = []
        for id_element in id_list_path:
            log_name = os.path.basename(os.path.dirname(os.path.dirname(id_element)))
            timestamp = os.path.basename(id_element).replace('stereo_front_left_','').replace('.resized.jpg','')
            self.ids.append(f"{log_name}/{timestamp}")

        return self.ids

    def get_paths(self, id):
        """
        Returns the path of the modalities to load given an ID
        """
        log_name, timestamp = id.split('/')
        rgb_path = os.path.join(
            self.data_path,
            self.mode,
            log_name,
            'stereo_front_left',
            f"stereo_front_left_{timestamp}.resized.jpg"
        )
        depth_path = os.path.join(
            self.data_path,
            self.mode,
            log_name,
            'stereo_depth',
            f"stereo_depth_{timestamp}.pfm"
        )
        label_path = os.path.join(
            self.data_path,
            self.mode,
            log_name,
            'annotations',
            f"stereo_front_left_{timestamp}.txt"
        )
        return rgb_path, None, depth_path, None, label_path

    def __getitem__(self, item):
        id = self.ids[item]

        rgb_path, _, depth_path, _, label_path = self.get_paths(id)

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        thermal = None

        depth = readPmf(depth_path)

        depth = applyLogJetColorMap(depth)

        # Normalize to 1
        if self.normalize:
            rgb = rgb.astype(np.float32)/255.
            depth = depth.astype(np.float32)/255.

        audio = None

        label = None
        if self.use_labels:
            label = self.get_annotations(id)

        if self.transformations is not None:
            rgb, thermal, depth, audio, label, id = self.transformations((
                rgb, thermal, depth, audio, label, id
            ))

        if self.use_labels:
            label = np.array(label, dtype=np.float32)
        rgb = np.transpose(np.array(rgb, dtype=np.float32), (2, 0, 1))
        #depth = depth[:, :, None]
        depth = np.transpose(np.array(depth, dtype=np.float32), (2, 0, 1))

        return rgb, thermal, depth, audio, label, id

    def get_annotations(self, id):
        rgb_path, _, depth_path, _, label_path = self.get_paths(id)
        if not os.path.exists(label_path) or not os.path.exists(depth_path):
            return []
        label = np.loadtxt(label_path, delimiter=',')
        if len(label.shape) < 2:
            label = label.reshape(1, 5)

        # Rescale the labels
        scale_width = 1232 / 2464
        label[:,0] *= scale_width
        label[:,2] *= scale_width
        scale_height = 1028 / 2056
        label[:,1] *= scale_height
        label[:,3] *= scale_height
        return np.array(label, dtype=np.float32)
