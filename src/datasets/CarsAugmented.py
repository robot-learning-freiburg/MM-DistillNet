# -*- coding: utf-8 -*-
"""Master Project -- Multi Modal Object Detection

CarsAugmented RGB Dataset

"""

# -------------------------------------------------------------------------------
#                                   Imports
# -------------------------------------------------------------------------------
# General Inputs
import logging
import os
import pickle
import xml.etree.ElementTree as ET

# Third Party
import cv2

import numpy as np

import pandas as pd

# Local Imports
from src.utils.utils import (
    EfficientDet_post_processing,
    yolo_post_processing,
    extract_transformations,
)

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

from tqdm import tqdm

from .BaseDataset import BaseDataset

# -------------------------------------------------------------------------------
#                           Logger Configuration
# -------------------------------------------------------------------------------
# Logging
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------
#                                 Methods
# -------------------------------------------------------------------------------


class CarsAugmented(BaseDataset):
    """
    Dataset to read VOC like datasets
    """
    def __init__(
        self,
        config,
        mode="train",
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

        self.mode = mode
        self.config = config

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
        super().__init__(config=config, mode=mode, classes=self.classes)

        logger.info(f"{mode}CarsAugmented with {self.num_images} images!")

    def get_id_list(self):
        id_list_path = os.path.join(
            self.data_path,
            "{}.txt".format(self.mode)
        )
        self.ids = [id.strip() for id in open(id_list_path)]
        self.num_images = len(self.ids)
        return self.ids

    def __len__(self):
        """Returns the number of items in the dataset"""
        return self.num_images

    def get_paths(self, id):
        directory, name = id.split(';')
        """Returns the location of the modalities and annotations"""
        ext = '.png' if 'KITTI' in directory else '.jpg'
        rgb_path = os.path.join(
            self.data_path,
            directory,
            'RGB',
            "{}{}".format(name, ext)
        )
        xml_path = os.path.join(
            self.data_path,
            directory,
            "Annotations",
            "{}.xml".format(name)
        )

        return rgb_path, None, None, None, xml_path

    def get_annotations(self, id):
        objects = []
        if not self.use_labels:
            return objects

        rgb_path, thermal_path, depth_path, audio_path, xml_path = self.get_paths(id)
        annot = ET.parse(xml_path)

        for obj in annot.findall('object'):
            xmin, xmax, ymin, ymax = [
                int(float(obj.find('bndbox').find(tag).text)) - 1 for tag in
                ["xmin", "xmax", "ymin", "ymax"]
            ]
            if obj.find('name').text.lower().strip() not in self.classes:
                continue
            label = self.classes.index(
                obj.find('name').text.lower().strip()
            )
            objects.append([xmin, ymin, xmax, ymax, label])

        objects = np.array(objects, dtype=np.float32)
        objects = self.filter_labels(objects)
        return objects

    def __getitem__(self, item):
        """Get a single item from the dataset"""
        id = self.ids[item]
        rgb_path, thermal_path, depth_path, audio_path, xml_path = self.get_paths(id)

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        self.height, self.width = rgb.shape[:2]

        thermal = None
        if self.use_thermal:
            thermal = rgb.copy()

        depth = None
        if self.use_depth:
            depth = cv2.imread(depth_path)
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)

        audio = None

        # Normalize to 1
        if self.normalize:
            rgb = rgb.astype(np.float32)/255.
            if self.use_thermal:
                thermal = thermal.astype(np.float32)/255.
            if self.use_depth:
                depth = depth.astype(np.float32)/255.

        label = self.get_annotations(id)

        if self.transformations is not None:
            rgb, thermal, depth, audio,  label, id = self.transformations((
                rgb, thermal, depth, audio, label, id
            ))

        if self.use_labels:
            label = np.array(label, dtype=np.float32)
        rgb = np.transpose(np.array(rgb, dtype=np.float32), (2, 0, 1))
        if self.use_thermal:
            thermal = np.transpose(
                np.array(thermal, dtype=np.float32), (2, 0, 1)
            )
        if self.use_depth:
            depth = np.transpose(
                np.array(depth, dtype=np.float32), (2, 0, 1)
            )

        return rgb, thermal, depth, None, label, id

