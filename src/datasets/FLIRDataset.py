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


class FLIRDataset(BaseDataset):
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
        self.id2label = self.gen_id2label_file()

    def get_id_list(self):
        # For the label, we create an easier format for our purposes
        id_list_path = glob.glob(f"{self.data_path}/{self.mode}/RGB/*.jpg")
        if len(id_list_path) < 1:
            raise Exception(f"No data on {self.data_path}!")

        self.ids = [os.path.splitext(
            os.path.basename(id)
        )[0] for id in id_list_path]

        return self.ids

    def get_paths(self, id):
        """
        Returns the path of the modalities to load given an ID
        """
        rgb_path = os.path.join(self.data_path, self.mode, f"RGB/{id}.jpg")
        thermal_path = os.path.join(self.data_path, self.mode, f"thermal_8_bit/{id}.jpeg")
        return rgb_path, thermal_path, None, None, f"{self.data_path}/{self.mode}/labels.json"

    def __getitem__(self, item):
        id = self.ids[item]

        rgb_path, thermal_path, _, _, label_path = self.get_paths(id)

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        thermal = None
        if self.use_thermal:
            thermal = cv2.imread(thermal_path, cv2.IMREAD_ANYDEPTH)

            thermal = cv2.normalize(
                thermal,
                np.zeros(thermal.shape),
                0,
                255,
                cv2.NORM_MINMAX
            )
            thermal = thermal.astype(np.float32)

        # Fix flir rgb to thermal size
        height, width = thermal.shape
        rgb = cv2.resize(rgb, (width, height))

        # Normalize to 1
        if self.normalize:
            rgb = rgb.astype(np.float32)/255.
            if self.use_thermal:
                thermal = thermal.astype(np.float32)/255.

        depth = None
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
        if self.use_thermal:
            thermal = thermal[:, :, None]
            thermal = np.transpose(np.array(thermal, dtype=np.float32), (2, 0, 1))
        if self.use_depth:
            # depth = depth[:, :, None]
            depth = np.transpose(np.array(depth, dtype=np.float32), (2, 0, 1))
        if self.use_audio:
            audio = np.transpose(audio, (2, 0, 1))

        return rgb, thermal, depth, audio, label, id


    def get_annotations(self, id):
        if id not in self.id2label: return []
        label = self.id2label[id]
        return np.array(label, dtype=np.float32)

    def gen_id2label_file(self):

        label_file_name = f"{self.data_path}/{self.mode}/labels.json"

        if not os.path.exists(label_file_name):
            # Create the file from scratch
            with open(f"{self.data_path}/{self.mode}/thermal_annotations.json") as f:
                ann = json.load(f)

            # Match id to filename
            thermal_id2filename = {}
            for filelist in ann["images"]:
                file_id = os.path.splitext(
                    os.path.basename(filelist['file_name'])
                )[0]
                thermal_id2filename[filelist['id']] = file_id

            # Match the thermal category to ours
            thermal_id2cat = {}
            for cat_list in ann['categories']:
                if cat_list['name'] in self.classes:
                    thermal_id2cat[cat_list['id']] = self.classes.index(
                        cat_list['name']
                    )

            # append labels as found
            id2label = {}
            for annotation in ann['annotations']:
                if annotation['category_id'] in thermal_id2cat:
                    x, y, w, h = annotation['bbox']
                    image_id = annotation['image_id']
                    if thermal_id2filename[image_id] not in id2label:
                        id2label[thermal_id2filename[image_id]] = []
                    id2label[thermal_id2filename[image_id]].append([
                        x,
                        y,
                        x+w,
                        y+h,
                        thermal_id2cat[annotation['category_id']]
                    ])
            with open(label_file_name, "w") as write_file:
                json.dump(id2label, write_file, indent=4, sort_keys=True)
            logger.info(f"{len(id2label.keys())} LabelF={label_file_name}!")

        with open(label_file_name) as f:
            return json.load(f)
