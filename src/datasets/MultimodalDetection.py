# -*- coding: utf-8 -*-
"""
There is More than Meets the Eye: Self-Supervised Multi-Object Detection
and Tracking with Sound by Distilling Multimodal Knowledge

Code to read the associated dataset
"""

# -------------------------------------------------------------------------------
#                                   Imports
# -------------------------------------------------------------------------------
# General Inputs
import logging
import os
import pickle
import re

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


class MultimodalDetection(BaseDataset):
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

        self.crop_left = 200
        self.crop_right = 1720
        self.ir_minval = 21800
        self.ir_maxval = 25000

        # Empirically found with noise filtering histogram
        self.depth_max = 153

        logger.info(f"MultimodalDetection {mode} with {self.num_images} images")

    def get_id_list(self):
        id_list_path = os.path.join(self.data_path, "{}{}{}.txt".format(
            self.mode,
            self.daytime,
            self.drive_type,
        ))
        logger.debug(f"Using ids from = {id_list_path}")
        self.ids = [id.strip() for id in open(id_list_path)]
        # Further redefine based on config request
        if 'None' not in self.config['id_filter']:
            r = re.compile(self.config['id_filter'])
            valid_ids = list(filter(r.match, self.ids))
            new_ids = list(set(self.ids) & set(valid_ids))
            self.ids = new_ids

        self.ids.sort()
        times = []
        new_ids = []
        burst = []
        for id in self.ids:
            drive, rgb_timestamp = id.split('/')
            secs, nsec, code = rgb_timestamp.split('_')
            if len(nsec) < 9:
                # Skip bogus times stamps
                continue
            times.append(int(str(secs) + str(nsec)))
            new_ids.append(id)
            burst.append(id)
        times = pd.to_datetime(times)
        df = pd.DataFrame(new_ids, columns=['ids'])
        df['time'] = times
        df['burst'] = burst
        self.ids = df.sort_values(by=['time', 'burst'])['ids'].to_list()

        self.num_images = len(self.ids)
        self.ids2intday = [i for i in range(len(self.ids)) if 'day' in self.ids[i]]
        return self.ids

    def get_paths(self, id, traditional_nms_kdlist_augmented=False):
        """
        Returns the path of the modalities to load given an ID
        """
        drive, rgb_timestamp = id.split('/')

        rgb_path = os.path.join(
            self.data_path,
            drive,
            "fl_rgb",
            f"fl_rgb_{rgb_timestamp}.png"
        )
        thermal_path = os.path.join(
            self.data_path,
            drive,
            'fl_ir_aligned',
            f"fl_ir_aligned_{rgb_timestamp}.png"
        )
        depth_path = os.path.join(
            self.data_path,
            drive,
            'fl_rgb_depth',
            # Due to size restrictions, we will allow a PNG
            # rather than the PFM. This affects a bit performance
            # but PFM space is very very big
            # f"fl_rgb_{rgb_timestamp}.pfm"
            f"fl_rgb_{rgb_timestamp}.png"
        )
        ext = 'pkl'
        if traditional_nms_kdlist_augmented:
            ext = 'wav'
        audio_paths = [os.path.join(
            self.data_path,
            drive,
            "audio",
            f"audio_{i}_{rgb_timestamp}.{ext}"
        ) for i in range(8)]

        return rgb_path, thermal_path, depth_path, audio_paths, None

    def __getitem__(self, item):

        id = self.ids[item]

        # Obtain the images to process
        rgb_path, thermal_path, depth_path,  audio_paths, label_path = self.get_paths(id)
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            print(f"rgb={rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[:, self.crop_left:self.crop_right, :]

        thermal = None
        if self.use_thermal:
            thermal = cv2.imread(thermal_path, cv2.IMREAD_ANYDEPTH)
            if thermal is None:
                print(f"thermal={thermal_path}")
            thermal = thermal[:, self.crop_left:self.crop_right]

            # normalize IR data
            # (is in range 0, 2**16 --> crop to relevant range(20800, 27000))
            thermal[thermal < self.ir_minval] = self.ir_minval
            thermal[thermal > self.ir_maxval] = self.ir_maxval

            thermal = cv2.normalize(
                thermal,
                np.zeros(thermal.shape),
                0,
                255,
                cv2.NORM_MINMAX
            )
            thermal = thermal.astype(np.float32)

        depth = None
        if self.use_depth:
            # Due to disk constraints, we only enable PNG
            # Not the actual depth file
            #depth = readPmf(depth_path)
            #depth = applyLogJetColorMap(depth)
            #depth = depth[:, self.crop_left:self.crop_right, :]
            depth = cv2.imread(depth_path)

        audios = [
            pickle.load(
                open(audio_path, 'rb'), encoding='latin1'
            ) for audio_path in audio_paths
        ]
        audio = np.transpose(np.stack(audios), (1, 2, 0))

        # Normalize to 1
        if self.normalize:
            rgb = rgb.astype(np.float32)/255.
            if self.use_thermal:
                thermal = thermal.astype(np.float32)/255.
            if self.use_depth:
                depth = depth.astype(np.float32)/255.

        label = None

        if self.config['data_augment_shift'] == 'True' and np.random.uniform() > 0.5:
            if thermal is not None:
                rgb = self.shift(rgb)
            if thermal is not None:
                thermal = self.shift(thermal)
            if depth is not None:
                depth = self.shift(depth)

        if self.transformations is not None:
            rgb, thermal, depth, audio, label, id = self.transformations((
                rgb, thermal, depth, audio, label, id
            ))

        rgb = np.transpose(np.array(rgb, dtype=np.float32), (2, 0, 1))
        audio = np.transpose(audio, (2, 0, 1))
        if self.use_thermal:
            thermal = thermal[:, :, None]
            thermal = np.transpose(np.array(thermal, dtype=np.float32), (2, 0, 1))
        if self.use_depth:
            depth = np.transpose(np.array(depth, dtype=np.float32), (2, 0, 1))

        return rgb, thermal, depth, audio, label, id

    def get_annotations(self, id):
        objects = []

        rgb_path, thermal_path, depth_path, audio_path, annotations_path = self.get_paths(id)

        if not self.use_labels or not os.path.exists(annotations_path):
            return objects

        objects = np.loadtxt(annotations_path, dtype=np.float32)
        objects = self.filter_labels(objects)
        return objects

    def get_clean_data(self, item):

        id = self.ids[item]

        # Obtain the images to process
        rgb_path, thermal_path, depth_path,  audio_paths, label_path = self.get_paths(id)
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise Exception(f"rgb={rgb_path}")
        rgb = rgb[:, self.crop_left:self.crop_right, :]

        thermal = None
        if self.use_thermal:
            thermal = cv2.imread(thermal_path, cv2.IMREAD_ANYDEPTH)
            thermal = thermal[:, self.crop_left:self.crop_right]

            thermal[thermal < self.ir_minval] = self.ir_minval
            thermal[thermal > self.ir_maxval] = self.ir_maxval

            thermal = cv2.normalize(
                thermal,
                np.zeros(thermal.shape),
                0,
                255,
                cv2.NORM_MINMAX
            )

        depth = None
        if self.use_depth:
            depth = readPmf(depth_path)
            depth = applyLogJetColorMap(depth)
            depth = depth[:, self.crop_left:self.crop_right, :]

        audios = [
            pickle.load(
                open(audio_path, 'rb'), encoding='latin1'
            ) for audio_path in audio_paths
        ]

        rgb, thermal, depth, audio, label, id = self.resizer((
            rgb, thermal, depth, None, None, id
        ))

        if rgb is not None:
            rgb = rgb.astype(np.uint8)
        if thermal is not None:
            thermal = thermal.astype(np.uint8)
        return rgb, thermal, depth, audios, None, id

    def shift(self, img, shift=2):
        img_new = np.zeros_like(img)
        if len(img.shape) == 2:
            # BW image
            img_new[:,0:img.shape[1]-shift] = img[:,shift:]
        else:
            img_new[:,0:img.shape[1]-shift, :] = img[:,shift:, :]
        return img_new

    def merge_audios(self, id1, id2):
        common_size = 768
        _, _, _, audio_paths1, _ = self.get_paths(id1, traditional_nms_kdlist_augmented=True)
        _, _, _, audio_paths2, _ = self.get_paths(id2, traditional_nms_kdlist_augmented=True)
        spectogram = []
        for i in range(len(audio_paths1)):
            audio1, sr = librosa.load(audio_paths1[i], sr=44100)
            audio2, sr = librosa.load(audio_paths2[i], sr=44100)
            spectogram.append(
                librosa.feature.melspectrogram(
                    y=(audio1 + audio2) / 2,
                    sr=44100,
                    n_fft=1024,
                    hop_length=256,
                    n_mels=80,
                )
            )
        spectogram = np.transpose(np.stack(spectogram), (1, 2, 0))
        spectogram = cv2.resize(
            spectogram,
            dsize=(common_size, common_size),
            interpolation=cv2.INTER_CUBIC
        )
        spectogram = np.transpose(spectogram, (2, 0, 1))
        return spectogram

    def yield_batch(self, batch_size, ids):
        batch_audios = []
        batch_rgb = []
        # Don't use the same ids in the batch
        this_ids = [self.ids.index(a) for a in ids]
        proposed_ids = np.random.choice(
                [a for a in self.ids2intday if a not in this_ids],
                size=batch_size)
        for i in range(batch_size):
            rgb, thermal, depth, naudio, label, ids2 = self.__getitem__(proposed_ids[i])
            batch_audios.append(self.merge_audios(ids[i], ids2))
            batch_rgb.append(rgb)
        return torch.from_numpy(np.stack(batch_rgb)), torch.from_numpy(np.stack(batch_audios))
