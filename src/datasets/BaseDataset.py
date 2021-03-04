# -*- coding: utf-8 -*-
"""Master Project -- Multi Modal Object Detection
A base dataset from which to load config
"""

# -------------------------------------------------------------------------------
#                                   Imports
# -------------------------------------------------------------------------------
# General Inputs
import logging
import os
import pickle
import re
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
    logits_to_ground_truth,
)

import torch
from src.datasets.transformations import Compose, Resizer
from torch.autograd import Variable
from torch.utils.data import Dataset

from tqdm import tqdm


# -------------------------------------------------------------------------------
#                           Logger Configuration
# -------------------------------------------------------------------------------
# Logging
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------
#                                 Methods
# -------------------------------------------------------------------------------


class BaseDataset(Dataset):
    """
    Dataset to read VOC like datasets
    """
    def __init__(
        self,
        config,
        classes,
        mode="train",
    ):
        """
        Creates a Dataset to yield multiple modalities and Labels
        if they are available
        Args:
                mode: Can be train, val or test
                config: A configuration file
                classes: what classes are supported

        Returns:
                A yielder object of images that ingerits from dataset
        """

        self.mode = mode
        if'drive_type' in config:
            self.drive_type = '_' + config['drive_type']
        else:
            self.drive_type = ''
        if'daytime' in config:
            self.daytime = '_' + config['daytime']
        else:
            self.daytime = ''

        self.is_training = True if mode == "train" else False
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.classes = classes
        self.num_classes = len(self.classes)

        # Extract the configuration from the config
        self.normalize = config.getboolean('normalize')
        self.transformations = Compose(
            extract_transformations(
                     config,
                     self.mode,
                     rgb_size=config.getint('image_size'),
                     thermal_size=config.getint('thermal_size'),
                     depth_size=config.getint('depth_size'),
                     audio_size=config.getint('audio_size'),

            )
        )
        self.resizer = Compose([Resizer(common_size=config.getint('image_size'))])
        self.data_path = config['data_path']

        if not os.path.exists(self.data_path):
            raise Exception(f"Cannot read input path {self.data_path}")

        teacher = config['teacher']
        self.predictions_file = f"{self.data_path}/{teacher}_{self.mode}{self.drive_type}_predictions.csv"

        self.ids = self.get_id_list()
        self.num_images = len(self.ids)

        # handle Modalities
        self.rgb_size = config.getint('image_size')
        self.thermal_size = config.getint('thermal_size')
        self.depth_size = config.getint('depth_size')
        self.audio_size = config.getint('audio_size')

        self.use_thermal = config.getboolean('use_thermal')
        self.use_depth = config.getboolean('use_depth')
        #self.use_audio = config.getboolean('use_audio')
        self.use_audio = True

        # Handle how labels will be used
        self.use_labels = config.getboolean('use_labels')
        self.valid_labels = list(range(len(self.classes)))

        # Allow handlign of specific labels
        # Also allow conversion between labels and predictions
        # so we cna translate from yolo to coco
        self.valid_classes_dict = {'labels_i2txt': {},'labels_txt2i': {}, 'predictions_txt2i':{}, 'predictions_i2txt':{},}
        for i, the_class in enumerate(self.classes):
            # Here we handle what labels everything uses
            if 'valid_labels' in config and the_class not in config['valid_labels'].split(','):
                continue
            self.valid_classes_dict['labels_txt2i'][the_class] = i
            self.valid_classes_dict['labels_i2txt'][i] = the_class
            self.valid_classes_dict['predictions_txt2i'][the_class] = self.get_prediction_id(the_class)
            self.valid_classes_dict['predictions_i2txt'][self.get_prediction_id(the_class)]  = the_class

        logger.debug(f"self.valid_classes_dict={self.valid_classes_dict}")

        logger.info(f"{mode}CarsAugmented with {self.num_images} images!")

    def get_prediction_id(self, the_class):
        VOC = {
            'aeroplane': 0,
            'bicycle': 1,
            'bird': 2,
            'boat': 3,
            'bottle': 4,
            'bus': 5,
            'car': 6,
            'cat': 7,
            'chair': 8,
            'cow': 9,
            'diningtable': 10,
            'dog': 11,
            'horse': 12,
            'motorbike': 13,
            'person': 14,
            'pottedplant': 15,
            'sheep': 16,
            'sofa': 17,
            'train': 18,
            'tvmonitor': 19,
        }
        COCO = {'car': 2}
        return VOC[the_class]

    def get_id_list(self):
        raise NotImplemented

    def __len__(self):
        """Returns the number of items in the dataset"""
        return self.num_images

    def get_paths(self, id):
        raise NotImplemented

    def __getitem__(self, item):
        raise NotImplemented

    def get_annotations(self,id):
        raise NotImplemented

    def filter_labels(self, labels):
        mask = np.isin(labels[:,4], list(self.valid_classes_dict['labels_txt2i'].values()))
        labels = labels[mask]
        return labels

    def refine_ids(self, model, config):
        """
        This utility provides a list of the images that yolo
        is able to predict something.
        We should ignore images without prediction
        This is important on batches inputs, as ignoring an image is tricky
        Args:
                model: Model that would asses whether image is easy to predict
                config: a parsed config file

        Returns:
                The reduced ids that are easy to predict. This way we can be
                certain that the teacher would be able to predict
        """

        teacher = config['teacher']

        # Get the id list
        self.get_id_list()

        # If using labels, we only want to let the predictions with meaningful
        # labels be exercised due to runtime limitations
        if self.use_labels:
            valid_ids = []
            for i, id in enumerate(self.ids):
                labels = self.get_annotations(id)
                if len(labels) < 1:
                    continue
                valid_labels = self.filter_labels(labels)
                if len(valid_labels) > 1:
                    valid_ids.append(id)
            new_ids = list(set(self.ids) & set(valid_ids))
            logger.debug(f"Reduced {len(self.ids)}->{len(new_ids)}")
            self.ids = new_ids
            #self.ids = new_ids[:len(self.ids)//4]
            self.num_images = len(self.ids)
            return

        # If not using labels, then reduce the ids based on the teacher
        if not os.path.exists(self.predictions_file):
            logger.warn(f"Building file {self.predictions_file}")
            anchors = None
            if 'yolov2' in teacher:
                if type(model) == torch.nn.DataParallel:
                    anchors = model.module.anchors
                else:
                    anchors = model.anchors
            valid_images = []
            for i, id in enumerate(tqdm(self.ids, desc=f"Predict File")):
                rgb, thermal, depth, audio, label, id = self.__getitem__(i)

                with torch.no_grad():
                    rgb = Variable(
                        torch.FloatTensor(depth).unsqueeze(dim=0),
                        requires_grad=False
                    ).to(self.device)
                    logits, features = model(rgb)

                    batch_predictions = logits_to_ground_truth(
                        logits=logits,
                        valid_classes_dict=self.valid_classes_dict,
                        anchors=anchors,
                        config=config,
                        include_scores=True,
                        text_classes=False,
                        crash_if_no_pred=False,
                    )
                    if not np.any(batch_predictions):
                        num_predictions = 0
                    else:
                        num_predictions = len(batch_predictions[0])
                    min_confidence = 0
                    for i, predictions in enumerate(batch_predictions):
                        for pred in predictions:
                            score = pred[4]
                            if score > min_confidence:
                                min_confidence = score

                    valid_images.append([id, num_predictions, min_confidence])

            # Save to a text file for debug
            np.savetxt(
                self.predictions_file,
                np.asarray(valid_images),
                delimiter=",", fmt='%s'
            )

        # Just keep ids where teacher predicted something
        dataframe = pd.read_csv(
            self.predictions_file,
            names=['ID', 'Num_pred', 'min_confidence'],
            dtype={
                'ID': str, 'Num_pred': np.int32,
                'min_confidence': np.float32
            },
        )

        # How good a prediction from teacher has to be, to
        # accept this image as valid
        if 'yolov2' in config['teacher']:
            minconf = 0.5
        elif 'EfficientDet' in config['teacher']:
            minconf = 0.40
        else:
            raise Exception("Unsupported student")

        valid_ids = dataframe[dataframe['min_confidence'] > minconf]['ID'].tolist()

        # Further redefine based on config request
        if 'None' not in config['id_filter']:
            r = re.compile(config['id_filter'])
            valid_ids = list(filter(r.match, valid_ids))

        new_ids = list(set(self.ids) & set(valid_ids))
        #self.ids = new_ids
        from random import shuffle
        shuffle(new_ids)
        logger.debug(f"Reduced {len(self.ids )}->{len(new_ids)}")
        self.ids = new_ids
        self.ids.sort()
        #self.ids = new_ids[:len(new_ids)//50]
        #self.ids = new_ids[:len(new_ids)//5]
        self.num_images = len(self.ids)
