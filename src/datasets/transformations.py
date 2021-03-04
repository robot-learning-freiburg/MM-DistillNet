# -*- coding: utf-8 -*-
"""Master Project -- Multi Modal Object Detection

This file contains transformations inspired from
https://github.com/uvipen/Yolo-v2-pytorch.git
Changes had to be made for thermal rgbs

"""

# General Inputs
import cv2
import logging
import math
import numpy as np
import os
import pandas as pd
from random import uniform
from tqdm import tqdm

# PyTorch Related
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import albumentations
import librosa

logger = logging.getLogger(__name__)


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for function_ in self.transforms:
            data = function_(data)
        return data


class RGBCrop(object):

    def __init__(self, max_crop=0.1):
        super().__init__()
        self.max_crop = max_crop

    def __call__(self, data):
        rgb, thermal, depth, audio, label, id = data
        height, width = rgb.shape[:2]
        xmin = width
        ymin = height
        xmax = 0
        ymax = 0
        for lb in label:
            xmin = min(xmin, lb[0])
            ymin = min(ymin, lb[1])
            xmax = max(xmax, lb[2])
            ymax = max(ymax, lb[2])
        cropped_left = uniform(0, self.max_crop)
        cropped_right = uniform(0, self.max_crop)
        cropped_top = uniform(0, self.max_crop)
        cropped_bottom = uniform(0, self.max_crop)
        new_xmin = int(min(cropped_left * width, xmin))
        new_ymin = int(min(cropped_top * height, ymin))
        new_xmax = int(max(width - 1 - cropped_right * width, xmax))
        new_ymax = int(max(height - 1 - cropped_bottom * height, ymax))

        rgb = rgb[new_ymin:new_ymax, new_xmin:new_xmax, :]
        label = [[
            lb[0] - new_xmin, lb[1] - new_ymin, lb[2] - new_xmin, lb[3] - new_ymin, lb[4]
        ] for lb in label]

        return rgb, thermal, depth, audio, label, id


class RGBVerticalFlip(object):

    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def __call__(self, data):
        rgb, thermal, depth, audio, label, id = data
        if uniform(0, 1) >= self.prob:
            rgb = cv2.flip(rgb, 1)
            width = rgb.shape[1]
            label = [[width - lb[2], lb[1], width - lb[0], lb[3], lb[4]] for lb in label]
        return rgb, thermal, depth, audio, label, id


class RGBHSVAdjust(object):

    def __init__(self, hue=30, saturation=1.5, value=1.5, prob=0.5):
        super().__init__()
        self.hue = hue
        self.saturation = saturation
        self.value = value
        self.prob = prob

    def __call__(self, data):

        def clip_hue(hue_channel):
            hue_channel[hue_channel >= 360] -= 360
            hue_channel[hue_channel < 0] += 360
            return hue_channel

        rgb, thermal, depth, audio, label, id = data
        adjust_hue = uniform(-self.hue, self.hue)
        adjust_saturation = uniform(1, self.saturation)
        if uniform(0, 1) >= self.prob:
            adjust_saturation = 1 / adjust_saturation
        adjust_value = uniform(1, self.value)
        if uniform(0, 1) >= self.prob:
            adjust_value = 1 / adjust_value
        rgb = rgb.astype(np.float32) / 255
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        rgb[:, :, 0] += adjust_hue
        rgb[:, :, 0] = clip_hue(rgb[:, :, 0])
        rgb[:, :, 1] = np.clip(adjust_saturation * rgb[:, :, 1], 0.0, 1.0)
        rgb[:, :, 2] = np.clip(adjust_value * rgb[:, :, 2], 0.0, 1.0)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_HSV2RGB)
        rgb = (rgb * 255).astype(np.float32)

        return rgb, thermal, depth, audio, label, id


class HSVAdjust(object):

    def __init__(self, hue=30, saturation=1.5, value=1.5, prob=0.5):
        super().__init__()
        self.hue = hue
        self.saturation = saturation
        self.value = value
        self.prob = prob

    def __call__(self, data):

        def clip_hue(hue_channel):
            hue_channel[hue_channel >= 360] -= 360
            hue_channel[hue_channel < 0] += 360
            return hue_channel

        rgb, thermal, depth, audio, label, id = data

        adjust_hue = uniform(-self.hue, self.hue)
        adjust_saturation = uniform(1, self.saturation)
        if uniform(0, 1) >= self.prob:
            adjust_saturation = 1 / adjust_saturation
        adjust_value = uniform(1, self.value)
        if uniform(0, 1) >= self.prob:
            adjust_value = 1 / adjust_value

        # For rgb
        rgb = rgb.astype(np.float32) / 255
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        rgb[:, :, 0] += adjust_hue
        rgb[:, :, 0] = clip_hue(rgb[:, :, 0])
        rgb[:, :, 1] = np.clip(adjust_saturation * rgb[:, :, 1], 0.0, 1.0)
        rgb[:, :, 2] = np.clip(adjust_value * rgb[:, :, 2], 0.0, 1.0)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_HSV2RGB)
        rgb = (rgb * 255).astype(np.float32)

        # For thermal
        if thermal is not None:
            thermal = thermal.astype(np.float32) / 255
            thermal = cv2.cvtColor(thermal, cv2.COLOR_RGB2HSV)
            thermal[:, :, 0] += adjust_hue
            thermal[:, :, 0] = clip_hue(thermal[:, :, 0])
            thermal[:, :, 1] = np.clip(adjust_saturation * thermal[:, :, 1], 0.0, 1.0)
            thermal[:, :, 2] = np.clip(adjust_value * thermal[:, :, 2], 0.0, 1.0)
            thermal = cv2.cvtColor(thermal, cv2.COLOR_HSV2RGB)
            thermal = (thermal * 255).astype(np.float32)

        # For depth
        if depth is not None:
            depth = depth.astype(np.float32) / 255
            depth = cv2.cvtColor(depth, cv2.COLOR_RGB2HSV)
            depth[:, :, 0] += adjust_hue
            depth[:, :, 0] = clip_hue(depth[:, :, 0])
            depth[:, :, 1] = np.clip(adjust_saturation * depth[:, :, 1], 0.0, 1.0)
            depth[:, :, 2] = np.clip(adjust_value * depth[:, :, 2], 0.0, 1.0)
            depth = cv2.cvtColor(depth, cv2.COLOR_HSV2RGB)
            depth = (depth * 255).astype(np.float32)
        return rgb, thermal, depth, audio, label, id


#class AudioAugmenter(object):
#    """Based on https://www.kaggle.com/huseinzol05/sound-augmentation-librosa"""
#
#    def __init__(self):
#        super().__init__()
#        self.augmentations = [
#            'pitch',
#            'speed',
#            'value',
#            'distribution_noise',
#            'hpss'
#        ]
#
#    def __call__(self, data):
#        rgb, thermal, depth, audio, label, id = data
#
#        augmentation = np.random.choice(self.audio_augmentations)
#
#        # Pitch Change
#        if 'pitch' in augmentation:
#            bins_per_octave = 12
#            pitch_pm = 2
#            pitch_change = pitch_pm * 2*(np.random.uniform())
#            for i in range(audio):
#                audio[i] = librosa.effects.pitch_shift(
#                    audio[i].astype('float64'),
#                    sample_rate=44100,
#                    n_steps=pitch_change,
#                    bins_per_octave=bins_per_octave
#                )
#        if 'speed' in augmentation:
#            speed_change = np.random.uniform(low=0.9, high=1.1)
#            for i in range(audio):
#                tmp = librosa.effects.time_stretch(
#                    audio[i].astype('float64'), speed_change
#                )
#                minlen = min(audio[i].shape[0], audio[i].shape[0])
#                audio[i] *= 0
#                audio[i][0:minlen] = tmp[0:minlen]
#
#        if 'value' in augmentation:
#            dyn_change = np.random.uniform(low=0.0, high=1.0)
#            for i in range(audio):
#                audio[i] = audio[i] * dyn_change
#
#        if 'distribution_noise' in augmentation:
#            noise_amp = 0.005*np.random.uniform()*np.amax(audio)
#            factor = np.random.normal(size=audio[0].shape[0])
#            for i in range(audio):
#                audio[i] = audio[i].astype('float64') + noise_amp * factor
#
#        if 'hpss' in augmentations:
#            for i in range(audio):
#                audio[i] = librosa.effects.hpss(audio[i].astype('float64'))
#
#        return rgb, thermal, depth, audio, label, id


class Audio2Spectogram(object):

    def __call__(self, audio):
        audios = []
        for individual_audio in audio:
            audios.append(
                librosa.feature.melspectrogram(
                    y=individual_audio,
                    sr=44100,
                    n_fft=1024,
                    hop_length=256,
                    n_mels=80,
                )
            )
        audio = np.transpose(np.stack(audios), (1, 2, 0))
        return audio


class Resize(object):

    def __init__(self, rgb_size=416, thermal_size=416, depth_size=416, audio_size=256):
        super().__init__()
        self.rgb_size = rgb_size
        self.thermal_size = thermal_size
        self.depth_size = depth_size
        self.audio_size = audio_size

    def __call__(self, data):
        rgb, thermal, depth, audio, label, id = data

        height, width = rgb.shape[:2]
        rgb = cv2.resize(rgb, (self.rgb_size, self.rgb_size))
        if thermal is not None:
            thermal = cv2.resize(thermal, (self.thermal_size, self.thermal_size))
        if depth is not None:
            depth = cv2.resize(depth, (self.depth_size, self.depth_size))
        if audio is not None:
            audio = cv2.resize(
                audio,
                dsize=(self.audio_size, self.audio_size),
                interpolation=cv2.INTER_CUBIC
            )

        width_ratio = float(self.rgb_size) / width
        height_ratio = float(self.rgb_size) / height
        if label is not None:
            new_label = []
            for lb in label:
                resized_xmin = lb[0] * width_ratio
                resized_ymin = lb[1] * height_ratio
                resized_xmax = lb[2] * width_ratio
                resized_ymax = lb[3] * height_ratio
                new_label.append([
                    resized_xmin,
                    resized_ymin,
                    resized_xmax,
                    resized_ymax,
                    lb[4]
                ])
            label = new_label

        return rgb, thermal, depth, audio, label, id


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, data):
        rgb, thermal, depth, audio, label, id = data

        rgb = (rgb.astype(np.float32) - self.mean) / self.std
        #if thermal is not None:
        #    thermal = (thermal.astype(np.float32) - self.mean) / self.std
        #if depth is not None:
        #    depth = (depth.astype(np.float32) - self.mean) / self.std

        return rgb, thermal, depth, audio, label, id


class ImageAugmenter(object):

    def __call__(self, data):
        rgb, thermal, depth, audio, label, id = data

        rgb = rgb.astype(np.float32)
        height, width, _ = rgb.shape
        albumentations_transform_pixel = {
            'Blur':albumentations.Blur(),
            #'CLAHE':albumentations.CLAHE(),
            'ChannelDropout':albumentations.ChannelDropout(),
            'ChannelShuffle':albumentations.ChannelShuffle(),
            'CoarseDropout':albumentations.CoarseDropout(),
            #'Equalize':albumentations.Equalize(),
            #'FancyPCA':albumentations.FancyPCA(),
            'GaussNoise':albumentations.GaussNoise(),
            'GaussianBlur':albumentations.GaussianBlur(),
            #'GlassBlur':albumentations.GlassBlur(),
            'HueSaturationValue':albumentations.HueSaturationValue(),
            'IAAAdditiveGaussianNoise':albumentations.IAAAdditiveGaussianNoise(),
            #'ISONoise':albumentations.ISONoise(),
            'RGBShift':albumentations.RGBShift(),
            'RandomBrightnessContrast':albumentations.RandomBrightnessContrast(),
            'RandomFog':albumentations.RandomFog(),
            #'RandomGamma':albumentations.RandomGamma(),
            'RandomRain':albumentations.RandomRain(),
            'RandomShadow':albumentations.RandomShadow(),
            'RandomSnow':albumentations.RandomSnow(),
            'RandomSunFlare':albumentations.RandomSunFlare(),
            'Solarize':albumentations.Solarize(),
        }
        albumentations_transform_bbox = {
            #'HorizontalFlip':albumentations.HorizontalFlip(),
            #'VerticalFlip':albumentations.VerticalFlip(),
            #'CenterCrop':albumentations.CenterCrop(height=height-10, width=width-10, p=0.5),
            #'RandomCropNearBBox':albumentations.RandomCropNearBBox(p=0.5),
            #'Crop':albumentations.Crop(x_min=10, y_min =10, y_max=height-10, x_max=width-10, p=0.5),
            #'ElasticTransform':albumentations.ElasticTransform(),
            #'ShiftScaleRotate':albumentations.ShiftScaleRotate(),
        }
        transform = np.random.choice(['None'] + list(albumentations_transform_pixel.keys()) +        list(albumentations_transform_bbox.keys()))

        if transform in albumentations_transform_pixel:
            aug = albumentations.Compose([
                    albumentations_transform_pixel[transform]
                ],
                bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
            )
            try:
                annots = np.array(annots).astype(np.float32)
                aug_result = aug(image=rgb, bboxes=annots[:,:4], labels=annots[:,4])
                rgb = aug_result['image']
                annots = np.hstack([aug_result['bboxes'], np.array(aug_result['labels']).reshape(-1, 1)])
            except Exception as e:
                print(f"transform={transform} aug_result['bboxes']={aug_result['bboxes']}            aug_result['labels']={aug_result['labels']}")
                raise Exception(e)

        elif transform in albumentations_transform_bbox:
            aug = albumentations.Compose([
                    albumentations_transform_bbox[transform]
                ],
                bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
            )
            try:
                annots = np.array(annots).astype(np.float32)
                aug_result = aug(image=rgb, bboxes=annots[:,:4], labels=annots[:,4])
                rgb = aug_result['image']
                label = np.hstack([aug_result['bboxes'], np.array(aug_result['labels']).reshape(-1, 1)])
            except Exception as e:
                print(f"transform={transform} aug_result['bboxes']={aug_result['bboxes']}            aug_result['labels']={aug_result['labels']}")
                raise Exception(e)

        return rgb, thermal, depth, audio, label, id

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, common_size=512):
        self.common_size = common_size

    def __call__(self, data):
        rgb, thermal, depth, audio, label, id = data
        height, width, _ = rgb.shape
        if height > width:
            scale = self.common_size / height
            resized_height = self.common_size
            resized_width = int(width * scale)
        else:
            scale = self.common_size / width
            resized_height = int(height * scale)
            resized_width = self.common_size


        rgb = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        rgb_new = np.zeros((self.common_size, self.common_size, 3))
        rgb_new[0:resized_height, 0:resized_width] = rgb

        thermal_new = thermal
        if thermal is not None:
            thermal = cv2.resize(thermal, (resized_width, resized_height))
            thermal_new = np.zeros((self.common_size, self.common_size))
            thermal_new[0:resized_height, 0:resized_width] = thermal

        depth_new = depth
        if depth is not None:
            depth = cv2.resize(depth, (resized_width, resized_height))
            depth_new = np.zeros((self.common_size, self.common_size, 3))
            depth_new[0:resized_height, 0:resized_width] = depth

        audio_new = audio
        if audio is not None:
            audio_new = cv2.resize(
                audio,
                dsize=(self.common_size, self.common_size),
                interpolation=cv2.INTER_CUBIC
            )

        # annots[:, :4] *= scale
        new_label = None
        if label is not None:
            new_label = []
            for lb in label:
                resized_xmin = lb[0] * scale
                resized_ymin = lb[1] * scale
                resized_xmax = lb[2] * scale
                resized_ymax = lb[3] * scale
                new_label.append([
                    resized_xmin,
                    resized_ymin,
                    resized_xmax,
                    resized_ymax,
                    lb[4]
                ])

        return rgb_new, thermal_new, depth_new, audio_new, new_label, id


class AudioAugmenter(object):

    def __call__(self, data):
        rgb, thermal, depth, audio, label, id = data

        augmentation = np.random.choice([
            'None',
            #'pitch',
            #'speed',
            'distribution_noise',
            #'value', #Value says how far away the car is
        ])

        # Pitch Change
        if 'pitch' in augmentation:
            bins_per_octave = 12
            pitch_pm = 2
            pitch_change = pitch_pm * 2*(np.random.uniform())
            for i in range(len(audio)):
                audio[i] = librosa.effects.pitch_shift(
                    audio[i].astype('float64'),
                    44100,
                    n_steps=pitch_change,
                    bins_per_octave=bins_per_octave
                )
        if 'speed' in augmentation:
            speed_change = np.random.uniform(low=0.95, high=1.05)
            for i in range(len(audio)):
                tmp = librosa.effects.time_stretch(
                    audio[i].astype('float64'), speed_change
                )
                minlen = min(audio[i].shape[0], audio[i].shape[0])
                audio[i] *= 0
                audio[i][0:minlen] = tmp[0:minlen]

        if 'value' in augmentation:
            dyn_change = np.random.uniform(low=0.0, high=1.0)
            for i in range(len(audio)):
                audio[i] = audio[i] * dyn_change

        if 'distribution_noise' in augmentation:
            noise_amp = 0.0005*np.random.uniform()*np.amax(audio[0])
            factor = np.random.normal(size=audio[0].shape[0])
            for i in range(len(audio)):
                audio[i] = audio[i].astype('float64') + noise_amp * factor

        # Transform to spectogram
        spectogramer = Audio2Spectogram()
        audio = spectogramer(audio)

        return rgb, thermal, depth, audio, label, id

class ThermalAugmenter(object):

    def __call__(self, data):
        rgb, thermal, depth, audio, label, id = data

        #thermal = thermal.astype(np.float32)
        albumentations_transform_pixel = {
            'Blur':albumentations.Blur(),
            #'MedianBlur':albumentations.MedianBlur(3),
            #'MotionBlur':albumentations.MotionBlur(),
            #'CLAHE':albumentations.CLAHE(),
            'GaussNoise':albumentations.GaussNoise(),
            #'GaussianBlur':albumentations.GaussianBlur(),
            #'GlassBlur':albumentations.GlassBlur(),
            #'IAAAdditiveGaussianNoise':albumentations.IAAAdditiveGaussianNoise(),
            'RandomBrightnessContrast':albumentations.RandomBrightnessContrast(),
            #'RandomFog':albumentations.RandomFog(),
            #'RandomRain':albumentations.RandomRain(),
            #'RandomShadow':albumentations.RandomShadow(),
            #'RandomSnow':albumentations.RandomSnow(),
            #'RandomSunFlare':albumentations.RandomSunFlare(),
            #'Solarize':albumentations.Solarize(),
        }
        transform = np.random.choice(['None'] + list(albumentations_transform_pixel.keys()))

        if transform in albumentations_transform_pixel:
            aug = albumentations.Compose([
                    albumentations_transform_pixel[transform]
                ],
                bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
            )
            try:
                annots = np.array(label).astype(np.float32)
                aug_result = aug(image=thermal, bboxes=annots[:,:4], labels=annots[:,4])
                thermal = aug_result['image']
            except Exception as e:
                print(f"transform={transform}")
                raise Exception(e)

        return rgb, thermal, depth, audio, label, id

class DepthAugmenter(object):

    def __call__(self, data):
        rgb, thermal, depth, audio, label, id = data

        #depth = depth.astype(np.float32)
        albumentations_transform_pixel = {
            'Blur':albumentations.Blur(),
            'MedianBlur':albumentations.MedianBlur(),
            'MotionBlur':albumentations.MotionBlur(),
            'GaussNoise':albumentations.GaussNoise(),
            'GaussianBlur':albumentations.GaussianBlur(),
            'GlassBlur':albumentations.GlassBlur(),
            'IAAAdditiveGaussianNoise':albumentations.IAAAdditiveGaussianNoise(),
        }
        transform = np.random.choice(['None'] + list(albumentations_transform_pixel.keys()))

        if transform in albumentations_transform_pixel:
            aug = albumentations.Compose([
                    albumentations_transform_pixel[transform]
                ],
                bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
            )
            try:
                annots = np.array(annots).astype(np.float32)
                aug_result = aug(image=depth, bboxes=annots[:,:4], labels=annots[:,4])
                depth = aug_result['image']
            except Exception as e:
                print(f"transform={transform} aug_result['bboxes']={aug_result['bboxes']}            aug_result['labels']={aug_result['labels']}")
                raise Exception(e)

        return rgb, thermal, depth, audio, label, id

