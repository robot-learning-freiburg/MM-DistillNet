import argparse
import configparser
import os
import json
from collections import OrderedDict

import pandas as pd
# Local Imports
from src.datasets.ArgoverseDataset import ArgoverseDataset
from src.datasets.FLIRDataset import FLIRDataset
from src.datasets.MultimodalDetection import MultimodalDetection
from src.datasets.transformations import Compose
from src.utils.utils import (
    evaluate,
    extract_transformations,
    load_model,
    plot_image_predictions,
    filter_model_dict,
    plot_audio_predictions,
)

from tabulate import tabulate

# PyTorch Related
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
#                         Logger Configuration
# ----------------------------------------------------------------------
# Logging
import logging
from logging.config import fileConfig
fileConfig('logs/logging_config.ini', disable_existing_loggers=False)
logger = logging.getLogger()
# Log everything for debug
fileh = logging.FileHandler(
    f"evaluate.log",
    'a'
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(module)s : %(lineno)d - %(message)s'
)
fileh.setFormatter(formatter)
fileh.setLevel(logging.DEBUG)
logger.addHandler(fileh)

# ==========================Main============================
# Get the desired parser
parser = argparse.ArgumentParser("Multi Modal Object Detection")
parser.add_argument(
    "--config_file",
    type=str,
    default="configs/yolov2_ranking_traditional.cfg",
    help="Path to a configuration file"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint path"
)
parser.add_argument(
    "--overwrite",
    type=str,
    default="",
    help="JSON like dictionary to overwrite the config."
)

parser.add_argument(
    "--just_plot",
    type=bool,
    default=False,
    help="Whether to just plot."
)

args = parser.parse_args()
if not os.path.exists(args.config_file):
    raise Exception(f"File {args.config_file} does not exist!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the configuration of this run
config = configparser.ConfigParser()
config.read(args.config_file)
config = config['DEFAULT']
config['rank'] = '0'
config['local_rank'] = '0'

if args.overwrite:
    for k, v in json.loads(args.overwrite).items():
        config[k] = str(v)

print(f"Begin evaluate with following configuration:")
print("\n" + tabulate(pd.DataFrame(
    config.items()),
    headers='keys',
    tablefmt='psql'
    )
)

# Load the teachers model
teacher_models = torch.nn.ModuleDict()
if config.getboolean('use_rgb'):
    logger.debug(f"Loading the RGB teacher model")
    teacher_models['rgb'] = load_model(config['teacher'], config, 'rgb').to(device)
if config.getboolean('use_audio'):
    # Handle special case of static teacher -- read static audio pre trained model
    teacher_models['audio'] = load_model(config['teacher'], config, 'audio_static').to(device)
if config.getboolean('use_depth'):
    logger.debug(f"Loading the depth teacher model")
    teacher_models['depth'] = load_model(config['teacher'], config, 'depth').to(device)
if config.getboolean('use_thermal'):
    logger.debug(f"Loading the thermal teacher model")
    teacher_models['thermal'] = load_model(config['teacher'], config, 'thermal').to(device)

teacher_models.eval()

logger.debug(f"Loading the student model with {config['student_modality']} modality")
student_model = load_model(config["student"], config, 'audio_student').to(device)

if args.checkpoint:
    logger.debug(f"loading checkpoint dict for student")
    checkpoint = torch.load(args.checkpoint)
    state_dict = checkpoint['state_dict']
    filtered_state_dict = filter_model_dict(student_model, state_dict)
    # load params
    student_model.load_state_dict(filtered_state_dict)
student_model.eval()

# Dataset
if config['dataset'] == 'ArgoverseDataset':
    dataset = ArgoverseDataset
elif config['dataset'] == 'CarsAugmented':
    dataset = CarsAugmented
elif config['dataset'] == 'FLIRDataset':
    dataset = FLIRDataset
elif config['dataset'] == 'MultimodalDetection':
    dataset = MultimodalDetection
else:
    raise Exception(f"Unsuported Dataset Was provided: {config['dataset']}")
test_set = dataset(
    config=config,
    mode="test",
)

# Go parallel
if (device.type == 'cuda') and (config.getint('ngpu') > 1):
    logger.info(f"Parallel exec for {config.getint('ngpu')} GPUs")
    # On Resume, prevent a teacher model of teacher model
    for modality, teacher_model in teacher_models.items():
        if not type(teacher_models) == torch.nn.DataParallel:
            teacher_models[modality] = nn.DataParallel(
                teacher_model,
                list(range(config.getint('ngpu')))
            )
    if not type(student_model) == torch.nn.DataParallel:
        student_model = nn.DataParallel(
            student_model,
            list(range(config.getint('ngpu')))
        )

# Evaluation
if not args.just_plot:
    evaluate(
        teacher_models,
        student_model,
        test_set,
        config,
    )

plot_audio_predictions(
    test_set,
    config,
    model=student_model,
)
