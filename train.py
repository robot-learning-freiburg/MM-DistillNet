# -*- coding: utf-8 -*-
"""
There is More than Meets the Eye: Self-Supervised Multi-Object
Detection and Tracking with Sound by Distilling Multimodal Knowledge
This code reproduces the finding from the above
paper
"""
# -------------------------------------------------------------
#                               Imports
# -------------------------------------------------------------

# General Inputs
import argparse
import configparser
import logging
import json
import os
from datetime import datetime
from logging.config import fileConfig

# Local Imports
from src.datasets.MultimodalDetection import MultimodalDetection
from src.optimization.train_methods import train
from src.utils.utils import (
    evaluate, load_model,
    filter_model_dict,
    make_reproducible_run,
    init_weights,
)

# Third Party
import torch
import torch.distributed as dist


# ----------------------------------------------------------------------
#                         Logger Configuration
# ----------------------------------------------------------------------
# Logging
fileConfig('logs/logging_config.ini', disable_existing_loggers=False)
logger = logging.getLogger()


# ----------------------------------------------------------------------
#                            Methods
# ----------------------------------------------------------------------
def pretrain(
    teacher_models,
    student_model,
    config,
    train_set,
    val_set,
):
    """
    Pre-trains a Model before traditional/adversarial training
    Args:
            teacher_models: A dict of pretrained model.
            student_model: The model to be trained
            config: A parsed configuration file
            train_set: A dataset object from where to draw inputs
            val_set:  A dataset object used for validation

    Returns:
            Nothing is returned but student model weights are improved
    """
    # Pre- Train the teacher model and student in adversarial
    # Or if it is vanilla training proceed anyhow with it

    # If a path is provided, load checkpoint
    if os.path.exists(config['pretrain']):
        checkpoint = torch.load(
            config['pretrain']
        )
        best_loss = checkpoint['best_loss']
        best_epoch = checkpoint['best_epoch']
        model_dict = filter_model_dict(
            student_model,
            checkpoint['state_dict']
        )
        student_model.load_state_dict(model_dict)
        logging.warn(f"Pretrain from {config['pretrain']}")
        logging.info(f"best_loss={best_loss} best_epoch={best_epoch}")
        logging.info(f"best_epoch={best_epoch}")
        return

    if config.getboolean('pretrain'):

        old_exp_name = config['exp_name']
        config['exp_name'] = f"{config['exp_name']}/pretrain"
        os.makedirs(config['exp_name'], exist_ok=True)
        logger.warn(f"Adversarial Pretrain stage on {config['exp_name']}")

        train(
            teacher_models,
            student_model,
            config,
            train_set,
            val_set,
            method='traditional'
        )
        config['exp_name'] = old_exp_name
    return


def train_multimodal_detection(config):
    """
    This utility performs multi modal training
    as dictated by a config file
    Args:
            config: A parsed configuration file

    Returns:
            Nothing is returned but student model weights are improved
    """

    # ===============================Model==================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make the run reproducible
    make_reproducible_run(config.getint('seed'))

    # Load the teachers model
    teacher_models = torch.nn.ModuleDict()
    if config.getboolean('use_rgb'):
        teacher_models['rgb'] = load_model(config['teacher'],
                                           config, 'rgb').to(device)
    if config.getboolean('use_audio'):
        teacher_models['audio'] = load_model(config['teacher'],
                                             config, 'audio_static').to(device)
    if config.getboolean('use_depth'):
        teacher_models['depth'] = load_model(config['teacher'],
                                             config, 'depth').to(device)
    if config.getboolean('use_thermal'):
        teacher_models['thermal'] = load_model(config['teacher'],
                                               config, 'thermal').to(device)

    # Make sure all teachers are in eval mode
    teacher_models.eval()

    # =========================Dataset===============================
    # Handle data
    logger.info("Obtaining the dataset...")
    if config['dataset'] == 'MultimodalDetection':
        dataset = MultimodalDetection
    else:
        raise Exception(f"Unsuported Dataset : {config['dataset']}")
    train_set = dataset(
        config=config,
        mode="train",
    )

    val_set = dataset(
        config=config,
        mode="val",
    )

    student_model = load_model(config['student'], config, 'audio_student').to(device)
    logger.debug(f"student_model={student_model}")

    # Level of initialization for the student
    # Load pre-trained weights for sure but decide if we
    # un-train a weight via config
    if config.getboolean('weights_init'):
        if hasattr(student_model.model_classifier, 'rgb'):
            init_weights(student_model.model_classifier.rgb.header)
            init_weights(student_model.model_regressor.rgb.header)
        else:
            raise Exception("No RGB")
        if hasattr(student_model.model_classifier, 'audio'):
            init_weights(student_model.model_classifier.audio.header)
            init_weights(student_model.model_regressor.audio.header)

    # =============================Train and Eval============================
    # Perform the actual Training
    tick = datetime.now()
    # Pretrain if needed
    pretrain(
        teacher_models,
        student_model,
        config,
        train_set,
        val_set,
    )

    # The actual training phase
    train(
        teacher_models,
        student_model,
        config,
        train_set,
        val_set,
        method=config['train_method'],
    )
    tock = datetime.now()
    diff = tock - tick
    logger.warn(f"Completed {config['exp_name']} after ({str(diff)})...")

    # Evaluate the performance of the best run
    student_model = load_model(config['student'], config, 'audio_student').to(device)
    map_location = {'cuda:%d' % 0: f"cuda:{config['rank']}"}
    checkpoint = torch.load(f"{config['exp_name']}/best.{config['rank']}.pth.tar",
                            map_location=map_location)
    new_state_dict = filter_model_dict(student_model, checkpoint['state_dict'])
    # load params
    student_model.load_state_dict(new_state_dict)
    student_model.eval()

    evaluate(
        teacher_models,
        student_model,
        val_set,
        config
    )

    logger.warn(f"Finished with everything...\n\n")

    return


# -----------------------------------------------------------------------
#                                Main
# -----------------------------------------------------------------------
if __name__ == "__main__":

    # Get the desired parser
    parser = argparse.ArgumentParser("Multi Modal Object Detection")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/best.cfg",
        help="Path to a configuration file"
    )

    parser.add_argument(
        "--overwrite",
        type=str,
        default="",
        help="JSON like dictionary to overwrite the config."
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="rank of this process"
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="This local_rank is used to set the device."
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="The number of cores"
    )

    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        raise Exception(f"File {args.config_file} does not exist!")

    # Get the configuration of this run
    config = configparser.ConfigParser()
    config.read(args.config_file)
    config = config['DEFAULT']

    if args.overwrite:
        for k, v in json.loads(args.overwrite).items():
            config[k] = str(v)
            logger.debug(f"Overwrite {k}->{v}")
    config['local_rank'] = str(args.local_rank)
    config['rank'] = str(args.rank)

    # Handle run area creation
    if not os.path.isdir(f"{config['exp_name']}"):
        os.mkdir(f"{config['exp_name']}")

    # Log everything for debug
    fileh = logging.FileHandler(
        f"{config['exp_name']}/{config['exp_name']}.{args.rank}.log",
        'a'
    )
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s : %(lineno)d - %(message)s'
    )
    fileh.setFormatter(formatter)
    fileh.setLevel(logging.DEBUG)
    logger.addHandler(fileh)

    # Taken from
    # https://pytorch.org/tutorials/beginner/aws_distributed_training_tutorial.html
    if (config.getint('ngpu') > 1) and config['engine'] == 'DistributedDataParallel':
        # Should run python train.py --rank 0 --local_rank 0 --checkpoint <>
        # Should run python train.py --rank 1 --local_rank 1 --checkpoint <>
        world_size = config.getint('ngpu') * args.nodes
        config['world_size'] = str(world_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '23457'
        print(f"Distributed rank {config['rank']}...")
        logger.info(f"Distributed rank {config['rank']}...")
        dist.init_process_group(
            backend='nccl',
            #  init_method=dist_url, --> Assume env
            rank=int(args.rank),
            world_size=world_size
        )
        local_rank = int(args.local_rank)
        dp_device_ids = [local_rank]
        torch.cuda.set_device(local_rank)

    # Perform the training!
    train_multimodal_detection(config)
