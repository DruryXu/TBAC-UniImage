# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import yaml
from transformers.trainer_utils import get_last_checkpoint
import random
import numpy as np
import torch


class ProcessorWrapper:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, tensor):
        return self.processor(tensor, return_tensors="pt")["pixel_values"].squeeze(0)


def possible_override_args(override_args, *args):
    if hasattr(override_args, "config_file") and override_args.config_file is not None:
        yaml_file = os.path.join("configs", override_args.config_file)
        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)

        for arg in args:
            for key, value in config.items():
                if hasattr(arg, key):
                    setattr(arg, key, value)
    return args[0] if len(args) == 1 else args


def get_full_dirs(training_args):
    if not os.path.isabs(training_args.output_dir):
        training_args.output_dir = os.path.join(
            training_args.base_dir, training_args.output_dir
        )
    if not os.path.isabs(training_args.data_dir):
        training_args.data_dir = os.path.join(
            training_args.base_dir, training_args.data_dir
        )
    if not os.path.isabs(training_args.logging_dir):
        training_args.logging_dir = os.path.join(
            training_args.base_dir, training_args.logging_dir
        )
    return training_args


def find_newest_checkpoint(checkpoint_path):
    # see if checkpoint_path's child contains pt or safetensors or pth
    if os.path.isdir(checkpoint_path) and any(
        x.endswith(("pt", "safetensors", "pth")) for x in os.listdir(checkpoint_path)
    ):
        return checkpoint_path

    else:
        return get_last_checkpoint(checkpoint_path)


def seed_everything(seed=42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
