# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
from dataclasses import dataclass, field

import PIL.Image
import numpy as np
import torch
from torch.optim import Optimizer
import transformers
# import wandb
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import TensorBoardCallback
import datasets

from models.metaquery import MetaQueryConfig, MetaQuery
from trainer import MetaQueryTrainer, MetaQueryCallback
from trainer_utils import possible_override_args, find_newest_checkpoint, get_full_dirs
from dataset import get_train_datasets
from accelerate.utils import release_memory

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import textwrap

# transformers.logging.set_verbosity_info()

datasets.disable_caching()

from PIL import PngImagePlugin

PIL.Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class OverrideArguments:
    config_file: str = None


@dataclass
class ModelArguments:
    _gradient_checkpointing: bool = True
    vae_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    in_channels: int = 32
    vae_downsample_f: int = 32
    noise_scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    scheduler_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    mllm_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    diffusion_model_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers"
    loss_type: str = "flow"
    num_metaqueries: int = 64
    modules_to_freeze: tuple[str] = ()
    modules_to_unfreeze: tuple[str] = ()
    max_input_text_tokens: int = 256
    connector_num_hidden_layers: int = 24
    system_prompt: str = (
        "You will be given an image or its caption. Please describe the content of the image in detail in your own words."
    )

    model_name_and_path: str = None

@dataclass
class DataArguments:
    train_datasets: dict[str, float] = field(
        default_factory=lambda: {
            "default_dataset": -1,
        }
    )
    eval_dataset: str = "cc12m_t2i"
    target_image_size: int = 512


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    base_dir: str = "/path/to/base_dir"
    output_dir: str = "output"
    data_dir: str = ".cache"
    eval_on_start: bool = True
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    eval_delay: int = 0
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    optim: str = "adamw_torch"
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 0.5
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr": 1e-5})
    warmup_steps: int = 5000
    logging_steps: int = 1
    save_steps: int = 1000
    save_total_limit: int = 1
    restore_callback_states_from_checkpoint: bool = True
    seed: int = 42
    data_seed: int = 42
    bf16: bool = True
    tf32: bool = False
    fp16: bool = False
    dataloader_num_workers: int = 4
    datasets_num_proc: int = os.getenv("OMP_NUM_THREADS", 12)
    dataloader_persistent_workers: bool = False
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True
    remove_unused_columns: bool = False
    run_name: str = "test"
    report_to: str = "wandb"
    ddp_find_unused_parameters: bool = False
    overwrite_output_dir: bool = False
    resume_from_checkpoint: str = None
    accelerator_config: dict = field(default_factory=lambda: {"dispatch_batches": False})

    def __post_init__(self):
        try:
            self = possible_override_args(override_args, self)
            self = get_full_dirs(self)
        except:
            pass
        super().__post_init__()


if __name__ == "__main__":
    override_parser = transformers.HfArgumentParser((OverrideArguments))
    override_args = override_parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )[0]
    parser = transformers.HfArgumentParser(
        (OverrideArguments, ModelArguments, DataArguments, TrainingArguments)
    )
    _, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args, data_args = possible_override_args(override_args, model_args, data_args)

    assert (
        data_args.target_image_size % model_args.vae_downsample_f == 0
    ), f"Image size must be divisible by {model_args.vae_downsample_f}"
    input_size = data_args.target_image_size // model_args.vae_downsample_f

    if training_args.resume_from_checkpoint is not None:
        training_args.resume_from_checkpoint = find_newest_checkpoint(
            training_args.resume_from_checkpoint
        )
        model = MetaQuery.from_pretrained(
            training_args.resume_from_checkpoint,
            input_size=input_size,
            ignore_mismatched_sizes=True,
            **model_args.__dict__,
        )
    else:
        model = MetaQuery(
            config=MetaQueryConfig(
                input_size=input_size,
                **model_args.__dict__,
            ),
        )

    with training_args.main_process_first(local=False):
        train_dataset, eval_dataset, gt, collate_fn = get_train_datasets(
            data_args,
            training_args,
            model_args,
            model.get_tokenize_fn(),
            model.get_tokenizer(),
        )

    trainer = MetaQueryTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=[MetaQueryCallback()],
    )

    if trainer.is_world_process_zero():
        if training_args.overwrite_output_dir and os.path.exists(
            training_args.output_dir
        ):
            shutil.rmtree(training_args.output_dir)

        os.makedirs(training_args.output_dir, exist_ok=True)

        gt_images, gt_captions = gt

        figures = []
        for gt_image, gt_caption in zip(gt_images, gt_captions):

            fig, ax = plt.subplots(figsize=(6, 6))

            ax.imshow(np.array(gt_image))

            gt_caption = textwrap.fill(gt_caption, width=100)
            ax.set_title(gt_caption, fontsize=8)

            ax.axis('off')

            figures.append(fig)

        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback):
                callback._init_summary_writer(training_args)
                callback.tb_writer.add_figure("gt_images", figures, 0)

    trainer.train()
