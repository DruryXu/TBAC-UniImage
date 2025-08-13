# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple, Union

import os
import datasets
import numpy as np
import torch
# import wandb
from diffusers.pipelines.pipeline_utils import numpy_to_pil
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, is_datasets_available, TrainerCallback, TrainerState, TrainerControl
from transformers.integrations import TensorBoardCallback
from transformers.trainer import (
    tpu_spmd_dataloader,
    time,
    speed_metrics,
    math,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, SaveStrategy
from transformers.utils import (
    is_torch_xla_available,
)

from accelerate.utils import (
    DistributedType,
)

import gc
# import logging
from tabulate import tabulate

# logger = logging.get_logger(__name__)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(
        XLA_FSDPV2_MIN_VERSION
    )
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False


class MetaQueryTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # handle multipe eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        start_time = time.time()

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    @staticmethod
    def metaquery_eval_data_collator(features):
        batch = {}
        for k in features[0].keys():
            batch[k] = [f[k] for f in features]

        return batch

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        if (
            hasattr(self, "_eval_dataloader")
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloader)
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.metaquery_eval_data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(
                eval_dataset, description="evaluation"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="evaluation"
            )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": False,
            "persistent_workers": False,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            self._eval_dataloader = eval_dataloader

        return self.accelerator.prepare(eval_dataloader)

    def log_images(self, logs: Dict[str, float]) -> None:
        logs["step"] = self.state.global_step

        for callback in self.callback_handler.callbacks:
            if isinstance(callback, TensorBoardCallback) and self.state.is_world_process_zero:
                callback.tb_writer.add_images(
                    "generated_images",
                    logs["images"],
                    self.state.global_step
                )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)

        sample_kwargs = {
            "guidance_scale": 3.0,
            "image_guidance_scale": 1.5,
            "num_inference_steps": 30,
            "return_tensor": True,
            "negative_prompt": "",
        }

        with torch.no_grad():
            samples = model.sample_images(**inputs, **sample_kwargs)

        samples = self._nested_gather(samples)

        self.log_images({"images": samples})

        del samples
        gc.collect()

        return (None, None, None)

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):

        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(
                metrics=metrics, trial=trial
            )

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        # """
        # Perform a training step on a batch of inputs.

        # Subclass and override to inject custom behavior.

        # Args:
        #     model (`nn.Module`):
        #         The model to train.
        #     inputs (`Dict[str, Union[torch.Tensor, Any]]`):
        #         The inputs and targets of the model.

        #         The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
        #         argument `labels`. Check your model's documentation for all accepted arguments.

        # Return:
        #     `torch.Tensor`: The tensor with training loss on this batch.
        # """
        # model.train()
        # if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
        #     self.optimizer.train()

        loss = super().training_step(
            model,
            inputs,
            num_items_in_batch
        )
        
        grad_norm = self.accelerator.clip_grad_norm_(
            model.parameters(),
            float("inf"),
        )

        decay = 0.99
        # detect loss spike
        if not hasattr(self, "running_loss"):
            self.running_loss = loss.item()
        self.running_loss = decay * self.running_loss + (1 - decay) * loss.item()

        if loss.item() > 20 * self.running_loss:
            self.accelerator.print(f"Loss Spiked: {loss.item()} > 20 * {self.running_loss}, Skip this Step")
            self.optimizer.zero_grad()

        # detect grad norm spike
        if not hasattr(self, "running_grad_norm"):
            self.running_grad_norm = grad_norm
        self.running_grad_norm = (
            decay * self.running_grad_norm + (1 - decay) * grad_norm
        )

        if grad_norm > 25 * self.running_grad_norm and grad_norm > 1:
            self.accelerator.print(f"Grad Norm Spiked: {grad_norm} > 25 * {self.running_grad_norm}, Skip this Step")
            self.optimizer.zero_grad()

        if np.isnan(grad_norm) or grad_norm > 1e6:
            self.accelerator.print(
                f"NaN grad norm detected in process {self.args.process_index} on {os.uname().nodename}, Skip this Step"
            )
            self.optimizer.zero_grad()

        return loss.detach()

    
class MetaQueryCallback(TrainerCallback):
    @staticmethod
    def print_params(model, modules_to_stat):
        for module in modules_to_stat:
            module_parts = module.split(".")
            curr_module = model
            for part in module_parts:
                curr_module = getattr(curr_module, part)
            num_params = sum(p.numel() for p in curr_module.parameters()) / 1e6
            print(f"{module} num of params: {num_params:.2f}M")

    def on_train_begin(self, args, state, control, model, **kwargs):
        if state.is_world_process_zero:
            stat = []
            for i, (n, p) in enumerate(model.named_parameters()):
                stat.append([i, n, p.shape, p.dtype, p.requires_grad])
            print(
                tabulate(stat, headers=["idx", "name", "shape", "dtype", "trainable"])
            )

            if hasattr(model.model, "connector") and not isinstance(
                model.model.connector, nn.Identity
            ):
                print(f"connector_in_dim = {model.model.connector_in_dim}")
                print(f"connector_out_dim = {model.model.connector_out_dim}")
                self.print_params(model, ["model.connector"])

            self.print_params(model, ["model.transformer"])

    def on_log(self, args, state, control, logs, **kwargs):
        if state.is_world_process_zero:
            logs['step'] = f"{state.global_step} / {state.max_steps}"
