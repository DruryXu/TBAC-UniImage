# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets import load_dataset, load_from_disk
import PIL
import io
import torch
from torchvision.transforms import v2
import random
from torch.utils.data.dataset import ConcatDataset
from functools import partial

import glob
import os
import random

from datasets.features.image import Image as ImageFeature
from datasets import Features, Value

from PIL import Image as PILImage

DEFAULT_IMAGE_SIZE = (512, 512)

class SafeImage(ImageFeature):
    def decode_example(self, value, token_per_repo_id=None):
        
        try:
            image = super().decode_example(value, token_per_repo_id)
            return image
        except Exception as e:
            print(f"Warning: Corrupted image data found. Error: {e}. Returning a placeholder image.")
            return PILImage.new("RGB", DEFAULT_IMAGE_SIZE, (0, 0, 0))

def read_image(example):
    if example is None or isinstance(example, PILImage.Image):
        return example

    if isinstance(example, str):
        try:
            image = PILImage.open(example).convert("RGB")
        except Exception as e:
            print(f"Read Image {example} Failed with {e}, Return a Placeholder Image.")
            image = PILImage.new("RGB", DEFAULT_IMAGE_SIZE)
    else:
        print(f"Unexcpeted Input {example}, Return a Placeholder Image.")
        image = PILImage.new("RGB", DEFAULT_IMAGE_SIZE)

    return image

def delete_keys_except(batch, except_keys):
    keys_to_delete = [key for key in list(batch.keys()) if key not in except_keys]
    for key in keys_to_delete:
        del batch[key]
    return batch

def _t2i_process_fn(batch, target_transform):
    images = batch["image"]
    captions = batch["caption"]
    captions = ["" if caption is None else caption for caption in captions]
    for i in range(len(images)):
        images[i] = images[i].convert("RGB")

    batch["target"] = [
        target_transform(image) if image is not None else None for image in images
    ]
    rand_probs = torch.rand((len(images), 1))
    null_caption_mask = rand_probs < 0.1
    captions = [
        caption if not null_caption_mask[i] else ""
        for i, caption in enumerate(captions)
    ]
    batch["caption"] = captions
    delete_keys_except(batch, ["target", "caption"])

    return batch

def _t2i_process_fn_map(example, target_transform):
    image = example["image"].convert("RGB")
    caption = example["caption"]
    
    caption = "" if caption is None else caption

    example["target"] = target_transform(image)

    rand_probs = random.random()
    null_caption_mask = rand_probs < 0.1
    caption = caption if not null_caption_mask else ""

    example["caption"] = caption
    delete_keys_except(example, ["target", "caption"])

    return example

def t2i_eval_process_fn(batch):
    captions = batch["caption"]
    batch["caption"] = captions
    delete_keys_except(batch, ["caption"])
    return batch

def _mix_process_fn(batch, target_transform):
    source_images = batch["source_image"]
    caption = batch["caption"]

    if isinstance(caption[0], list):
        caption = [item[-1] for item in caption]

    rand_probs = torch.rand((len(batch["target_image"]), 1))
    null_caption_mask = rand_probs < 0.2
    null_image_mask = (rand_probs >= 0.1) & (rand_probs < 0.3)
    caption = [
        caption if not null_caption_mask[i] else "" for i, caption in enumerate(caption)
    ]

    for i in range(len(source_images)):
        image = source_images[i]

        if image is None:
            continue
        
        image = read_image(image)

        if not null_image_mask[i]:
            source_images[i] = image.convert("RGB")
        else:
            source_images[i] = PILImage.new("RGB", (image.width, image.height))

    batch["caption"], batch["input_images"] = caption, source_images
    
    batch["target"] = [
        target_transform(read_image(img).convert("RGB")) for img in batch["target_image"]
    ]
    delete_keys_except(batch, ["target", "input_images", "caption"])
    return batch

def _inst_process_fn(batch, target_transform):
    source_images = batch["source_image"]
    caption = batch["caption"]

    rand_probs = torch.rand((len(batch["target_image"]), 1))
    null_caption_mask = rand_probs < 0.2
    null_image_mask = (rand_probs >= 0.1) & (rand_probs < 0.3)
    caption = [
        caption if not null_caption_mask[i] else "" for i, caption in enumerate(caption)
    ]
    source_images = (
        [
            (
                image.convert("RGB")
                if not null_image_mask[i]
                else PIL.Image.new("RGB", (image.width, image.height))
            )
            for i, image in enumerate(source_images)
        ]
        if source_images is not None
        else None
    )
    batch["caption"], batch["input_images"] = caption, source_images
    batch["target"] = [
        target_transform(img.convert("RGB")) for img in batch["target_image"]
    ]
    delete_keys_except(batch, ["target", "input_images", "caption"])
    return batch

def _inst_process_fn_map(example, target_transform):
    source_image = example["source_image"]
    
    if not isinstance(source_image, list):
        source_image = [source_image]

    caption = example["caption"]
    if not isinstance(caption, list):
        caption = [caption]

    rand_probs = random.random()

    null_caption_mask = rand_probs < 0.2
    null_image_mask = (rand_probs >= 0.1) & (rand_probs < 0.3)
    caption = caption[0] if not null_caption_mask else ""

    source_image = (
        [
            (
                image
                if not null_image_mask
                else PIL.Image.new("RGB", (image.width, image.height))
            )
            for i, image in enumerate(source_image)
        ]
        if source_image is not None
        else None
    )

    example["caption"], example["input_images"] = caption, source_image
    example["target"] = target_transform(example["target_image"].convert("RGB"))

    delete_keys_except(example, ["target", "input_images", "caption"])
    return example

def inst_eval_process_fn(batch):
    source_image = batch["source_image"]

    for i in range(len(source_image)):
        source_image[i] = read_image(source_image[i])
            
    caption = batch["caption"]

    batch["caption"], batch["input_images"] = caption, source_image
    delete_keys_except(batch, ["caption", "input_images"])
    return batch


def _collate_fn(batch, tokenize_func, tokenizer):
    none_idx = [i for i, example in enumerate(batch) if example["target"] is None]
    if len(none_idx) > 0:
        batch = [example for i, example in enumerate(batch) if i not in none_idx]
    return_dict = {"target": torch.stack([example["target"] for example in batch])}
    input_images = [
        example["input_images"] if "input_images" in example else None
        for example in batch
    ]

    if any(input_images):
        (
            return_dict["input_ids"],
            return_dict["attention_mask"],
            return_dict["pixel_values"],
            return_dict["image_sizes"],
        ) = tokenize_func(
            tokenizer, [example["caption"] for example in batch], input_images
        )
    else:
        return_dict["input_ids"], return_dict["attention_mask"] = tokenize_func(
            tokenizer, [example["caption"] for example in batch]
        )
    return return_dict


def get_train_datasets(data_args, training_args, model_args, tokenize_func, tokenizer):

    # We have reorganized the data using our own custom method, so the data handling logic
    # differs from the standard format for datasets downloaded from Hugging Face.
    #
    # When using your own data, please format it as follows:
    # - For text-to-image generation: Organize the data in the `blip3o` format.
    # - For image-text-to-image generation: Structure the data as a triplet of
    #   `source_image`, `prompt`, and `target_image`.

    safe_image_features = Features({
        'jpg': SafeImage(),
        'txt': Value("string"),
        '__key__': Value("string"),
        '__url__': Value("string")
    })

    if "4o" in data_args.train_datasets:
        train_dataset = load_from_disk(
            "path/to/blip-3o-60k/and/sharegpt-4o-image"
        )

        train_dataset = train_dataset.rename_column("prompt", "caption")
        train_dataset = train_dataset.shuffle(seed=training_args.data_seed)

        eval_dataset = train_dataset.select(
            range(training_args.world_size)
        )
    elif "4oedit" in data_args.train_datasets:
        train_dataset = load_from_disk(
            "path/to/GPT-Image-Edit-1.5M"
        )

        train_dataset = train_dataset.shuffle(seed=training_args.data_seed)

        eval_dataset = train_dataset.select(
            range(training_args.world_size)
        )
    else:
        if "blip3o" in data_args.train_datasets:
            image_folders = [
                "path/to/BLIP3o-Pretrain-Long-Caption",
                "path/to/BLIP3o-Pretrain-JourneyDB"
            ]

        data_files = []
        for folder in image_folders:
            data_files.extend(glob.glob(os.path.join(folder, "*.tar")))

        train_dataset = load_dataset(
            "webdataset",
            data_files=data_files,
            split="train",
            features=safe_image_features,
            streaming=True
        )

        train_dataset = train_dataset.shuffle(seed=training_args.data_seed, buffer_size=10_000)

        train_dataset = train_dataset.rename_column("jpg", "image")
        train_dataset = train_dataset.rename_column("txt", "caption")
        train_dataset = train_dataset.remove_columns(
            [
                col
                for col in train_dataset.column_names
                if not col in (["image", "caption"])
            ]
        )

        eval_dataset = load_dataset(
            "webdataset",
            data_files=data_files[0],
            split="train"
        )

        eval_dataset = eval_dataset.rename_column("jpg", "image")
        eval_dataset = eval_dataset.rename_column("txt", "caption")
        eval_dataset = eval_dataset.remove_columns(
            [
                col
                for col in eval_dataset.column_names
                if not col in (["image", "caption"])
            ]
        )

        eval_dataset = eval_dataset.select(
            range(training_args.world_size)
        )

    target_transform = v2.Compose(
        [
            v2.Resize(data_args.target_image_size),
            v2.CenterCrop(data_args.target_image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5]),
        ]
    )

    ground_truth_transform = v2.Compose(
        [
            v2.Resize(data_args.target_image_size),
            v2.CenterCrop(data_args.target_image_size),
        ]
    )

    if "4o" in data_args.train_datasets or "4oedit" in data_args.train_datasets:
        mix_process_fn = partial(_mix_process_fn, target_transform=target_transform)
        train_dataset = train_dataset.shuffle(seed=training_args.data_seed)

        train_dataset.set_transform(mix_process_fn)

    elif "blip3o" in data_args.train_datasets:
        t2i_process_fn = partial(_t2i_process_fn_map, target_transform=target_transform)
        
        train_dataset = train_dataset.map(t2i_process_fn)
    else:
        t2i_process_fn = partial(_t2i_process_fn, target_transform=target_transform)

        train_dataset.set_transform(t2i_process_fn)

    collate_fn = partial(_collate_fn, tokenize_func=tokenize_func, tokenizer=tokenizer)

    gt_images, gt_captions = [], []

    for item in eval_dataset:
        gt_images.append(
            item["image"] if "image" in eval_dataset.column_names else item["target_image"]
        )

        for i in range(len(gt_images)):
            gt_images[i] = read_image(gt_images[i])

        gt_captions.append(
            item["caption"] if not isinstance(item["caption"], list) else item["caption"][-1]
        )

    gt_images = [ground_truth_transform(image.convert("RGB")) for image in gt_images]

    if "blip3o" in data_args.train_datasets:
        eval_dataset.set_transform(t2i_eval_process_fn)
    else:
        eval_dataset.set_transform(inst_eval_process_fn)

    return train_dataset, eval_dataset, (gt_images, gt_captions), collate_fn
