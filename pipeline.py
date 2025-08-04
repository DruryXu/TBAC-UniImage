# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union, List
from typing import Tuple

import torch
from PIL import Image
from diffusers.pipelines.pipeline_utils import BaseOutput
import numpy as np
import PIL
from dataclasses import dataclass
from models.tbacuniimage import TBACUniImage


@dataclass
class TBACUniImagePipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    # text: Optional[List[str]] = [""]


class TBACUniImagePipeline(TBACUniImage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        caption: Optional[str] = "",
        image: Optional[
            Union[Image.Image, List[Image.Image], torch.Tensor, List[torch.Tensor]]
        ] = None,
        negative_prompt: Optional[str] = "",
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[TBACUniImagePipelineOutput, Tuple]:

        samples = self.sample_images(
            caption=caption,
            input_images=image,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        )
        if not return_dict:
            return (samples,)

        return TBACUniImagePipelineOutput(images=samples)
