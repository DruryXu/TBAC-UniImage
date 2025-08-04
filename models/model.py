# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List

import torch
from torch import nn
from torchvision import transforms as v2

from transformers import PretrainedConfig, PreTrainedModel, AutoProcessor
from transformers import (
    LlavaOnevisionForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    Qwen2Config,
)

from diffusers.models.normalization import RMSNorm
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers import SanaTransformer2DModel, UNet2DConditionModel

from typing import Any, Dict, Optional, Tuple, Union

# logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class LayerCondSanaTransformer2DModel(SanaTransformer2DModel):
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: Optional[int] = 32,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        num_layers: int = 20,
        num_cross_attention_heads: Optional[int] = 20,
        cross_attention_head_dim: Optional[int] = 112,
        cross_attention_dim: Optional[int] = 2240,
        caption_channels: int = 2304,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
        attention_bias: bool = False,
        sample_size: int = 32,
        patch_size: int = 1,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: Optional[int] = None,
        guidance_embeds: bool = False,
        guidance_embeds_scale: float = 0.1,
        qk_norm: Optional[str] = None,
        timestep_scale: float = 1.0,
    ):
        super().__init__(
            in_channels,
            out_channels,
            num_attention_heads,
            attention_head_dim,
            num_layers,
            num_cross_attention_heads,
            cross_attention_head_dim,
            cross_attention_dim,
            caption_channels,
            mlp_ratio,
            dropout,
            attention_bias,
            sample_size,
            patch_size,
            norm_elementwise_affine,
            norm_eps,
            interpolation_scale,
            guidance_embeds,
            guidance_embeds_scale,
            qk_norm,
            timestep_scale,
        )

        self.num_layers = num_layers

    def get_num_layers(self):
        return self.num_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples: Optional[Tuple[torch.Tensor]] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, ...], Transformer2DModelOutput]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        # if USE_PEFT_BACKEND:
        #     # weight the lora layers by setting `lora_scale` for each PEFT layer
        #     scale_lora_layers(self, lora_scale)
        # else:
        #     if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
        #         logger.warning(
        #             "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
        #         )

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size, num_channels, height, width = hidden_states.shape
        p = self.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        hidden_states = self.patch_embed(hidden_states)

        if guidance is not None:
            timestep, embedded_timestep = self.time_embed(
                timestep, guidance=guidance, hidden_dtype=hidden_states.dtype
            )
        else:
            timestep, embedded_timestep = self.time_embed(
                timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )

        # encoder_num_layers, encoder_batch_size, encoder_seq_len, encoder_hs_dim = encoder_hidden_states.shape

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        # encoder_hidden_states = encoder_hidden_states.view(batch_size, encoder_num_layers, -1, hidden_states.shape[-1])

        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        # 2. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for index_block, (block, encoder_hs) in enumerate(zip(self.transformer_blocks, encoder_hidden_states)):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hs,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

        else:
            for index_block, (block, encoder_hs) in enumerate(zip(self.transformer_blocks, encoder_hidden_states)):
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    encoder_hs,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

        # 3. Normalization
        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)

        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_height, post_patch_width, self.config.patch_size, self.config.patch_size, -1
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)

        # if USE_PEFT_BACKEND:
        #     # remove `lora_scale` from each PEFT layer
        #     unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

class MLLMInContextConfig(PretrainedConfig):
    model_type = "mllm-in-context"

    def __init__(
        self,
        mllm_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        diffusion_model_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        in_channels: int = 32,
        input_size: int = 32,
        num_metaqueries: int = 64,
        _gradient_checkpointing: bool = True,
        max_input_text_tokens: int = 256,
        connector_num_hidden_layers: int = 24,
        system_prompt: str = "You will be given an image or its caption. Please describe the content of the image in detail in your own words.",
        **kwargs,
    ):
        super().__init__()
        self.mllm_id = mllm_id
        self.diffusion_model_id = diffusion_model_id
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_metaqueries = num_metaqueries
        self._gradient_checkpointing = _gradient_checkpointing
        self.max_input_text_tokens = max_input_text_tokens
        self.connector_num_hidden_layers = connector_num_hidden_layers
        self.system_prompt = system_prompt


class MLLMInContext(PreTrainedModel):
    config_class = MLLMInContextConfig

    def __init__(
        self,
        config: MLLMInContextConfig,
    ) -> None:
        super().__init__(config)
        self._gradient_checkpointing = config._gradient_checkpointing
        self.config = config
        if "Qwen2.5-VL" in config.mllm_id:
            self.mllm_type = "qwenvl"
        elif "Qwen" in config.mllm_id:
            self.mllm_type = "qwenlm"
        elif "Llama" in config.mllm_id:
            self.mllm_type = "llamaml"
        else:
            self.mllm_type = "llavaov"

        if self.mllm_type == "llavaov":
            self.mllm_backbone = LlavaOnevisionForConditionalGeneration.from_pretrained(
                config.mllm_id, attn_implementation="sdpa"
            )
            self.mllm_backbone.language_model.config.use_sliding_window = False
            self.mllm_backbone.language_model.config.sliding_window = None
            num_embeddings = self.mllm_backbone.get_input_embeddings().num_embeddings
            self.num_embeddings = num_embeddings
            if config.num_metaqueries > 0:
                try:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2
                    )
                except:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2, mean_resizing=False
                    )

            def freeze_hook(grad):
                grad[: self.num_embeddings].zero_()
                return grad

            self.mllm_backbone.language_model.model.embed_tokens.weight.register_hook(
                freeze_hook
            )
            self.mllm_hidden_size = self.mllm_backbone.config.text_config.hidden_size
            self.mllm_backbone.language_model.lm_head = nn.Identity()

            self.tokenizer = AutoProcessor.from_pretrained(config.mllm_id)
            self.tokenizer.tokenizer.padding_side = "left"
            self.tokenizer.resize_fn = v2.Compose([v2.Resize(384), v2.CenterCrop(384)])
            # 0.5B 896

        elif self.mllm_type == "qwenvl":
            self.mllm_backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.mllm_id, attn_implementation="sdpa"
            )
            self.mllm_backbone.model.config.use_sliding_window = False
            self.mllm_backbone.model.config.sliding_window = None
            num_embeddings = self.mllm_backbone.get_input_embeddings().num_embeddings
            self.num_embeddings = num_embeddings
            if config.num_metaqueries > 0:
                try:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2
                    )
                except:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries + 2, mean_resizing=False
                    )

            def freeze_hook(grad):
                grad[: self.num_embeddings].zero_()
                return grad

            self.mllm_backbone.model.embed_tokens.weight.register_hook(freeze_hook)
            self.mllm_hidden_size = self.mllm_backbone.config.hidden_size
            self.mllm_backbone.lm_head = nn.Identity()

            min_pixels = 256 * 28 * 28
            max_pixels = 256 * 28 * 28
            self.tokenizer = AutoProcessor.from_pretrained(
                config.mllm_id, min_pixels=min_pixels, max_pixels=max_pixels
            )
            self.tokenizer.tokenizer.padding_side = "left"
            self.tokenizer.resize_fn = None
            # 3B 2048
            # 7B 3584

        elif self.mllm_type == "qwenlm":
            self.mllm_backbone = AutoModelForCausalLM.from_pretrained(
                config.mllm_id, attn_implementation="sdpa"
            )
            self.mllm_backbone.model.config.use_sliding_window = False
            self.mllm_backbone.model.config.sliding_window = None
            num_embeddings = self.mllm_backbone.get_input_embeddings().num_embeddings
            self.num_embeddings = num_embeddings
            if config.num_metaqueries > 0:
                try:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries
                    )
                except:
                    self.mllm_backbone.resize_token_embeddings(
                        num_embeddings + config.num_metaqueries, mean_resizing=False
                    )

            def freeze_hook(grad):
                grad[: self.num_embeddings].zero_()
                return grad

            self.mllm_backbone.model.embed_tokens.weight.register_hook(freeze_hook)
            self.mllm_hidden_size = self.mllm_backbone.config.hidden_size
            self.mllm_backbone.lm_head = nn.Identity()

            self.tokenizer = AutoProcessor.from_pretrained(config.mllm_id)
            self.tokenizer.padding_side = "left"
            self.tokenizer.resize_fn = None
        else:
            raise ValueError(f"Unsupported model: {config.mllm_id}")

        self.tokenizer.mllm_type = self.mllm_type
        self.tokenizer.max_input_text_tokens = config.max_input_text_tokens
        self.tokenizer.num_metaqueries = config.num_metaqueries
        self.tokenizer.system_prompt = config.system_prompt
        self.pad_token_id = getattr(
            self.tokenizer, "tokenizer", self.tokenizer
        ).pad_token_id
        if config.num_metaqueries > 0:
            tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        f"<|pad_token_{i}|>"
                        for i in range(num_embeddings - len(tokenizer))
                    ]
                }
            )
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [f"<|img{i}|>" for i in range(self.tokenizer.num_metaqueries)]
                }
            )
            self.boi_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
            self.eoi_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

        if "Sana" in config.diffusion_model_id:
            self.transformer = LayerCondSanaTransformer2DModel.from_pretrained(
                config.diffusion_model_id,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            )
            # self.transformer.caption_projection = nn.Identity()
            input_scale = math.sqrt(5.5)
            # 2304 --> 2240

        elif "stable-diffusion-v1-5" in config.diffusion_model_id:
            self.transformer = UNet2DConditionModel.from_pretrained(
                config.diffusion_model_id, subfolder="unet"
            )
            input_scale = 1
            # 768

        else:
            raise ValueError(f"Unsupported model: {config.diffusion_model_id}")

        self.connector_in_dim = self.mllm_hidden_size
        self.connector_out_dim = (
            getattr(self.transformer.config, "caption_channels", None)
            or getattr(self.transformer.config, "encoder_hid_dim", None)
            or getattr(self.transformer.config, "cross_attention_dim", None)
        )

        norm = RMSNorm(self.connector_out_dim, eps=1e-5, elementwise_affine=True)
        with torch.no_grad():
            norm.weight.fill_(input_scale)

        self.connector = nn.Sequential(
            nn.Linear(self.connector_in_dim, self.connector_out_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.connector_out_dim, self.connector_out_dim),
            norm,
        )

        if config._gradient_checkpointing:
            try:
                self.mllm_backbone.gradient_checkpointing_enable(
                    {"use_reentrant": False}
                )
            except:
                pass
            if not isinstance(self.connector, nn.Identity):
                for module in self.connector:
                    if isinstance(module, Qwen2Encoder):
                        module.gradient_checkpointing_enable({"use_reentrant": False})
            self.transformer.enable_gradient_checkpointing()

    def get_tokenizer(self):
        return self.tokenizer

    def get_tokenize_fn(self):
        return self.tokenize

    def get_resize_fn(self):
        return self.resize_fn

    @staticmethod
    @torch.no_grad()
    def tokenize(
        tokenizer, caption, image=None, text_response=None, add_generation_prompt=True
    ):
        if not isinstance(caption, List):
            caption = [caption]

        prefix = (
            [
                {
                    "role": "system",
                    "content": (
                        tokenizer.system_prompt
                        if tokenizer.mllm_type == "qwenlm"
                        else [{"type": "text", "text": tokenizer.system_prompt}]
                    ),
                },
            ]
            if tokenizer.system_prompt is not None
            else []
        )

        if not add_generation_prompt or tokenizer.num_metaqueries <= 0:
            suffix = ""
        else:
            suffix = (
                "<|vision_start|>"
                + "".join([f"<|img{i}|>" for i in range(tokenizer.num_metaqueries)])
                + "<|vision_end|><|im_end|>"
            )

        caption = [
            tokenizer.decode(
                tokenizer(text=cap, return_tensors="pt", padding=False).input_ids[
                    0, : tokenizer.max_input_text_tokens
                ]
            )
            for cap in caption
        ]

        if isinstance(image, list) and all([img is None for img in image]):
            print("find a pure text batch!!!")
            image = None

        if image is not None:
            # If image is not a list, wrap it in a list
            if not isinstance(image, list):
                image = [image]
            # If each batch item is not a list, wrap it in a single-element list (or empty list if None)
            for i, img in enumerate(image):
                if img and not isinstance(img, list):
                    image[i] = [img]

            # Resize each image in each batch if resize_fn is not None
            if tokenizer.resize_fn is not None:
                image = [
                    [tokenizer.resize_fn(sub_img) for sub_img in imgs] if imgs else None
                    for imgs in image
                ]

            conversations = [
                prefix
                + [
                    {
                        "role": "user",
                        "content": (
                            [{"type": "image"} for _ in imgs]
                            + [{"type": "text", "text": cap}]
                            if imgs
                            else [{"type": "text", "text": cap}]
                        ),
                    },
                ]
                for cap, imgs in zip(caption, image)
            ]
            kwargs = {"images": [imgs for imgs in image if imgs]}

        elif tokenizer.mllm_type in ["qwenlm", "llamaml"]:
            conversations = [
                prefix
                + [
                    {
                        "role": "user",
                        "content": cap,
                    },
                ]
                for cap in caption
            ]
            kwargs = dict()

        else:
            conversations = [
                prefix
                + [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": cap}],
                    },
                ]
                for cap in caption
            ]
            kwargs = dict()

        prompts = [
            tokenizer.apply_chat_template(conv, add_generation_prompt=True)
            for conv in conversations
        ]
        if text_response is not None:
            prompts = [p + t.strip() for p, t in zip(prompts, text_response)]
        if tokenizer.num_metaqueries > 0:
            prompts = [p + suffix for p in prompts]
        
        text_inputs = tokenizer(
            text=prompts,
            return_tensors="pt",
            padding=True,
            **kwargs,
        )

        if tokenizer.mllm_type == "qwenvl" and "pixel_values" in text_inputs:
            text_inputs["pixel_values"] = text_inputs["pixel_values"].unsqueeze(0)

        return text_inputs.values()

    def find_target_pos(self, groups):
        group_changes = groups[:-1] != groups[1:]
        last_indices_except_last_group = torch.where(group_changes)[0]
        last_index_of_last_group = torch.tensor([len(groups) - 1], device=groups.device)
        final_indices = torch.cat([last_indices_except_last_group, last_index_of_last_group])

        return final_indices

    def encode_condition(
        self, input_ids, attention_mask, pixel_values, image_sizes, **kwargs
    ):
        if self.mllm_type == "llavaov":
            prompt_embeds = self.mllm_backbone(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
            ).logits
        elif self.mllm_type == "qwenvl":
            prompt_embeds = self.mllm_backbone(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_sizes,
                attention_mask=attention_mask,
                output_hidden_states=True
            ).hidden_states

            prompt_embeds = torch.stack(prompt_embeds, dim=0)

        elif self.mllm_type in ["qwenlm", "llamaml"]:
            prompt_embeds = self.mllm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
        else:
            raise ValueError(f"Unsupported model: {self.mllm_type}")

        if self.tokenizer.num_metaqueries > 0:
            # Get positions for all sequences in batch at once
            boi_indices, boi_pos = torch.where(input_ids == self.boi_token_id)
            eoi_indeices, eoi_pos = torch.where(input_ids == self.eoi_token_id)

            # Create mask for selecting tokens between BOI and EOI
            batch_size, seq_len = input_ids.shape
            
            boi_pos = boi_pos[self.find_target_pos(boi_indices)]
            eoi_pos = eoi_pos[self.find_target_pos(eoi_indeices)]

            indices = torch.arange(seq_len, device=input_ids.device)[None, :].expand(
                batch_size, -1
            )
            mask = (indices > boi_pos[:, None]) & (indices < eoi_pos[:, None])
            prompt_embed_mask = mask[None].expand(prompt_embeds.size(0), -1, -1)

            prompt_embeds = prompt_embeds[prompt_embed_mask].view(
                prompt_embeds.size(0), batch_size, -1, prompt_embeds.size(-1)
            )

            attention_mask = attention_mask[mask].view(batch_size, -1)

            prompt_embeds = prompt_embeds[-self.transformer.get_num_layers() - 1: -1]

        return self.connector(prompt_embeds), attention_mask

    def forward(self, x, timestep, prompt_embeds=None, attention_mask=None):
        if isinstance(self.transformer, LayerCondSanaTransformer2DModel):
            model_pred = self.transformer(
                hidden_states=x,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=attention_mask,
            ).sample
            return model_pred
        elif isinstance(self.transformer, UNet2DConditionModel):
            model_pred = self.transformer(
                sample=x,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
            ).sample
            return model_pred
        else:
            raise ValueError(
                f"Unsupported model: {self.transformer.__class__.__name__}"
            )
