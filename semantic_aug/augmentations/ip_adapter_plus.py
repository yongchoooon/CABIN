from semantic_aug.generative_augmentation import GenerativeAugmentation

from diffusers import AutoPipelineForText2Image
from diffusers.utils import logging
from PIL import Image

from typing import Tuple

import os
import torch
import random
import json

DEFAULT_MODEL_PATH = "stable-diffusion-v1-5/stable-diffusion-v1-5"

DEFAULT_SD15_IP_PLUS_SUBFOLDER = "models"
DEFAULT_SD15_IP_PLUS_WEIGHT_NAME = "ip-adapter-plus_sd15.bin"

class CABINAugmentation(GenerativeAugmentation):

    def __init__(self,
                 model_path: str = DEFAULT_MODEL_PATH,
                 guidance_scale: float = 7.5,
                 num_inference_steps: int = 30,
                 device: str = 'cuda:0',
                 examples_per_class: int = None,
                 dataset: str = None,
                 prompt_templates: str = "prompt_generation/prompt_templates",
                 **kwargs):
        super(CABINAugmentation, self).__init__()

        self.device = device
        self.dataset = dataset

        with open(os.path.join(prompt_templates, dataset + ".json"), "r") as f:
            self.template = json.load(f)

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_path,
            torch_dtype = torch.float16
        ).to(self.device)
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter", 
            subfolder=DEFAULT_SD15_IP_PLUS_SUBFOLDER, 
            weight_name=DEFAULT_SD15_IP_PLUS_WEIGHT_NAME
        )
        self.pipe.requires_safety_checker = False
        self.pipe.safety_checker = None

        logging.disable_progress_bar()
        self.pipe.set_progress_bar_config(disable=True)

        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.examples_per_class = examples_per_class


    def forward(self, image: Image.Image, label: int, metadata: dict, num: int, phase_name: str = None, **kwargs) -> Tuple[Image.Image, int]:

        class_name = metadata.get("name", "")
        class_template = self.template[class_name]
        style_image = image.resize((256, 256))
        
        scale = self.get_scale_by_phase_name(num, phase_name)
        prompt = self.get_text_prompt(self.dataset, class_name, class_template, scale=scale)

        scale_dict = self.get_scale_dict(scale)
        self.pipe.set_ip_adapter_scale(scale_dict)

        aug_image = self.pipe(
            prompt = prompt,
            ip_adapter_image = style_image,
            negative_prompt = "text, watermark, low quality, worst quality",
            guidance_scale=self.guidance_scale, 
            num_inference_steps = self.num_inference_steps,
        ).images[0]

        canvas = aug_image.resize(image.size, Image.BILINEAR)

        return canvas, label, prompt, scale
    
    def get_scale_by_phase_name(self, num: int, phase_name: str):
        if phase_name == "Phase_div" or phase_name == None:
            scale = (num // 10) / 10
        elif phase_name == "Phase_key":
            scale = (num // 10) / 10 + 0.8
        else:
            raise ValueError("Invalid stage name")

        return scale

    def get_scale_dict(self, scale):
        scale_dict = {
            "down": {"block_0": [scale], "block_1": [scale], "block_2": [scale], "block_3": [scale]},
            "up": {"block_0": [0, scale, 0], "block_1": [0, scale, 0], "block_2": [0, scale, 0], "block_3": [0, scale, 0]},
        }
        return scale_dict
    
    def get_text_prompt(self, class_name, class_template, scale=0.5):
    
        if self.dataset == 'eurosat':
            photo_type = 'satellite photo'
        else:
            photo_type = 'photo'
        
        action_or_pose = class_template['action_or_pose']
        invariant_features = class_template['invariant_features']
        cooccurrence = class_template['cooccurrence']
        background = class_template['background']
        
        extra_context = f"surrounded by {random.choice(cooccurrence)}, set against a {random.choice(background)} background. "
        
        if scale < 0.7:
            
            text_prompt = (
                f"A {photo_type} of a {random.choice(action_or_pose)} {class_name}, {', '.join(invariant_features)}. "
                f"{extra_context}"
            )
        else:
            text_prompt = (
                f"A {photo_type} of a {class_name}, {', '.join(invariant_features)}."
            )
        return text_prompt