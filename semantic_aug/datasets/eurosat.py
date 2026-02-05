from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Tuple

import numpy as np
import torchvision.transforms as transforms
import torch
import json
import os

from PIL import Image
from collections import defaultdict


DEFAULT_EUROSAT_DIR = "eurosat"

class EuroSATDataset(FewShotDataset):

    class_names = [
        'Annual Crop',
        'Forest',
        'Herbaceous Vegetation',
        'Highway',
        'Industrial',
        'Pasture',
        'Permanent Crop',
        'Residential',
        'River',
        'Sea/Lake'
    ]

    num_classes: int = len(class_names)

    def __init__(self, *arg, split: str = "train", seed: int = 0,
                 image_dir: str = DEFAULT_EUROSAT_DIR,
                 examples_per_class: int = None,
                 generative_aug: GenerativeAugmentation = None,
                 synthetic_probability: float = 0.5,
                 image_size: Tuple[int] = (64, 64), 
                 **kwargs):

        super(EuroSATDataset, self).__init__(
            *arg, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability,
            generative_aug=generative_aug, **kwargs)

        with open(os.path.join(image_dir, "split_zhou_EuroSAT.json"), "r") as f:
            labels = json.load(f)
        splited_labels = labels[split]
        
        image_files = [os.path.join(DEFAULT_EUROSAT_DIR, "2750", label[0]) for label in splited_labels]

        class_to_images = defaultdict(list)

        for image_idx, image_path in enumerate(image_files):
            class_name = self.class_names[splited_labels[image_idx][1]]
            class_to_images[class_name].append(image_path)

        rng = np.random.default_rng(seed)
        class_to_ids = {key: rng.permutation(
            len(class_to_images[key])) for key in self.class_names}
        
        if examples_per_class is not None:
            class_to_ids = {key: ids[:examples_per_class] 
                            for key, ids in class_to_ids.items()}
        
        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids] 
            for key, ids in class_to_ids.items()}
        
        self.all_images = sum([
            self.class_to_images[key] 
            for key in self.class_names], [])
        
        self.all_labels = [i for i, key in enumerate(
            self.class_names) for _ in self.class_to_images[key]]
        
        train_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15.0),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                  std=[0.26862954, 0.26130258, 0.27577711])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                  std=[0.26862954, 0.26130258, 0.27577711])
        ])

        self.transform = {"train": train_transform, "val": val_transform}[split]

    def __len__(self):
        
        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> Image.Image:

        return Image.open(self.all_images[idx]).convert('RGB')

    def get_label_by_idx(self, idx: int) -> int:

        return self.all_labels[idx]
    
    def get_metadata_by_idx(self, idx: int) -> dict:

        return dict(name=self.class_names[self.all_labels[idx]])