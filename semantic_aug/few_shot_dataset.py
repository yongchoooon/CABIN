from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Tuple
from torch.utils.data import Dataset
from collections import defaultdict
from itertools import product
from tqdm import tqdm
from PIL import Image

import torchvision.transforms as transforms
import torch
import numpy as np
import abc
import random
import os
import csv

class FewShotDataset(Dataset):

    num_classes: int = None
    class_names: int = None

    def __init__(self, examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 synthetic_dir: str = None,
                 ):

        self.examples_per_class = examples_per_class
        self.generative_aug = generative_aug
        self.only_real = False

        self.synthetic_probability = synthetic_probability
        self.synthetic_dir = synthetic_dir
        self.synthetic_examples = defaultdict(list)
        self.source_csv_path = synthetic_dir+'/source_image_prompt.csv'

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5]),
        ])
        
        if synthetic_dir is not None:
            os.makedirs(synthetic_dir, exist_ok=True)
        try:
            if os.path.exists(self.source_csv_path) == False:
                with open(self.source_csv_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['source_image_path', 'aug_image_path', 'used_prompt', 'scale'])
        except:
            pass
    
    @abc.abstractmethod
    def get_image_by_idx(self, idx: int) -> Image.Image:

        return NotImplemented
    
    @abc.abstractmethod
    def get_label_by_idx(self, idx: int) -> int:

        return NotImplemented
    
    @abc.abstractmethod
    def get_metadata_by_idx(self, idx: int) -> dict:

        return NotImplemented

    def generate_augmentations(self, num_repeats: int, phase_name: str = None):

        self.synthetic_examples.clear()
        options = product(range(len(self)), range(num_repeats))
        pbar = tqdm(list(options), desc="Generating Augmentations")
        for idx, num in pbar:

            image_ori = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)

            aug_image_path = os.path.join(self.synthetic_dir, f"aug-{idx}-{num}.png")

            image_gen, label, used_prompt, scale = self.generative_aug(
                    image_ori, label, self.get_metadata_by_idx(idx), num, phase_name)
            
            with open(self.source_csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.all_images[idx], aug_image_path, used_prompt, scale])

            if self.synthetic_dir is not None:
                pil_image_gen = image_gen

                pil_image_gen.save(aug_image_path)

            self.synthetic_examples[idx].append((aug_image_path, label))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        if self.only_real:
            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)

            return self.transform(image), label
        else:
            if len(self.synthetic_examples[idx]) > 0 and \
                    np.random.uniform() < self.synthetic_probability:

                image, label = random.choice(self.synthetic_examples[idx])
                if isinstance(image, str): image = Image.open(image)
            
                return self.transform(image), label

        image = self.get_image_by_idx(idx)
        label = self.get_label_by_idx(idx)

        return self.transform(image), label
