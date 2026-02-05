from semantic_aug.datasets.caltech101 import CalTech101Dataset
from semantic_aug.datasets.flowers102 import Flowers102Dataset
from semantic_aug.datasets.pets import PetsDataset
from semantic_aug.datasets.food import FoodDataset
from semantic_aug.datasets.cub import CUBDataset
from semantic_aug.datasets.eurosat import EuroSATDataset
from semantic_aug.augmentations.ip_adapter_plus import CABINAugmentation

from semantic_aug.classifier import ClassificationModel
from semantic_aug.loss import LabelSmoothLoss

from transformers import get_cosine_schedule_with_warmup

from torch.utils.data import DataLoader
from itertools import product
from tqdm import trange

import torch
import torch.nn.functional as F
import random
import numpy as np
import os
import copy
import pandas as pd

import argparse

import importlib.util

os.environ["TOKENIZERS_PARALLELISM"] = "true"

DATASETS = {
    "caltech": CalTech101Dataset,
    "flowers": Flowers102Dataset,
    "pets": PetsDataset,
    "food": FoodDataset,
    "cub": CUBDataset,
    "eurosat": EuroSATDataset,
}

AUGMENTATIONS = {
    "cabin": CABINAugmentation,
}

def run_experiment_with_1_and_2(examples_per_class: int = 0, 
                                seed: int = 0, 
                                dataset: str = "pets", 
                                iterations_per_epoch: int = 200,
                                num_epochs_1: int = 50, 
                                num_epochs_2: int = 50, 
                                batch_size: int = 32, 
                                aug: str = None, 
                                guidance_scale: float = None, 
                                model_path: str = None, 
                                image_size: int = 256, 
                                classifier_backbone: str = "resnet50", 
                                num_inference_steps: int = 30, 
                                num_workers: int = 4, 
                                phase_name_1: str = None,
                                phase_name_2: str = None,
                                synthetic_probability_1: float = 1.0,
                                synthetic_probability_2: float = 0.5,
                                synthetic_dir_1: str = None,
                                synthetic_dir_2: str = None,
                                num_synthetic_1: int = 0,
                                num_synthetic_2: int = 0,
                                lr: float = 0.00001,
                                prompt_templates: str = None,
                                device: str = 'cuda:0', 
):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if aug is not None:
        aug = AUGMENTATIONS[aug](
            model_path=model_path,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            device=device,
            examples_per_class=examples_per_class,
            dataset=dataset,
            prompt_templates=prompt_templates,
        )

    train_dataset_1 = DATASETS[dataset](
        split="train", examples_per_class=examples_per_class, 
        synthetic_probability=synthetic_probability_1,
        synthetic_dir=synthetic_dir_1,
        generative_aug=aug, seed=seed,
        image_size=(image_size, image_size))
    
    train_dataset_2 = DATASETS[dataset](
        split="train", examples_per_class=examples_per_class, 
        synthetic_probability=synthetic_probability_2,
        synthetic_dir=synthetic_dir_2,
        generative_aug=aug, seed=seed,
        image_size=(image_size, image_size))

    train_dataset_1.generate_augmentations(num_synthetic_1, phase_name = phase_name_1)
    train_dataset_2.generate_augmentations(num_synthetic_2, phase_name = phase_name_2)

    val_dataset = DATASETS[dataset](
        split="val", seed=seed, 
        image_size=(image_size, image_size))

    train_sampler_1 = torch.utils.data.RandomSampler(
        train_dataset_1, replacement=True,
        num_samples=batch_size * iterations_per_epoch)
    
    train_dataloader_1 = DataLoader(
        train_dataset_1, batch_size=batch_size,
        sampler = train_sampler_1, num_workers=num_workers)
    
    train_sampler_2 = torch.utils.data.RandomSampler(
        train_dataset_2, replacement=True,
        num_samples=batch_size * iterations_per_epoch)

    train_dataloader_2 = DataLoader(
        train_dataset_2, batch_size=batch_size,
        sampler = train_sampler_2, num_workers=num_workers)
    
    val_sampler = torch.utils.data.RandomSampler(
        val_dataset, replacement=True, 
        num_samples=batch_size * iterations_per_epoch)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, 
        sampler=val_sampler, num_workers=num_workers)

    model = ClassificationModel(
        train_dataset_1.num_classes,
        backbone=classifier_backbone,
        train_dataset=train_dataset_1,
        device=device,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    total_steps_each_phase = num_epochs_1 * len(train_dataloader_1)
    warmup_steps = int(0.1 * total_steps_each_phase)

    scheduler_1 = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps_each_phase)
    scheduler_2 = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps_each_phase)

    best_model_state_1 = None
    best_training_accuracy = 0
    best_epoch_1 = 0

    records = []

    for epoch in trange(num_epochs_1, desc=f"Training on Phase 1 Images [{phase_name_1}]"):
        model.train()

        epoch_loss_1 = torch.zeros(
            train_dataset_1.num_classes, 
            dtype=torch.float32, device=device)
        epoch_accuracy_1 = torch.zeros(
            train_dataset_1.num_classes, 
            dtype=torch.float32, device=device)
        epoch_size_1 = torch.zeros(
            train_dataset_1.num_classes, 
            dtype=torch.float32, device=device)
        
        train_dataloader_1_only_real = copy.deepcopy(train_dataloader_1)
        train_dataloader_1_only_real.dataset.only_real = True

        for i, ((images, labels), (images_real, labels_real)) in enumerate(zip(train_dataloader_1, train_dataloader_1_only_real)):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            prediction = logits.argmax(dim=1)

            criterion = LabelSmoothLoss(train_dataset_1.num_classes, smoothing = 0.7, reduction = 'none')
            loss = criterion(logits, labels)

            if len(labels.shape) > 1: labels = labels.argmax(dim=1)

            accuracy = (prediction == labels).float()

            optim.zero_grad()
            loss.mean().backward()
            optim.step()

            scheduler_1.step()

            images_real, labels_real = images_real.to(device), labels_real.to(device)

            logits_real = model(images_real)
            prediction_real = logits_real.argmax(dim=1)

            loss_real = F.cross_entropy(logits_real, labels_real, reduction="none")
            if len(labels_real.shape) > 1: labels_real = labels_real.argmax(dim=1)

            accuracy_real = (prediction_real == labels_real).float()

            with torch.no_grad():
            
                epoch_size_1.scatter_add_(0, labels_real, torch.ones_like(loss))
                epoch_loss_1.scatter_add_(0, labels_real, loss_real)
                epoch_accuracy_1.scatter_add_(0, labels_real, accuracy_real)

        training_loss_1 = epoch_loss_1 / epoch_size_1.clamp(min=1)
        training_accuracy_1 = epoch_accuracy_1 / epoch_size_1.clamp(min=1)

        training_loss_1 = training_loss_1.cpu().numpy()
        training_accuracy_1 = training_accuracy_1.cpu().numpy()

        model.eval()

        epoch_loss_1 = torch.zeros(
            train_dataset_1.num_classes, 
            dtype=torch.float32, device=device)
        epoch_accuracy_1 = torch.zeros(
            train_dataset_1.num_classes, 
            dtype=torch.float32, device=device)
        epoch_size_1 = torch.zeros(
            train_dataset_1.num_classes, 
            dtype=torch.float32, device=device)

        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)

                logits = model(images)
                prediction = logits.argmax(dim=1)

                loss = F.cross_entropy(logits, labels, reduction="none") # Just for logging
                accuracy = (prediction == labels).float()

                with torch.no_grad():
            
                    epoch_size_1.scatter_add_(0, labels, torch.ones_like(loss))
                    epoch_loss_1.scatter_add_(0, labels, loss)
                    epoch_accuracy_1.scatter_add_(0, labels, accuracy)

        validation_loss_1 = epoch_loss_1 / epoch_size_1.clamp(min=1)
        validation_accuracy_1 = epoch_accuracy_1 / epoch_size_1.clamp(min=1)

        validation_loss_1 = validation_loss_1.cpu().numpy()
        validation_accuracy_1 = validation_accuracy_1.cpu().numpy()

        if training_accuracy_1.mean() > best_training_accuracy:
            best_training_accuracy = training_accuracy_1.mean()
            best_model_state_1 = copy.deepcopy(model.state_dict())
            best_epoch_1 = epoch

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=training_loss_1.mean(), 
            metric="Loss", 
            split="Training",
            type=phase_name_1
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=validation_loss_1.mean(), 
            metric="Loss", 
            split="Validation",
            type=phase_name_1
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=training_accuracy_1.mean(), 
            metric="Accuracy", 
            split="Training",
            type=phase_name_1
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=validation_accuracy_1.mean(), 
            metric="Accuracy", 
            split="Validation",
            type=phase_name_1
        ))

        for i, name in enumerate(train_dataset_1.class_names):

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=training_loss_1[i], 
                metric=f"Loss {name.title()}", 
                split="Training",
                type=phase_name_1
            ))

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=validation_loss_1[i], 
                metric=f"Loss {name.title()}", 
                split="Validation",
                type=phase_name_1
            ))

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=training_accuracy_1[i], 
                metric=f"Accuracy {name.title()}", 
                split="Training",
                type=phase_name_1
            ))

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=validation_accuracy_1[i], 
                metric=f"Accuracy {name.title()}", 
                split="Validation",
                type=phase_name_1
            ))

    print("====================================================")
    print(f"Best Training Accuracy on synthetic images: ", best_training_accuracy)
    print("Epoch: ", best_epoch_1)
    print("====================================================")

    model.load_state_dict(best_model_state_1)

    for epoch in trange(num_epochs_2, desc=f"Training on Phase 2 Images [{phase_name_2}]"):
        model.train()

        epoch_loss_2 = torch.zeros(
            train_dataset_2.num_classes, 
            dtype=torch.float32, device=device)
        epoch_accuracy_2 = torch.zeros(
            train_dataset_2.num_classes, 
            dtype=torch.float32, device=device)
        epoch_size_2 = torch.zeros(
            train_dataset_2.num_classes, 
            dtype=torch.float32, device=device)

        for i, (images, labels) in enumerate(train_dataloader_2):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            prediction = logits.argmax(dim=1)

            criterion = LabelSmoothLoss(train_dataset_1.num_classes, smoothing = 0.6, reduction = 'none')
            loss = criterion(logits, labels)

            if len(labels.shape) > 1: labels = labels.argmax(dim=1)

            accuracy = (prediction == labels).float()

            optim.zero_grad()
            loss.mean().backward()
            optim.step()

            scheduler_2.step()

            with torch.no_grad():
            
                epoch_size_2.scatter_add_(0, labels, torch.ones_like(loss))
                epoch_loss_2.scatter_add_(0, labels, loss)
                epoch_accuracy_2.scatter_add_(0, labels, accuracy)

        training_loss_2 = epoch_loss_2 / epoch_size_2.clamp(min=1)
        training_accuracy_2 = epoch_accuracy_2 / epoch_size_2.clamp(min=1)

        training_loss_2 = training_loss_2.cpu().numpy()
        training_accuracy_2 = training_accuracy_2.cpu().numpy()

        model.eval()

        epoch_loss_2 = torch.zeros(
            train_dataset_2.num_classes,
            dtype=torch.float32, device=device)
        epoch_accuracy_2 = torch.zeros(
            train_dataset_2.num_classes,
            dtype=torch.float32, device=device)
        epoch_size_2 = torch.zeros(
            train_dataset_2.num_classes,
            dtype=torch.float32, device=device)

        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)

                logits = model(images)
                prediction = logits.argmax(dim=1)

                loss = F.cross_entropy(logits, labels, reduction="none") # Just for logging
                accuracy = (prediction == labels).float()

                with torch.no_grad():
                        
                    epoch_size_2.scatter_add_(0, labels, torch.ones_like(loss))
                    epoch_loss_2.scatter_add_(0, labels, loss)
                    epoch_accuracy_2.scatter_add_(0, labels, accuracy)

        validation_loss_2 = epoch_loss_2 / epoch_size_2.clamp(min=1)
        validation_accuracy_2 = epoch_accuracy_2 / epoch_size_2.clamp(min=1)

        validation_loss_2 = validation_loss_2.cpu().numpy()
        validation_accuracy_2 = validation_accuracy_2.cpu().numpy()

        if training_accuracy_2.mean() > best_training_accuracy:
            best_training_accuracy = training_accuracy_2.mean()

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=training_loss_2.mean(), 
            metric="Loss", 
            split="Training",
            type=phase_name_2
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=validation_loss_2.mean(), 
            metric="Loss", 
            split="Validation",
            type=phase_name_2
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=training_accuracy_2.mean(), 
            metric="Accuracy", 
            split="Training",
            type=phase_name_2
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=validation_accuracy_2.mean(), 
            metric="Accuracy", 
            split="Validation",
            type=phase_name_2
        ))

        for i, name in enumerate(train_dataset_2.class_names):

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=training_loss_2[i], 
                metric=f"Loss {name.title()}", 
                split="Training",
                type=phase_name_2
            ))

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=validation_loss_2[i], 
                metric=f"Loss {name.title()}", 
                split="Validation",
                type=phase_name_2
            ))

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=training_accuracy_2[i], 
                metric=f"Accuracy {name.title()}", 
                split="Training",
                type=phase_name_2
            ))

            records.append(dict(
                seed=seed, 
                examples_per_class=examples_per_class,
                epoch=epoch, 
                value=validation_accuracy_2[i], 
                metric=f"Accuracy {name.title()}", 
                split="Validation",
                type=phase_name_2
            ))

    return records


if __name__ == "__main__":

    parser = argparse.ArgumentParser("One-Shot Baseline")

    parser.add_argument("--config", type=str, required=True, default=None)
    
    args = parser.parse_args()

    config_name = args.config
    config_file_path = os.path.join("configs", f"{config_name}.py")

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file {config_file_path} not found")
    
    module_name = os.path.splitext(os.path.basename(config_file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, config_file_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    device = getattr(cfg, 'device', 'cuda')
    device_num = getattr(cfg, 'device_num', 0)
    device = f'{device}:{device_num}'
    
    torch.cuda.set_device(device_num)

    os.makedirs(cfg.logdir, exist_ok=True)

    all_trials = []

    options = product(range(cfg.num_trials), cfg.examples_per_class)
    options = np.array(list(options))

    option_list = options.tolist()

    for seed, examples_per_class in option_list:

        hyperparameters = dict(
            examples_per_class=examples_per_class,
            seed=seed, 
            dataset=cfg.dataset,
            num_epochs_1=cfg.num_epochs_1,
            num_epochs_2=cfg.num_epochs_2,
            iterations_per_epoch=cfg.iterations_per_epoch, 
            batch_size=cfg.batch_size,
            model_path=cfg.model_path,
            aug=cfg.aug,
            strength=cfg.strength, 
            guidance_scale=cfg.guidance_scale,
            probs=cfg.probs,
            compose=cfg.compose,
            image_size=cfg.image_size,
            classifier_backbone=cfg.classifier_backbone,
            num_inference_steps=cfg.num_inference_steps,
            num_workers=cfg.num_workers,
            phase_name_1=cfg.phase_name_1,
            phase_name_2=cfg.phase_name_2,
            synthetic_probability_1=cfg.synthetic_probability_1,
            synthetic_probability_2=cfg.synthetic_probability_2,
            num_synthetic_1=cfg.num_synthetic_1,
            num_synthetic_2=cfg.num_synthetic_2,
            lr=cfg.lr,
            prompt_templates=cfg.prompt_templates,
            device=device,
            )
        synthetic_dir_1 = cfg.synthetic_dir_1.format(**hyperparameters)
        synthetic_dir_2 = cfg.synthetic_dir_2.format(**hyperparameters)

        all_trials.extend(run_experiment_with_1_and_2(
            synthetic_dir_1=synthetic_dir_1,
            synthetic_dir_2=synthetic_dir_2,
            **hyperparameters))

        path = f"results_{seed}_{examples_per_class}.csv"
        path = os.path.join(cfg.logdir, path)

        pd.DataFrame.from_records(all_trials).to_csv(path)

        print(f"n={examples_per_class} saved to: {path}")