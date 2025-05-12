#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define Albumentations augmentations
augmentation1 = A.Compose([
    # Inception-style cropping: random crop, flip, and resize to 32x32
    A.RandomResizedCrop((32, 32), scale=(0.08, 1.0)),
    A.HorizontalFlip(),
    A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

augmentation2 = A.Compose([
    A.RandomResizedCrop((32, 32), scale=(0.5, 1.0)),
    A.HorizontalFlip(),
    A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    A.GaussianBlur(blur_limit=3),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

augmentation2 = transforms.Compose([
    #transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomCrop(32, padding=4),  # Randomly crop with padding
    transforms.Resize(32),  # Resize to the desired size
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    #transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
