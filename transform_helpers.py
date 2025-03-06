#!/usr/bin/env python3

import torchvision.transforms as transforms

augment1 = transforms.Compose([
    #transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomCrop(32, padding=4),  # Randomly crop with padding
    transforms.Resize(32),  # Resize to the desired size
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    #transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

augment2 = transforms.Compose([
    #transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomCrop(32, padding=4),  # Randomly crop with padding
    transforms.Resize(32),  # Resize to the desired size
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.GaussianBlur(kernel_size=3),
    #transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

augment3 = transforms.Compose([
    #transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
    transforms.RandomCrop(32, padding=4),  # Randomly crop with padding
    transforms.Resize(32),  # Resize to the desired size
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.RandomRotation(degrees=30),
    transforms.GaussianBlur(kernel_size=3),
    #transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])