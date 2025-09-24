# Task 1: Vision Transformers vs CNNs - Robustness Analysis

## Overview

This task implements a comprehensive comparison between Vision Transformers (ViT-B/16) and Convolutional Neural Networks (ResNet-50) on the STL-10 dataset, focusing on various robustness aspects including domain generalization, shape vs. texture bias, translation invariance, and occlusion sensitivity.

## Table of Contents

1. [Dependencies](#dependencies)
2. [Dataset](#dataset)
3. [Models](#models)
4. [Experiments](#experiments)
5. [Custom Datasets](#custom-datasets)
6. [Results](#results)
7. [Usage](#usage)

## Dependencies

### Required Libraries
```python
# Core libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torch.nn as nn

# Additional utilities
import random
from pathlib import Path
import torchvision.transforms.functional as TF
import os
import tarfile
import urllib.request
from typing import Tuple, List, Dict, Callable
from PIL import Image
from tqdm import tqdm
import csv

# Analysis tools
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
```

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 8GB RAM
- 20GB free disk space for datasets

## Datasets

### STL-10 Dataset
- **Classes**: 10 categories (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck)
- **Training images**: 5,000 (500 per class)
- **Test images**: 8,000 (800 per class)
- **Image size**: 96×96 pixels (resized to 224×224 for model input)
- **Format**: RGB images

### Stylized Dataset
Generated using AdaIN style transfer for shape vs. texture bias analysis:
- **Source**: STL-10 test set
- **Modification**: Style transfer applied while preserving shape labels
- **Purpose**: Evaluate shape bias vs. texture bias

### PACS Dataset
Used for domain generalization experiments:
- **Domains**: Photo, Art Painting, Cartoon, Sketch
- **Classes**: 7 categories (dog, elephant, giraffe, guitar, horse, house, person)
- **Purpose**: Test domain shift robustness


## Models

### ResNet-50
- **Architecture**: Deep residual network with 50 layers
- **Pre-training**: ImageNet weights
- **Fine-tuning**: Only final classification layer (10 classes)
- **Frozen layers**: All backbone parameters
- **Optimizer**: Adam (lr=0.005)

### Vision Transformer (ViT-B/16)
- **Architecture**: Base ViT with 16×16 patch size
- **Pre-training**: ImageNet weights
- **Fine-tuning**: Only classification head (10 classes)
- **Frozen layers**: All transformer blocks
- **Optimizer**: Adam (lr=0.005)

## Experiments

### 1. Standard Fine-tuning
**Objective**: Establish baseline performance on STL-10

**Process**:
- Split training set: 90% train, 10% validation
- Fine-tune classification heads only
- 5 epochs training
- Monitor training/validation loss and accuracy

**Metrics**:
- Training accuracy
- Validation accuracy
- Test accuracy

### 2. Shape vs. Texture Bias
**Objective**: Evaluate reliance on shape vs. texture cues

**Method**:
- Use stylized STL-10 images (shape preserved, texture modified)
- Compare accuracy on stylized vs. original images
- Calculate shape bias ratio

**Style Transfer**:
- AdaIN (Adaptive Instance Normalization) method
- Texture from artistic paintings
- Shape information preserved

### 3. Translation Invariance
**Objective**: Test sensitivity to spatial translations

**Test Directions**:
- 8 directions: up, down, left, right, up_left, up_right, down_left, down_right
- Translation distance: 30 pixels
- Fill modes: black (0) and mean ImageNet values

### 4. Patch-based Robustness
**Objective**: Analyze sensitivity to local perturbations

#### Patch Shuffling

#### Patch Occlusion

### 5. Domain Generalization (PACS)
**Objective**: Test robustness to domain shift

**Setup**:
- Train on 3 domains: Photo, Art Painting, Cartoon
- Test on held-out domain: Sketch
- Use same fine-tuning protocol




## Usage

### 1. Basic Setup
```python
# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Load datasets
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

stl10_train = torchvision.datasets.STL10(root='./stl_data', split='train', download=True, transform=transforms)
stl10_test = torchvision.datasets.STL10(root='./stl_data', split='test', download=True, transform=transforms)
```

### 2. Model Fine-tuning
```python
# ResNet-50
resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 10)
# Freeze backbone, train only classifier
for param in resnet.parameters():
    param.requires_grad = False
for param in resnet.fc.parameters():
    param.requires_grad = True

# ViT-B/16  
vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
vit.heads.head = torch.nn.Linear(vit.heads.head.in_features, 10)
# Freeze backbone, train only head
for param in vit.parameters():
    param.requires_grad = False
for param in vit.heads.head.parameters():
    param.requires_grad = True
```

These models were evaluated on all the modified datasets to test their OOD generalisation and their inductive biases. 