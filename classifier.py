import os
import torch
from torch import nn
from torch import inference_mode
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchinfo import summary
import random
from PIL import Image
import glob
from pathlib import Path
import numpy
import matplotlib.pyplot as pyplot
import seaborn
import time
import torchinfo
import classifier as mm
from tqdm.auto import tqdm
from timeit import default_timer as timer

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(64, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )
        
        # Use a placeholder for in_features that will be dynamically calculated
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1, out_features=2)  # Replace 1 with the correct value dynamically
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.classifier(x)
        return x

    def compute_linear_input_size(self, input_shape):
        # Pass a dummy tensor through the convolutional layers
        dummy_input = torch.randn(1, *input_shape)
        x = self.conv_layer_1(dummy_input)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        return x.numel()