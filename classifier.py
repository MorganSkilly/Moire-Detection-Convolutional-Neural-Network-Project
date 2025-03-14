import torch
from torch import nn

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define the convolutional layers
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),  # Changed 3 to 1 for grayscale (single channel)
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

        # Use a placeholder for in_features, to be calculated dynamically
        self.classifier = None
        
    def calculate_flattened_size(self, x):
        # Pass the input through the conv layers and calculate the size
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        # Return the size of the tensor after flattening
        return x.numel()

    def forward(self, x: torch.Tensor):
        # Pass the input through the conv layers
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)

        # Flatten the tensor
        x = torch.flatten(x, 1)

        # If the classifier hasn't been initialized yet, we calculate the correct size
        if self.classifier is None:
            flattened_size = self.calculate_flattened_size(torch.randn(1, 1, 512, 512))  # Changed 3 to 1 for grayscale
            self.classifier = nn.Sequential(
                nn.Linear(flattened_size, 2)  # Set in_features based on the actual size
            )

        # Pass the flattened tensor through the classifier
        x = self.classifier(x)
        return x
