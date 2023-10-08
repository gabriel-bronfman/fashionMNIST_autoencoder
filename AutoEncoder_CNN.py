import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
 
# Define the autoencoder architecture
class AE_CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Building an CNN encoder with GeLu activation layers
        # 1x28x28 ==> 8x4x4
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 3, stride=3, padding=1),  # b, 16, 10, 10
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            torch.nn.Conv2d(4, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

        # Building an CNN decoder with GeLu activation layers
        # 8x4x4 ==> 1x28x28
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(8, 8, 3, stride=2),  # b, 8, 4,4
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(8, 4, 5, stride=3, padding=1),  # b, 4, 15, 15
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(4, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
