"defining convolutional autoencoder"
import torch
from torch import nn
import torch.nn.functional as F


class ConvAE(nn.Module):
    """convolutional autoencoder with Conv2d, ConvTranspose2d"""

    def __init__(self):
        """initialize autoencoder w/ two hidden layers
        encoder & decoder have symmetric architecture
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=5)
        self.deconv1 = \
            nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5)
        self.deconv2 = \
            nn.ConvTranspose2d(in_channels=5, out_channels=3, kernel_size=5)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """encode process saves MaxPooling indicies at stack; self.indicies

        Args:
            x (torch.Tensor): input image

        Returns:
            torch.Tensor: latent tensor
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """recover original image from latent vector

        Args:
            x (torch.Tensor): latent vector

        Returns:
            torch.Tensor: recovered image
        """
        x = self.deconv2(x)
        x = F.relu(x)
        x = self.deconv1(x)
        x = F.relu(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """pass through encoder, decoder sequentially

        Args:
            x (torch.Tensor): input image

        Returns:
            _type_: torch.Tensor
        """
        x = self.encode(x)
        x = self.decode(x)
        return x
