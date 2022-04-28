"defining classic autoencoder"
from typing import Dict
import torch
from torch import nn


class AutoEncoder(nn.Module):
    """classic autoencoder with FC layers only"""

    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, latent_dim: int):
        """initialize autoencoder w/ two hidden layers
        encoder & decoder have symmetric architecture

        Args:
            input_dim (int): flattened image dimension
            hidden_dim (int): # hidden nodes
            latent_dim (int): size of the bottleneck
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
            nn.ReLU()
        )

    @classmethod
    def from_config(cls, config: Dict):
        """initialize from wandb config object

        Args:
            config (Dict): dictionary with neccessary hyperparams.

        Returns:
            AutoEncoder: autoencoder based on given config
        """
        return cls(
            input_dim=784,  # 1*28*28 hardcoded for MNIST
            hidden_dim1=config.hidden_dim1,
            hidden_dim2=config.hidden_dim2,
            latent_dim=config.latent_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """pass through encoder, decoder sequentially

        Args:
            x (torch.Tensor): input image

        Returns:
            _type_: torch.Tensor
        """
        original_shape = x.size()
        x = x.flatten(start_dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(original_shape)
        return x

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """calculate a latent vector in AE

        Args:
            x (torch.Tensor): input image

        Returns:
            torch.Tensor: latent vector
        """
        with torch.no_grad():
            x = x.flatten(start_dim=1)
            x = self.encoder(x)
        return x
