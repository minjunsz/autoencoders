"defining primitive autoencoder"
import torch
from torch import nn


class AutoEncoder(nn.Module):
    """primitive autoencoder with FC layers only"""

    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, latent_dim: int):
        """initialize autoencoder w/ one hidden layer
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
        x = x.flatten(start_dim=1)
        x = self.encoder(x)
        return x
