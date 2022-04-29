from typing import List

import torch
from torch import nn
from torch.nn import functional as F


class Stack(nn.Module):
    def __init__(self, channels, height, width):
        super(Stack, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.height, self.width)


class VAE(nn.Module):
    # pylint: disable=missing-class-docstring
    def __init__(self) -> None:
        super(VAE, self).__init__()

        self.hidden_size = 2
        self.channels = 1
        self.height = 28
        self.width = 28
        final_height = (self.height // 4 - 3) // 2 + 1
        final_width = (self.width // 4 - 3) // 2 + 1
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x14x14
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x7x7
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),  # 128*3*3
            nn.Flatten(start_dim=1),
            nn.Linear(
                128 * final_height * final_width, 32 * final_height * final_width
            ),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32 * final_height * final_width),
            nn.Linear(32 * final_height * final_width, self.hidden_size * 2),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, 32 * final_height * final_width),
            nn.BatchNorm1d(32 * final_height * final_width),
            nn.ReLU(),
            nn.Linear(
                32 * final_height * final_width, 128 * final_height * final_width
            ),
            nn.BatchNorm1d(128 * final_height * final_width),
            nn.ReLU(),
            Stack(128, 3, 3),
            nn.ConvTranspose2d(128, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 2, 2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, 2, 2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, self.channels, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = result[:, :2]
        log_var = result[:, 2:4]

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    # pylint: disable=missing-function-docstring
    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (torch.device) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
