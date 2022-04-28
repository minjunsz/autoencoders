# %%
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import wandb


def get_device() -> torch.device:
    """return cuda device if a GPU available.

    Returns:
        _type_: torch.device
    """
    device = torch.device('cuda') \
        if torch.cuda.is_available() else torch.device('cpu')
    print("Device: ", device)
    return device


def get_mnist_dataloader() -> Dict[str, DataLoader]:
    """return a dictionary including MNIST dataloader (for both train/test)
    this can be done with torch.utils.data.datalo

    Returns:
        _type_: Dict[str, DataLoader]
    """
    DATA_DIR = "../data"
    train_dataset = datasets.MNIST(
        root=DATA_DIR, train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(
        root=DATA_DIR, train=False, transform=transforms.ToTensor())

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=wandb.config.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=wandb.config.batch_size,
        shuffle=False
    )

    return {
        "train_loader": train_loader,
        "test_loader": test_loader
    }


def recover_image(model: nn.Module, sample_images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """recover given image(sample_images) by autoencoder 

    Args:
        model (nn.Module): autoencoder
        sample_images (torch.Tensor): images to reconstruct
        device (torch.device): result of get_device() function

    Returns:
        torch.Tensor: recovered image
    """
    with torch.no_grad():
        model.eval()
        output = model(sample_images.to(device))
    return output


def train(
        model: nn.Module, optimizer: OptimizerOptimizer, criterion: _Loss, device: torch.device,
        config: Dict[str, Any], train_loader: DataLoader, val_loader: DataLoader,
        sample_images: Optional[torch.Tensor] = None, log_interval: int = 5) -> Dict[str, Any]:
    """train loop

    Args:
        model (nn.Module): model must be loaded on proper device before running this method
        optimizer (Optimizer): optimizer must be initialized with parameters on proper device
        criterion (_Loss): Probably use MSE Loss for autoencoder (End-to-End train)
        device (torch.device): result of get_device() function
        config (Dict[str, Any]): pass wandb.config object
        train_loader (DataLoader): pytorch dataloader for test loop
        val_loader (DataLoader): pytorch dataloader for validation step
        sample_imgs (Optional[torch.Tensor], optional): sample image used to compare model after it's trained. Defaults to None.
        log_interval (int, optional): log interval. Defaults to 5.

    Returns:
        Dict[str, Any]: dictionary contains two data;
            'model'-> model itself, 
            'initial_result' -> recovered 'sample imgs' after first train loop
    """
    for epoch in tqdm(range(config.num_epochs), desc="train progress"):
        epoch_loss = 0.
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)

            model.train()
            output = model(images)
            loss = criterion(output, images)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss /= len(train_loader)
        wandb.log({"Train loss": epoch_loss})

        if epoch == 0 and sample_images is not None:
            epoch1_output = recover_image(
                model=model, sample_images=sample_images, device=device)

        if epoch % log_interval == (log_interval-1):
            val_loss = 0
            with torch.no_grad():
                for (images, _) in val_loader:
                    images = images.to(device)
                    model.eval()
                    output = model(images)
                    loss = criterion(output, images)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            wandb.log({"Validation loss": val_loss})

    return {
        "model": model,
        "initial_result": epoch1_output
    }


# %%
