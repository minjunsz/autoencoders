import torch
import wandb
import yaml
from torchvision.utils import make_grid


def init_wandb(config_path: str):
    """initialize wandb with given config file

    Args:
        config_path (str): path to the configuration file (UTF-8 encoded)

    Returns:
        Dict: wandb config dictionary
    """
    with open(config_path, 'r', encoding='UTF8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # comment out these lines if you want to select specific config file in interactive mode
    # with open("conv-AE-config.yaml", 'r', encoding='UTF8') as f:
    #     configs = yaml.load(f, Loader=yaml.FullLoader)

    wandb.init(project="autoencoder", entity="minjunsz",
               config=config, group=config["model_type"])
    return wandb.config


def log_images(sample_images: torch.Tensor, epoch1_output: torch.Tensor, final_output: torch.Tensor):
    """log sampled image at each step into wandb

    Args:
        sample_images (torch.Tensor): Original image
        epoch1_output (torch.Tensor): recovered image after 1st train loop
        final_output (torch.Tensor): recovered image after whole train loop
    """
    sample_output = torch.cat(list(
        map(make_grid,
            [sample_images, epoch1_output, final_output])),
        dim=1
    )
    sample_output = wandb.Image(
        sample_output,
        caption="Top: Original, Middle: 1st epoch, Bottom: Trained"
    )
    wandb.log({
        "examples": sample_output
    })
