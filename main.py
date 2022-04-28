# %%
import torch
from torch import nn
from torch.optim import Adam

from autoencoders.utils.template import (get_device, get_mnist_dataloader,
                                         recover_image, train)
from autoencoders.utils.cli_arguments import get_arguments
from autoencoders.utils.model_factory import ModelFactory
from autoencoders.utils.wandb_utils import init_wandb, log_images

# %%


def main():
    """main function"""
    cli_args = get_arguments()
    config = init_wandb(cli_args.configFile)

    device = get_device()
    dataloaders = get_mnist_dataloader(config)

    print("Config File: ", cli_args.configFile)
    print("Target Model: ", config.model_type)
    print("Device: ", device)

    model = ModelFactory.create_model(config)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    sample_images, _ = next(iter(dataloaders["train_loader"]))
    sample_images = (sample_images[:6]).to(device)

    train_result = train(
        model=model, optimizer=optimizer, criterion=criterion,
        train_loader=dataloaders["train_loader"], val_loader=dataloaders["test_loader"],
        config=config, device=device, log_interval=5, sample_images=sample_images
    )
    model = train_result["model"]

    epoch1_output = train_result['initial_result']
    final_output = recover_image(model, sample_images, device)

    log_images(sample_images, epoch1_output, final_output)


# %%
if __name__ == "__main__":
    main()
