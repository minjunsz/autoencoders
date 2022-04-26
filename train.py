# %%
import argparse
import os
import yaml

import torch
import wandb
from torch import nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from classic_autoencoder import AutoEncoder

# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    '--configFile', help="config yaml file name", default='config.yaml')
args = parser.parse_args()
print("Config File: ", args.configFile)
with open(args.configFile, 'r', encoding='UTF8') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

os.environ["WANDB_NOTEBOOK_NAME"] = "standard autoencoder"
wandb.init(project="autoencoder", entity="minjunsz",
           config=configs, group=configs["model_type"])
DATA_DIR = "../data"
device = torch.device('cuda') \
    if torch.cuda.is_available() else torch.device('cpu')
print("Learning on: ", device)
# %%
train_dataset = datasets.MNIST(
    root=DATA_DIR, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(
    root=DATA_DIR, train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=wandb.config.batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=wandb.config.batch_size,
    shuffle=False
)
# %%
if wandb.config["model_type"] == "classic-AE":
    first_img, _ = train_dataset[0]
    input_dim = first_img.flatten().size(0)

    model = AutoEncoder(
        input_dim=input_dim,
        hidden_dim1=wandb.config.hidden_dim1,
        hidden_dim2=wandb.config.hidden_dim2,
        latent_dim=wandb.config.latent_dim
    )

model = model.to(device)
optimizer = Adam(model.parameters(), lr=wandb.config.learning_rate)
criterion = nn.MSELoss()
# %%
sample_images, _ = next(iter(train_loader))
sample_images = (sample_images[:6]).to(device)

# %%
for epoch in tqdm(range(wandb.config.num_epochs), desc="train progress"):
    epoch_loss = 0.
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)

        model.train()
        output = model(images)
        loss = criterion(output, images)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_loss /= len(train_loader)
    wandb.log({"Train loss": epoch_loss*100})

    if epoch == 0:
        with torch.no_grad():
            epoch1_output = model(sample_images.to(device))

    if epoch % 5 == 4:
        eval_loss = 0
        with torch.no_grad():
            for (images, _) in test_loader:
                images = images.to(device)
                model.eval()
                output = model(images)
                loss = criterion(output, images)
                eval_loss += loss.item()
        eval_loss /= len(test_loader)
        wandb.log({"Validation loss": eval_loss*100})

# %%
with torch.no_grad():
    final_output = model(sample_images.to(device))

images = torch.cat(
    list(map(lambda images: make_grid(images),
             [sample_images, epoch1_output, final_output])),
    dim=1
)
images = wandb.Image(
    images, caption="Top: Original, Middle: 1st epoch, Bottom: Trained")
wandb.log({
    "examples": images
})
