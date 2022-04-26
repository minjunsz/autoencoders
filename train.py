# %%
import os

import torch
import wandb
from torch import nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from standard_autoencoder import AutoEncoder

# %%
os.environ["WANDB_NOTEBOOK_NAME"] = "standard autoencoder"
CONFIG = {
    "num_epochs": 30,
    "batch_size": 64,
    "learning_rate": 0.005,
    "hidden_dim1": 128,
    "hidden_dim2": 64,
    "latent_dim": 4
}
wandb.init(project="autoencoder", entity="minjunsz",
           config=CONFIG, group="standard_autoencoder")
DATA_DIR = "../data"
device = torch.device('cuda') \
    if torch.cuda.is_available() else torch.device('cpu')
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
sample_images = sample_images[:6]

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
