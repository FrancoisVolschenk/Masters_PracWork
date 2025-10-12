import torch
import torchvision.datasets as datasets
from torch import nn, optim
from pytorch_msssim import ssim
from VAE_Attia_et_al import ConvVAE
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from datetime import datetime


def memory_stats():
    print(torch.cuda.memory_allocated() / 1024**2)
    print(torch.cuda.memory_reserved() / 1024**2)


print("Configuring training device")
# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
memory_stats()
torch.cuda.empty_cache()
NUM_EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 0.00002  # 1e-3
KL_WEIGHT = 0.01
SSIM_WEIGHT = 0.75
MSE_WEIGHT = 1 - SSIM_WEIGHT - KL_WEIGHT
TIME_STAMP = datetime.now().strftime("%b%d_%H:%M")

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5)),
    ]  # -> [-1, 1]
)

print("Loading the dataset")

dataset = ImageFolder(
    root=f"./Fingerprint_Synthesis/dataset/light_bg", transform=transform
)
train_loader = DataLoader(
    dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)

print("Building the model")
model = ConvVAE(image_channels=1, latent_dim=128).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
mse_loss = nn.MSELoss(reduction="sum")

MODEL_NAME = f"{TIME_STAMP}"

writer = SummaryWriter(f"runs/VAE/{MODEL_NAME}")
global_step = 0

# Do training
print("Starting the training")
epoch = 0
goodenough = False
# for epoch in range(NUM_EPOCHS):
while not goodenough:
    print(f"{epoch=}")
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (x, _) in loop:
        # forward pass
        x = x.to(DEVICE)
        x_reconstructed, mu, logvar = model(x)

        # Compute loss
        # loss = 0
        # if CONFIGS["MSE"]:
        reconstruction_loss = mse_loss(x_reconstructed, x)
        # loss += (MSE_WEIGHT) * reconstruction_loss
        writer.add_scalar("Loss/MSE", reconstruction_loss.item(), global_step)
        # if CONFIGS["SSIM"]:
        ssim_loss = 1 - ssim(x_reconstructed, x, data_range=2.0, size_average=True)
        # loss += (SSIM_WEIGHT) * ssim_loss
        writer.add_scalar("Loss/SSIM", ssim_loss.item(), global_step)
        if ssim_loss < 0.15:
            goodenough = True
        # if CONFIGS["KLDIV"]:
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (
            (((KL_WEIGHT) * kl_divergence))
            + ((MSE_WEIGHT) * reconstruction_loss)
            + ((SSIM_WEIGHT) * ssim_loss)
        )

        writer.add_scalar("Loss/KL", kl_divergence.item(), global_step)

        # Backprop
        writer.add_scalar("Loss/train", loss.item(), global_step)

        global_step += 1
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loop.set_postfix(
            kl=kl_divergence.item(),
            mse=reconstruction_loss.item(),
            ssim=ssim_loss.item(),
            combined_loss=loss.item(),
        )
    epoch += 1

writer.flush()
writer.close()

# Save the model
print("Saving the model")
torch.save(
    model.state_dict(),
    f"./Fingerprint_Synthesis/model/{MODEL_NAME}",
)

del model
del optimizer
del train_loader

# tensorboard --logdir=runs
