import random
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
from torchvision.utils import make_grid
import os
from PIL import Image
import numpy as np
from GAN_Minaee_et_al import Discriminator


def memory_stats():
    print(torch.cuda.memory_allocated() / 1024**2)
    print(torch.cuda.memory_reserved() / 1024**2)


print("Configuring training device")
# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
memory_stats()
torch.cuda.empty_cache()
NUM_EPOCHS = 500
BATCH_SIZE = 20
LEARNING_RATE = 0.00002  # 1e-3
base_KL_WEIGHT = 0.4
KL_WEIGHT = 0.05
SSIM_WEIGHT = 0.75
MSE_WEIGHT = 1 - SSIM_WEIGHT - KL_WEIGHT
TIME_STAMP = datetime.now().strftime("%b%d_%H:%M")

transform = transforms.Compose(
    [
        # transforms.RandomResizedCrop(
        #     512, scale=(0.9, 1.1), interpolation=Image.BILINEAR
        # ),
        # transforms.RandomRotation(10, fill=255),
        # transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), shear=5, fill=255),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]  # -> [-1, 1]
)


def process_image(model, img_path):
    img = Image.open(img_path)

    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        reconstructed, mu, logvar = model(x)

    reconstructed = (reconstructed.squeeze(0).cpu() * 0.5) + 0.5
    original = (x.squeeze(0).cpu() * 0.5) + 0.5
    combined = torch.cat((original, reconstructed), dim=1)

    return combined


print("Loading the dataset")

dataset = ImageFolder(
    root=f"./Masters_PracWork/Fingerprint_Synthesis/dataset/light_bg",
    transform=transform,
)
train_loader = DataLoader(
    dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)

print("Building the model")
model = ConvVAE(image_channels=1, latent_dim=128).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
mse_loss = nn.MSELoss(reduction="sum")

discriminator = Discriminator(image_channels=1).to(DEVICE)
optimizer_D = optim.AdamW(
    discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
)
criterion_adv = nn.BCEWithLogitsLoss()

MODEL_NAME = f"{TIME_STAMP}"

writer = SummaryWriter(f"./Masters_PracWork/runs/VAE_GAN/{MODEL_NAME}")
global_step = 0

test_images = []
# -- Create a list of images to use for progress tracking
files = os.listdir("./Masters_PracWork/Fingerprint_Synthesis/dataset/light_bg/fp")
for i in range(4):
    test_images.append(
        f"./Masters_PracWork/Fingerprint_Synthesis/dataset/light_bg/fp/{random.choice(files)}"
    )
files = os.listdir(
    "./Masters_PracWork/Fingerprint_Synthesis/dataset/Cross_Fp_Processed/fp"
)
for i in range(4):
    test_images.append(
        f"./Masters_PracWork/Fingerprint_Synthesis/dataset/Cross_Fp_Processed/fp/{random.choice(files)}"
    )
files = []

# Do training
print("Starting the training")
epoch = 0
goodenough = False
# for epoch in range(NUM_EPOCHS):
while not goodenough:
    # KL_WEIGHT = base_KL_WEIGHT * (1 / (1 + np.exp(-0.1 * (epoch - 10))))
    print(f"{epoch=}")
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (x, _) in loop:
        # forward pass
        x = x.to(DEVICE)
        x_reconstructed, mu, logvar = model(x)

        real_labels = torch.ones(x.size(0), 1, device=DEVICE)
        fake_labels = torch.zeros(x.size(0), 1, device=DEVICE)

        # --- Train Discriminator
        optimizer_D.zero_grad()
        real_pred = discriminator(x)
        fake_pred = discriminator(x_reconstructed.detach())
        loss_real = criterion_adv(real_pred, real_labels)
        loss_fake = criterion_adv(fake_pred, fake_labels)
        loss_D = (loss_real + loss_fake) / 2
        writer.add_scalar("Loss/D", loss_D.item(), global_step)
        loss_D.backward()
        optimizer_D.step()

        reconstruction_loss = mse_loss(x_reconstructed, x)
        writer.add_scalar("Loss/MSE", reconstruction_loss.item(), global_step)
        ssim_loss = 1 - ssim(x_reconstructed, x, data_range=2.0, size_average=True)
        writer.add_scalar("Loss/SSIM", ssim_loss.item(), global_step)
        if ssim_loss < 0.2:
            goodenough = True
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        fake_pred = discriminator(x_reconstructed)
        adv_loss = criterion_adv(fake_pred, real_labels)

        loss = (
            ((KL_WEIGHT) * kl_divergence)
            + ((MSE_WEIGHT) * reconstruction_loss)
            + ((SSIM_WEIGHT) * ssim_loss)
            + (0.1 * adv_loss)
        )

        writer.add_scalar(
            "Loss/KL(clamped)", min(kl_divergence.item(), 1e6), global_step
        )
        writer.add_scalar(
            "Loss/KL_weighted", (KL_WEIGHT * kl_divergence).item(), global_step
        )
        writer.add_scalar("Loss/ADV", adv_loss.item(), global_step)

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
        # --- Show the progress of the generated images
        if i % 500 == 0:
            with torch.no_grad():
                samples = []
                for image in test_images:
                    samples.append(
                        process_image(
                            model,
                            image,
                        )
                    )
                grid = make_grid(samples, nrow=4, normalize=True)
                writer.add_image(f"{MODEL_NAME}/Reconstructions", grid, global_step)
    if epoch % 15 == 0:
        # -- Save a checkpoint version
        torch.save(
            model.state_dict(),
            f"./Masters_PracWork/Fingerprint_Synthesis/model/VAE_GAN/{MODEL_NAME}",
        )
        torch.save(
            discriminator.state_dict(),
            f"./Masters_PracWork/Fingerprint_Synthesis/model/VAE_GAN/{MODEL_NAME}_Discriminator",
        )
    epoch += 1

writer.flush()
writer.close()

# Save the model
print("Saving the model")
torch.save(
    model.state_dict(),
    f"./Masters_PracWork/Fingerprint_Synthesis/model/VAE_GAN/{MODEL_NAME}",
)
torch.save(
    discriminator.state_dict(),
    f"./Masters_PracWork/Fingerprint_Synthesis/model/VAE_GAN/{MODEL_NAME}_Discriminator",
)

del model
del optimizer
del optimizer_D
del discriminator
del train_loader

# tensorboard --logdir=runs
