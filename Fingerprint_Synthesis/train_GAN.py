import torch
import torch.nn as nn
import torch.optim as optim
from GAN_Minaee_et_al import TVGan, Discriminator
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from datetime import datetime

# --- SAMPLE TEST CODE
# x = torch.randn(8, 100, device="cuda")
# gen = TVGan().to("cuda")
# out = gen(x)
# print(out.shape)
# out.mean().backward()

# disc = Discriminator().to("cuda")
# y = disc(out)
# print(y.shape)
# --- END SAMPLE TEST CODE


def memory_stats():
    print(torch.cuda.memory_allocated() / 1024**2)
    print(torch.cuda.memory_reserved() / 1024**2)


print("Configuring training device")
# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
memory_stats()
torch.cuda.empty_cache()
BATCH_SIZE = 32
DISCRIMINATOR_TRIGGER = 4
TIME_STAMP = datetime.now().strftime("%b%d_%H:%M")

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5)),
    ]  # -> [-1, 1]
)

print("Loading the dataset")

# dataset = ImageFolder(
#     root="./Fingerprint_Synthesis/dataset/Cross_Fp_Processed_64x64", transform=transform
# )
dataset = ImageFolder(
    root="./Fingerprint_Synthesis/dataset/Cross_Fp_Processed", transform=transform
)
train_loader = DataLoader(
    dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)

print("Building the model")
generator = TVGan(image_channels=1).to(DEVICE)
discriminator = Discriminator(image_channels=1).to(DEVICE)

MODEL_NAME = f"{TIME_STAMP}"

# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()

LEARNING_RATE = 0.0002
beta1 = 0.5
NUM_EPOCHS = 250
lambda_tv = 0.05

optimizer_G = optim.AdamW(
    generator.parameters(), lr=LEARNING_RATE, betas=(beta1, 0.999)
)
optimizer_D = optim.AdamW(
    discriminator.parameters(), lr=LEARNING_RATE, betas=(beta1, 0.999)
)


def total_variation_loss(img):
    diff_x = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    diff_y = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    return torch.mean(diff_x) + torch.mean(diff_y)


writer = SummaryWriter(f"runs/GAN/{MODEL_NAME}")
global_step = 0
print("Starting the training")
for epoch in range(NUM_EPOCHS):
    print(f"{epoch=}")
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (real_imgs, _) in loop:
        real_imgs = real_imgs.to(DEVICE)
        batch_size = real_imgs.size(0)

        # Create labels for real and fake images
        real_labels = torch.ones(batch_size, 1, device=DEVICE)  # * 2 - 1

        if global_step < 1000 or global_step % DISCRIMINATOR_TRIGGER == 0:
            fake_labels = torch.zeros(batch_size, 1, device=DEVICE)  # * 2 - 1

            # --- Train Discriminator
            optimizer_D.zero_grad()
            outputs_real = discriminator(real_imgs)
            loss_real = criterion(outputs_real, real_labels)
            writer.add_scalar("Loss/REAL", loss_real.item(), global_step)

            z = torch.randn(batch_size, 100, device=DEVICE)
            fake_imgs = generator(z)
            outputs_fake = discriminator(fake_imgs.detach())
            loss_fake = criterion(outputs_fake, fake_labels)
            writer.add_scalar("Loss/FAKE", loss_fake.item(), global_step)

            loss_D = (loss_real + loss_fake) / 2
            writer.add_scalar("Loss/D", loss_D.item(), global_step)
            loss_D.backward()
            optimizer_D.step()

        # --- Train Generator
        optimizer_G.zero_grad()

        z = torch.randn(batch_size, 100, device=DEVICE)
        fake_imgs = generator(z)

        outputs = discriminator(fake_imgs)
        adv_loss = criterion(
            outputs, real_labels
        )  # Try to get the score of the outputs to all show as real
        writer.add_scalar("Loss/ADV", adv_loss.item(), global_step)
        tv_loss = total_variation_loss(fake_imgs) * lambda_tv
        writer.add_scalar("Loss/TV", tv_loss.item(), global_step)

        loss_G = adv_loss + tv_loss
        writer.add_scalar("Loss/G", loss_G.item(), global_step)
        loss_G.backward()
        optimizer_G.step()
        global_step += 1
        loop.set_postfix(
            loss_real=loss_real.item(),
            loss_fake=loss_fake.item(),
            loss_D=loss_D.item(),
            adv_loss=adv_loss.item(),
            tv_loss=tv_loss.item(),
            loss_G=loss_G.item(),
        )
        # --- Show the progress of the generated images
        if i % 500 == 0:
            with torch.no_grad():
                sample_z = torch.randn(16, 100, device=DEVICE)
                samples = generator(sample_z)
                grid = make_grid(samples, nrow=4, normalize=True)
                writer.add_image("Generated", grid, global_step)


writer.flush()
writer.close()

# Save the model
print("Saving the model")
torch.save(
    generator.state_dict(),
    f"./Fingerprint_Synthesis/model/GANs/{MODEL_NAME}_Generator",
)
torch.save(
    discriminator.state_dict(),
    f"./Fingerprint_Synthesis/model/GANs/{MODEL_NAME}_Discriminator",
)

del generator
del discriminator
del optimizer_D
del optimizer_G
del train_loader
