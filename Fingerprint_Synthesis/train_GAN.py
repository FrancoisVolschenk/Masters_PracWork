import os

import torch
import torch.nn as nn
import torch.optim as optim
from ray.tune import RunConfig

from GAN_Minaee_et_al import TVGan, Discriminator
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
# from tqdm import tqdm
from datetime import datetime
from ray import tune
from ray.train import Checkpoint
from ray.tune import get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import tempfile
from pathlib import Path

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

config = {
    "lr_G": tune.loguniform(1e-4, 1e-1),
    "lr_D": tune.loguniform(1e-4, 1e-1),
    "beta1": tune.uniform(0.3, 0.9),
    "lambda_tv": tune.loguniform(1e-3, 1e-1),
    "input_dim": tune.choice([100, 128, 256]),
    "batch_size": tune.choice([2, 4, 8, 16, 32]),
    "discriminator_trigger": tune.choice([1, 2, 3, 4, 5]),
    "clip_value": tune.choice([0.5, 1.0, 2.0, 5.0]),
}

print("Configuring training device")
# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
memory_stats()
torch.cuda.empty_cache()
DISCRIMINATOR_TRIGGER = 4
TIME_STAMP = datetime.now().strftime("%b%d_%H:%M")

def load_data(batch_size):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5)),
        ]  # -> [-1, 1]
    )

    print("Loading the training dataset")
    train_dataset = ImageFolder(
        root="/home/francois/Documents/UniversityWork/UJ_Masters/Development/Masters_PracWork/Fingerprint_Synthesis/dataset/Cross_Fp_Processed",
        transform=transform,
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    print("Loading the evaluation dataset")
    test_dataset = ImageFolder(
        root="/home/francois/Documents/UniversityWork/UJ_Masters/Development/Masters_PracWork/Fingerprint_Synthesis/dataset/light_bg",
        transform=transform,
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    return train_loader, test_loader

def total_variation_loss(img):
    diff_x = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    diff_y = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
    return torch.mean(diff_x) + torch.mean(diff_y)

def train_GAN(config, data_dir=None):
    LEARNING_RATE_G = config["lr_G"]
    LEARNING_RATE_D = config["lr_D"]
    beta1 = config["beta1"]
    NUM_EPOCHS = 250
    lambda_tv = config["lambda_tv"]
    BATCH_SIZE = config["batch_size"]
    DISCRIMINATOR_TRIGGER = config["discriminator_trigger"]
    clip_value = config["clip_value"]
    input_dim = config["input_dim"]

    print("Building the model")
    generator = TVGan(image_channels=1, input_dim=input_dim).to(DEVICE)
    discriminator = Discriminator(image_channels=1).to(DEVICE)

    MODEL_NAME = f"{TIME_STAMP}"

    criterion = nn.BCEWithLogitsLoss()

    optimizer_G = optim.AdamW(
        generator.parameters(), lr=LEARNING_RATE_G, betas=(beta1, 0.999)
    )
    optimizer_D = optim.AdamW(
        discriminator.parameters(), lr=LEARNING_RATE_D, betas=(beta1, 0.999)
    )
    train_loader, test_loader = load_data(BATCH_SIZE)

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            generator.load_state_dict(checkpoint_state["generator_state_dict"])
            discriminator.load_state_dict(checkpoint_state["discriminator_state_dict"])
            optimizer_G.load_state_dict(checkpoint_state["optimizer_g_state_dict"])
            optimizer_D.load_state_dict(checkpoint_state["optimizer_d_state_dict"])
            global_step = checkpoint_state["global_step"]
    else:
        start_epoch = 0
        global_step = 0


    # writer = SummaryWriter(f"runs/GAN/{MODEL_NAME}")
    writer = SummaryWriter(log_dir=tune.get_context().get_trial_dir())

    print("Starting the training")
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"{epoch=}")
        generator.train()
        discriminator.train()

        total_loss_G = 0.0
        total_loss_D = 0.0

        # loop = tqdm(enumerate(train_loader), total=len(train_loader))
        # for i, (real_imgs, _) in loop:
        for i, (real_imgs, _) in enumerate(train_loader):
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
                # writer.add_scalar("Loss/REAL", loss_real.item(), global_step)

                z = torch.randn(batch_size, input_dim, device=DEVICE)
                fake_imgs = generator(z)
                outputs_fake = discriminator(fake_imgs.detach())
                loss_fake = criterion(outputs_fake, fake_labels)
                # writer.add_scalar("Loss/FAKE", loss_fake.item(), global_step)

                loss_D = (loss_real + loss_fake) / 2
                # writer.add_scalar("Loss/D", loss_D.item(), global_step)
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=clip_value)
                optimizer_D.step()
                total_loss_D += loss_D.item()

            # --- Train Generator
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, input_dim, device=DEVICE)
            fake_imgs = generator(z)

            outputs = discriminator(fake_imgs)
            adv_loss = criterion(
                outputs, real_labels
            )  # Try to get the score of the outputs to all show as real

            # writer.add_scalar("Loss/ADV", adv_loss.item(), global_step)
            tv_loss = total_variation_loss(fake_imgs) * lambda_tv
            # writer.add_scalar("Loss/TV", tv_loss.item(), global_step)

            loss_G = adv_loss + tv_loss
            # writer.add_scalar("Loss/G", loss_G.item(), global_step)
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=clip_value)
            optimizer_G.step()
            global_step += 1
            total_loss_G += loss_G.item()

            # loop.set_postfix(
            #     loss_real=loss_real.item(),
            #     loss_fake=loss_fake.item(),
            #     loss_D=loss_D.item(),
            #     adv_loss=adv_loss.item(),
            #     tv_loss=tv_loss.item(),
            #     loss_G=loss_G.item(),
            # )

            # --- Show the progress of the generated images
            if i % 500 == 0:
                with torch.no_grad():
                    sample_z = torch.randn(16, input_dim, device=DEVICE)
                    samples = generator(sample_z)
                    grid = make_grid(samples, nrow=4, normalize=True)
                    writer.add_image("Generated", grid, global_step)
        avg_loss_G = total_loss_G / len(train_loader)
        avg_loss_D = total_loss_D / len(train_loader)
        # Report progress to Ray Tune
        metrics = {"loss_G": avg_loss_G, "loss_D": avg_loss_D}
        tune.report(metrics)

        checkpoint_data = {
            "epoch"                   : epoch + 1,
            "generator_state_dict"    : generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict"  : optimizer_G.state_dict(),
            "optimizer_d_state_dict"  : optimizer_D.state_dict(),
            "global_Step": global_step,
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            tune.report(
                metrics,
                checkpoint=checkpoint,
            )


    writer.flush()
    writer.close()

    # # Save the model
    # print("Saving the model")
    # torch.save(
    #     generator.state_dict(),
    #     f"./Masters_PracWork/Fingerprint_Synthesis/model/GANs/{MODEL_NAME}_Generator",
    # )
    # torch.save(
    #     discriminator.state_dict(),
    #     f"./Masters_PracWork/Fingerprint_Synthesis/model/GANs/{MODEL_NAME}_Discriminator",
    # )

    del generator
    del discriminator
    del optimizer_D
    del optimizer_G
    del train_loader
    del test_loader


scheduler = ASHAScheduler(
    metric="loss_G",
    mode="min",
    max_t=250,
    grace_period=5,
    reduction_factor=2,
)
trainable_with_resources = tune.with_resources(train_GAN, {"cpu": 4, "gpu": 1})
tuner = tune.Tuner(
    trainable_with_resources,
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=10,
    ),
    param_space=config,
    run_config=RunConfig(
        name="gan_tuning",
        storage_path="file:///home/francois/Documents/UniversityWork/UJ_Masters/Development/Masters_PracWork/Fingerprint_Synthesis/runs/ray_tune",
        verbose=1,
    ),
)


results = tuner.fit()
best_result = results.get_best_result(metric="loss_G", mode="min")
print("Best config:", best_result.config)