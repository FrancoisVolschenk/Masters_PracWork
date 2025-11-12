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
from ray import tune
from ray.train import Checkpoint
from ray.tune import get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import tempfile
from pathlib import Path
from ray.tune import RunConfig

BASE_PATH = "/home/francois/Documents/UniversityWork/UJ_Masters/Development/Masters_PracWork/Fingerprint_Synthesis"


def memory_stats():
    print(torch.cuda.memory_allocated() / 1024**2)
    print(torch.cuda.memory_reserved() / 1024**2)

config = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "latent_dim": tune.choice([64, 100, 128, 256]),
    "batch_size": tune.choice([16, 32]),
    "clip_value": tune.choice([0.5, 1.0, 2.0, 5.0]),
    "mse_type": tune.choice(["mean", "sum"]),
    "base_kl_weight": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
    "kl_weight": tune.loguniform(0.1, 0.8),
    "ssim_weight": tune.choice([0.25, 0.5, 0.75])
}

print("Configuring training device")
# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
memory_stats()
torch.cuda.empty_cache()

TIME_STAMP = datetime.now().strftime("%b%d_%H:%M")

transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            512, scale=(0.9, 1.1), interpolation=Image.BILINEAR
        ),
        transforms.RandomRotation(10, fill=255),
        transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), shear=5, fill=255),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]  # -> [-1, 1]
)

def load_data(batch_size):
    print("Loading the training dataset")

    dataset = ImageFolder(
        root=f"{BASE_PATH}/dataset/Combined",
        # root=f"{BASE_PATH}/dataset/light_bg",
        transform=transform,
    )
    train_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    print("Loading the evaluation dataset")
    test_dataset = ImageFolder(
        root=f"{BASE_PATH}/dataset/light_bg",
        transform=transform,
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return train_loader, test_loader


def process_image(model, img_path):
    img = Image.open(img_path)

    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        reconstructed, mu, logvar = model(x)

    reconstructed = (reconstructed.squeeze(0).cpu() * 0.5) + 0.5
    original = (x.squeeze(0).cpu() * 0.5) + 0.5
    combined = torch.cat((original, reconstructed), dim=1)

    return combined

def train_VAE(config, data_dir = None):
    NUM_EPOCHS = 500
    BATCH_SIZE = config["batch_size"]
    LEARNING_RATE = config["lr"]
    latent_dim = config["latent_dim"]
    mse_type = config["mse_type"] # mean or sum
    clip_value = config["clip_value"]
    base_KL_WEIGHT = config["base_kl_weight"]
    KL_WEIGHT = config["kl_weight"]
    SSIM_WEIGHT = config["ssim_weight"]
    MSE_WEIGHT = min((1 - SSIM_WEIGHT - KL_WEIGHT), 0.1)

    print("Building the model")
    model = ConvVAE(image_channels=1, latent_dim=latent_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    mse_loss = nn.MSELoss(reduction=mse_type)

    MODEL_NAME = f"{TIME_STAMP}"
    train_loader, test_loader = load_data(BATCH_SIZE)

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["model_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
            global_step = checkpoint_state["global_step"]
    else:
        start_epoch = 0
        global_step = 0

    # writer = SummaryWriter(f"./Masters_PracWork/runs/VAE/{MODEL_NAME}")
    # writer = SummaryWriter(log_dir=tune.get_context().get_trial_dir())

    # test_images = []
    # # -- Create a list of images to use for progress tracking
    # files = os.listdir(f"{BASE_PATH}/dataset/light_bg/fp")
    # for i in range(4):
    #     test_images.append(
    #         f"{BASE_PATH}/dataset/light_bg/fp/{random.choice(files)}"
    #     )
    # files = os.listdir(
    #     f"{BASE_PATH}/dataset/Cross_Fp_Processed/fp"
    # )
    # for i in range(4):
    #     test_images.append(
    #         f"{BASE_PATH}/dataset/Cross_Fp_Processed/fp/{random.choice(files)}"
    #     )
    # files = []

    # Do training
    print("Starting the training")
    for epoch in range(start_epoch, NUM_EPOCHS):
        KL_WEIGHT = base_KL_WEIGHT * (1 / (1 + np.exp(-0.1 * (epoch - 10))))
        print(f"{epoch=}")
        # loop = tqdm(enumerate(train_loader), total=len(train_loader))
        total_loss_MSE = 0
        total_loss_SSIM = 0
        total_loss_KL = 0
        total_loss_combined = 0
        # for i, (x, _) in loop:
        for i, (x, _) in enumerate(train_loader):
            # forward pass
            x = x.to(DEVICE)
            x_reconstructed, mu, logvar = model(x)

            reconstruction_loss = mse_loss(x_reconstructed, x)
            # writer.add_scalar("Loss/MSE", reconstruction_loss.item(), global_step)
            ssim_loss = 1 - ssim(x_reconstructed, x, data_range=2.0, size_average=True)
            # writer.add_scalar("Loss/SSIM", ssim_loss.item(), global_step)
            # if ssim_loss < 0.15:
            #     goodenough = True
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss_MSE += reconstruction_loss.item()
            total_loss_SSIM += ssim_loss.item()
            total_loss_KL += kl_divergence.item()

            loss = (
                ((KL_WEIGHT) * kl_divergence)
                + ((MSE_WEIGHT) * reconstruction_loss)
                + ((SSIM_WEIGHT) * ssim_loss)
            )
            total_loss_combined += loss.item()

            # writer.add_scalar(
            #     "Loss/KL(clamped)", min(kl_divergence.item(), 1e6), global_step
            # )
            # writer.add_scalar(
            #     "Loss/KL_weighted", (KL_WEIGHT * kl_divergence).item(), global_step
            # )

            # Backprop
            # writer.add_scalar("Loss/train", loss.item(), global_step)

            global_step += 1
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
            optimizer.step()
            # loop.set_postfix(
            #     kl=kl_divergence.item(),
            #     mse=reconstruction_loss.item(),
            #     ssim=ssim_loss.item(),
            #     combined_loss=loss.item(),
            # )
            # --- Show the progress of the generated images
            # if i % 500 == 0:
            #     with torch.no_grad():
            #         samples = []
            #         for image in test_images:
            #             samples.append(
            #                 process_image(
            #                     model,
            #                     image,
            #                 )
            #             )
            #         grid = make_grid(samples, nrow=4, normalize=True)
            #         writer.add_image(f"{MODEL_NAME}/Reconstructions", grid, global_step)
        # if epoch % 15 == 0:
        #     # -- Save a checkpoint version
        #     torch.save(
        #         model.state_dict(),
        #         f"./Masters_PracWork/Fingerprint_Synthesis/model/{MODEL_NAME}",
        #     )
        # epoch += 1
        avg_loss_MSE = total_loss_MSE / len(train_loader)
        avg_loss_SSIM = total_loss_SSIM / len(train_loader)
        avg_loss_KL = total_loss_KL / len(train_loader)
        avg_loss_combined = total_loss_combined / len(train_loader)

        metrics = {"loss_mse": avg_loss_MSE, "loss_ssim": avg_loss_SSIM, "loss_kl": avg_loss_KL, "loss_combined": avg_loss_combined}
        if epoch % 50 == 0:

            checkpoint_data = {
                "epoch"                   : epoch + 1,
                "model_state_dict"    : model.state_dict(),
                "optimizer_state_dict"  : optimizer.state_dict(),
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
        else:
            tune.report(metrics)

    # writer.flush()
    # writer.close()

    # # Save the model
    # print("Saving the model")
    # torch.save(
    #     model.state_dict(),
    #     f"./Masters_PracWork/Fingerprint_Synthesis/model/{MODEL_NAME}",
    # )

    del model
    del optimizer
    del train_loader
    del test_loader

# tensorboard --logdir=runs
scheduler = ASHAScheduler(
    metric="loss_combined",
    mode="min",
    max_t=500,
    grace_period=25,
    reduction_factor=3,
)
trainable_with_resources = tune.with_resources(train_VAE, {"cpu": 32, "gpu": 0.25})
tuner = tune.Tuner(
    trainable_with_resources,
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        num_samples=40,
        reuse_actors=True
    ),
    param_space=config,
    run_config=RunConfig(
        name="vae_tuning",
        storage_path=f"file://{BASE_PATH}/runs/ray_tune",
        verbose=1,
    ),
)


results = tuner.fit()
best_result = results.get_best_result(metric="loss_combined", mode="min")
print("Best config:", best_result.config)