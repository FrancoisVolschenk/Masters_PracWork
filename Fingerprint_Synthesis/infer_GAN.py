import os
from pathlib import Path

import torchvision
from torch.utils.data import DataLoader
from torch_fidelity import calculate_metrics
from torchvision import transforms
from torchvision.datasets import ImageFolder

from GAN_Minaee_et_al import TVGan
import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "Oct09_20:24_Generator"
torch.manual_seed(0)

BATCH_SIZE = 32


def load_data():
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5)),
        ]  # -> [-1, 1]
    )

    print("Loading the training dataset")
    train_dataset = ImageFolder(
        root="dataset/Cross_Fp_Processed",
        transform=transform,
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    print("Loading the evaluation dataset")
    test_dataset = ImageFolder(
        root="dataset/light_bg",
        transform=transform,
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    return train_loader, test_loader

def load_model(model_name):
    # Recreate the model
    model = TVGan(image_channels=1).to(DEVICE)
    model.load_state_dict(
        torch.load(
            f"model/GANs/{model_name}",
            map_location=DEVICE,
        )
    )
    model.eval()
    return model


def generate_image(model, seed_vector=None):

    if seed_vector is None:
        print("Seeding own vector")
        seed_vector = torch.randn(1, 100, device=DEVICE)

    print("Generating image")
    with torch.no_grad():
        generated = model(seed_vector)

    print("Renormalizing the image")
    generated = (generated + 1) / 2  # scale from [-1,1] → [0,1]
    generated = generated.squeeze(0).cpu()

    print("Converting tensore to image")
    return to_pil_image(generated)


def generate_batch(model, n=16):
    z = torch.randn(n, 100, device=DEVICE)
    with torch.no_grad():
        generated = model(z)
    generated = (generated + 1) / 2
    return generated.cpu()

def compute_fid(generator, val_loader, num_samples=1000, batch_size=64):
    temp_dir = Path("./real_fid_cache")
    temp_dir.mkdir(exist_ok=True)

    count = 0
    for imgs, _ in val_loader:
        for i in range(imgs.size(0)):
            torchvision.utils.save_image(imgs[i], temp_dir / f"real_{count:05d}.png", normalize=True)
            count += 1

    generator.eval()
    temp_fake_dir = Path("./fake_fid_temp")
    temp_fake_dir.mkdir(exist_ok=True)

    count = 0
    with torch.no_grad():
        while count < num_samples:
            z = torch.randn(batch_size, 100, device=DEVICE)
            fake_imgs = generator(z)
            for i in range(fake_imgs.size(0)):
                torchvision.utils.save_image(
                    fake_imgs[i],
                    temp_fake_dir / f"fake_{count:05d}.png",
                    normalize=True,
                )
                count += 1
                if count >= num_samples:
                    break

    # Compute FID comparing fake dir with real stats
    metrics = calculate_metrics(
        input1=str(temp_fake_dir),
        input2=str(temp_dir),
        fid=True,
        kid=False,
        input2_cache_name="temp_cache",
        verbose=False,
    )
    os.system(f"rm -rf {temp_fake_dir}")  # cleanup
    return metrics["frechet_inception_distance"]

if __name__ == "__main__":
    # n = int(input("Please enter the number of images you wish to generate: "))
    model = load_model(MODEL_NAME)

    train_data, test_data = load_data()
    fid_score = compute_fid(model, train_data)
    print(f"FID = {fid_score:.2f}")

    # img = generate_image(model)
    # # img.show()
    # img.save("generated_fingerprint.png")
    #
    # imgs = generate_batch(model, n)
    # for index, img in enumerate(imgs):
    #     to_pil_image(img).save(
    #         f"Masters_PracWork/Fingerprint_Synthesis/dataset/generated_by_GAN/Generated_{index}.png"
    #     )
    # print(f"Successfully generated {n} images!")
    # grid = make_grid(imgs, nrow=4)
    # plt.imshow(grid.permute(1, 2, 0))
    # plt.axis("off")
    # plt.show()
