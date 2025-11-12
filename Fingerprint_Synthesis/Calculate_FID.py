import torchvision
from torch.utils.data import DataLoader
from torch_fidelity import calculate_metrics
from pathlib import Path
import torch
import os

from torchvision import transforms
from torchvision.datasets import ImageFolder

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

def compute_real_stats(val_loader=None, cache_path="./fid_real_stats.pt"):
    if os.path.exists(cache_path):
        print(f"Loading the cached FID stats from {cache_path}")
        return torch.load(cache_path)

    temp_dir = Path("./real_fid_cache")
    temp_dir.mkdir(exist_ok=True)

    count = 0
    for imgs, _ in val_loader:
        for i in range(imgs.size(0)):
            torchvision.utils.save_image(imgs[i], temp_dir / f"real_{count:05d}.png", normalize=True)
            count += 1

    metrics = calculate_metrics(
        input1=str(temp_dir),
        input2=str(temp_dir), # compare the set to itself to compute mean and covariance
        fid=True,
        kid=False,
        verbose=False,
        cuda=True,
    )
    os.system(f"rm -rf {temp_dir}")
    torch.save(metrics, cache_path)
    return metrics

def compare_datasets(set1, set2):
    temp_dir1 = Path("./set1_cache")
    temp_dir1.mkdir(exist_ok=True)
    temp_dir2 = Path("./set2_cache")
    temp_dir2.mkdir(exist_ok=True)

    count = 0
    for imgs, _ in set1:
        for i in range(imgs.size(0)):
            torchvision.utils.save_image(imgs[i], temp_dir1 / f"real_{count:05d}.png", normalize=True)
            count += 1

    count = 0
    for imgs, _ in set2:
        for i in range(imgs.size(0)):
            torchvision.utils.save_image(imgs[i], temp_dir2 / f"real_{count:05d}.png", normalize=True)
            count += 1

    metrics = calculate_metrics(
        input1=str(temp_dir1),
        input2=str(temp_dir2),
        fid=True,
        kid=False,
        verbose=False,
        cuda=True,
    )
    os.system(f"rm -rf {temp_dir1}")
    os.system(f"rm -rf {temp_dir2}")
    return metrics

# def compute_fid(generator, val_real_stats, num_samples=1000, batch_size=64):
#     generator.eval()
#     temp_fake_dir = Path("./fake_fid_temp")
#     temp_fake_dir.mkdir(exist_ok=True)
#
#     count = 0
#     with torch.no_grad():
#         while count < num_samples:
#             z = torch.randn(batch_size, 100, device=DEVICE)
#             fake_imgs = generator(z)
#             for i in range(fake_imgs.size(0)):
#                 torchvision.utils.save_image(
#                     fake_imgs[i],
#                     temp_fake_dir / f"fake_{count:05d}.png",
#                     normalize=True,
#                 )
#                 count += 1
#                 if count >= num_samples:
#                     break
#
#     # Compute FID comparing fake dir with real stats
#     metrics = calculate_metrics(
#         input1=str(temp_fake_dir),
#         input2=None,  # None means use precomputed real stats
#         fid=True,
#         kid=False,
#         input2_cache=val_real_stats,  # torch-fidelity supports passing stats directly
#         verbose=False,
#     )
#     os.system(f"rm -rf {temp_fake_dir}")  # cleanup
#     return metrics["frechet_inception_distance"]

if __name__ == "__main__":
    # print(os.listdir("dataset/light_bg"))
    train_dataloader, val_dataloader = load_data()
    val_real_stats = compare_datasets(train_dataloader, val_dataloader)
    print(val_real_stats)