import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from VAE_Attia_et_al import ConvVAE  # replace with actual filename (no .py)

# -------------------------------
# 1. Argument parsing
# -------------------------------
parser = argparse.ArgumentParser(description="t-SNE visualization for VAE latent space (ImageFolder dataset)")
parser.add_argument("--model_path", type=str, required=True, help="Path to trained VAE weights (.pth)")
parser.add_argument("--data_path", type=str, required=True, help="Path to image dataset root folder")
parser.add_argument("--latent_dim", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--sample_limit", type=int, default=5000)
parser.add_argument("--perplexity", type=float, default=30.0)
parser.add_argument("--num_clusters", type=int, default=10)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 2. Load model
# -------------------------------
vae = ConvVAE(image_channels=1, latent_dim=args.latent_dim).to(device)
vae.load_state_dict(torch.load(args.model_path, map_location=device))
vae.eval()

# -------------------------------
# 3. Load dataset
# -------------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(root=args.data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

print(f"Loaded {len(dataset)} images from {args.data_path}")

# -------------------------------
# 4. Encode dataset into latent space (with caching)
# -------------------------------
cache_path = Path("latent_cache.npz")

if cache_path.exists():
    print(f"Found cached latent encodings at {cache_path}. Loading...")
    data = np.load(cache_path, allow_pickle=True)
    latents = data["latents"]
    filenames = data["filenames"]
    print(f"Loaded {latents.shape[0]} latent vectors from cache.")
else:
    print("Encoding dataset into latent space...")
    latents = []
    filenames = []

    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc="Encoding images", unit="batch"):
            x = x.to(device)
            mu, _ = vae.encode(x)
            latents.append(mu.cpu())
            if len(latents) * args.batch_size >= args.sample_limit:
                break

    latents = torch.cat(latents, dim=0).numpy()
    filenames = [Path(p[0]).name for p in dataset.samples[:len(latents)]]

    # Save to cache
    np.savez(cache_path, latents=latents, filenames=filenames)
    print(f"Saved {latents.shape[0]} latent vectors to cache: {cache_path}")

# -------------------------------
# 5. Apply t-SNE
# -------------------------------
print("Running t-SNE (this may take a few minutes)...")
tsne = TSNE(
    n_components=2,
    perplexity=args.perplexity,
    learning_rate=200,
    max_iter=1000,
    random_state=42,
    init="pca",
    metric="euclidean"
)
embeddings = tsne.fit_transform(latents)

# -------------------------------
# 6. Cluster latents
# -------------------------------
print(f"Running KMeans clustering (k={args.num_clusters})...")
kmeans = KMeans(n_clusters=args.num_clusters, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(latents)

# -------------------------------
# 7. Plot
# -------------------------------
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=cluster_labels, cmap="tab20", s=8, alpha=0.8)
plt.colorbar(scatter, label="Cluster ID")
plt.title("t-SNE Visualization of VAE Latent Space (Clustered)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.tight_layout()
plt.show()

# -------------------------------
# 8. (Optional) Save embeddings
# -------------------------------
out_path = Path("tsne_embeddings.npz")
np.savez(out_path, embeddings=embeddings, clusters=cluster_labels, filenames=filenames)
print(f"Saved t-SNE embeddings to {out_path}")
