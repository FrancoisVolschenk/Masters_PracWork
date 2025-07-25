import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Define a transform (optional but recommended)
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL image or numpy ndarray to FloatTensor and scales values to [0, 1]
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST (precomputed)
])

# 2. Load the training and test datasets
train_dataset = datasets.MNIST(
    root='./data',         # Where to store/download the data
    train=True,            # True = training set, False = test set
    download=True,         # Downloads if dataset is not found
    transform=transform    # Apply the transform when loading
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 3. Wrap datasets into DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,         # How many samples per batch
    shuffle=True           # Shuffle the data for training
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False
)

# 4. Quick check: get one batch
images, labels = next(iter(train_loader))
print(f'Batch shape: {images.shape}')  # Should print: torch.Size([64, 1, 28, 28])
print(f'Labels shape: {labels.shape}')  # Should print: torch.Size([64])

# 5. Visualize one sample
import matplotlib.pyplot as plt

plt.imshow(images[0].squeeze(), cmap='gray')
plt.title(f'Label: {labels[0].item()}')
plt.axis('off')
plt.show()
