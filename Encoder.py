import DataGen
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
import random
import MNIST_data_reader

#We are creating an encoder to condense the images of circles down into a single value (make it try to estimate the radius of the circle)
# The image comes in as 28x28 pixels, we want to flatten it into 784 pixels and pass it throug the encoder.

def set_seed(self, seed=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

class Encoder(nn.Module):
    def __init__(self, input_shape, z_size, base_model):
        super().__init__()
        self.input_shape = input_shape
        self.z_size = z_size
        self.base_model = base_model

        output_size = self._get_output_size()
        self.lin_latent = nn.Linear(output_size, z_size)

    def _get_output_size(self):
        device = next(self.base_model.parameters()).device.type
        dummy = torch.zeros(1, *self.input_shape, device=device)
        size = self.base_model(dummy).size(1)
        return size

    def forward(self, x):
        base_out = self.base_model(x)
        out = self.lin_latent(base_out)
        return out

class AutoEncoder(nn.Module):
    def __init__(self, z_size, input_shape):
        super().__init__()
        self.z_size = z_size
        self.input_shape = input_shape

        base_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape), 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2048),
            nn.LeakyReLU()
        )

        # Build the model
        self.encoder = Encoder(input_shape, z_size, base_model)
        self.decoder = nn.Sequential(
            nn.Linear(z_size, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, np.prod(input_shape)),
            nn.Unflatten(1, input_shape)
        )

    def forward(self, x):
        enc_out = self.encoder(x)
        return self.decoder(enc_out)

def train_and_save_model():
    set_seed(13)
    # Set the model params
    z_size = 10
    input_shape = (1, 28, 28)

    # Get the dataset
    # dataset, dataloader = DataGen.load_and_prep_dataset()

    dataset = MNIST_data_reader.train_dataset
    dataloader = MNIST_data_reader.train_loader

    print("Setting up the AE")
    model_ae = AutoEncoder(z_size, input_shape)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_ae.to(device)
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model_ae.parameters(), 0.0003)

    num_epochs = 50

    train_losses = []
    print("Training the AE")
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}")
        batch_losses = []
        for images, labels in dataloader:
            model_ae.train()
            x = images.to(device)

            optim.zero_grad()
            y_hat = model_ae(x).squeeze(1)
            loss = loss_fn(y_hat, x)
            loss.backward()
            optim.step()

            batch_losses.append(np.array([loss.data.item()]))

        train_losses.append(np.array(batch_losses).mean(axis=0))
        print(f'Epoch {epoch:03d} | Loss >> {train_losses[-1][0]:.4f}')


    print("Training complete. Saving model to file")
    with open("AutoEncoder.bin", "wb") as model_file:
        pickle.dump(model_ae, model_file)

    output = model_ae(dataset[7][0])
    img_tensor = output.squeeze()
    # Normalize to [0, 1] for display (optional depending on your data)
    img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

    # Plot
    plt.imshow(img_tensor.cpu().detach().numpy(), cmap='gray')
    plt.axis('off')
    plt.show()

def load_model():
    with open("AutoEncoder.bin", "rb") as model_file:
        model = pickle.load(model_file)
    return model

if __name__ == "__main__":
    # train_and_save_model()
    model_ae = load_model()
    #
    # dataset, dataloader = DataGen.load_and_prep_dataset()
    dataset = MNIST_data_reader.train_dataset

    fig, axes = plt.subplots(2, 15, figsize=(15, 6))  # Adjust figsize as you like
    axes = axes.flatten()  # Flatten in case axes is a 2D array

    for i in range(30):
        output = model_ae(dataset[random.randint(0, len(dataset) - 1)][0])
        img_tensor = output.squeeze()
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

        ax = axes[i]
        ax.imshow(img_tensor.cpu().detach().numpy(), cmap='gray')
        ax.axis('off')

    plt.tight_layout()  # Optional: improves spacing between subplots
    plt.show()


