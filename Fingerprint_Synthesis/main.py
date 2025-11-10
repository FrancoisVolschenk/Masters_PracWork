import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"HIP runtime Version: {torch.version.hip}")


# Input image -> Hidden Dimension -> mean, std deviation -> Reparameterizatino trick -> Decoder -> Output image
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, z_dimension):
        super().__init__()

        # Encoder
        self.img_2hid = nn.Linear(input_dimension, hidden_dimension)
        self.hid_2mu = nn.Linear(hidden_dimension, z_dimension)
        self.hid_2sigma = nn.Linear(hidden_dimension, z_dimension)

        # decoder
        self.z_2hid = nn.Linear(z_dimension, hidden_dimension)
        self.hid_2img = nn.Linear(hidden_dimension, input_dimension)

        self.relu = nn.ReLU()

    def encode(self, x):
        # q_phi(z|x)
        hidden_output = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(hidden_output), self.hid_2sigma(hidden_output)
        return mu, sigma

    def decode(self, z):
        # p_theta(x|z)
        hidden_output = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(hidden_output))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + sigma * epsilon
        x_reconstructed = self.decode(z_reparameterized)

        # return mu and sigma for the loss function for the KL Divergence
        return x_reconstructed, mu, sigma


class ConvVAE(nn.Module):
    def __init__(self, image_channels=3, latent_dim=256):
        super().__init__()

        # -------- Encoder --------
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # determine the shape of the encoder's output
        dummy_input = torch.zeros(1, image_channels, 288, 288)
        with torch.no_grad():
            enc_out = self.encoder(dummy_input)
        self.enc_out_shape = enc_out.shape
        self.flatten_dim = enc_out.view(1, -1).shape[1]

        # Latent space layers
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        # -------- Decoder --------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(
            -1, self.enc_out_shape[1], self.enc_out_shape[2], self.enc_out_shape[3]
        )
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


if __name__ == "__main__":
    pass
