import torch
import torch.nn as nn


class TVGan(nn.Module):
    def __init__(
        self,
        image_channels=1,
    ):
        super().__init__()

        self.fc_input = nn.Linear(100, 512 * 4 * 4)

        # -------- Generator --------
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),  # 4 -> 8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 8 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 64 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 128 -> 256
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, image_channels, 4, 2, 1),  # 256 -> 512
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc_input(z)
        x = x.view(x.size(0), 512, 4, 4)
        return self.generator(x)


class Discriminator(nn.Module):
    def __init__(
        self,
        image_channels=1,
    ):
        super().__init__()
        # -------- Discriminator --------
        self.discriminator = nn.Sequential(
            nn.Conv2d(image_channels, 16, 4, 2, 1),  # -> (16 x 256 x 256)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1),  # -> (32 x 128 x 128)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # -> (64 x 64 x 64)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # -> (128 x 32 x 32)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # -> (256 x 16 x 16)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),  # -> (512 x 8 x 8)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 4, 2, 1),  # -> (512 x 4 x 4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1),  # Output: single scalar (logit)
        )

    def forward(self, x):
        return self.discriminator(x)
