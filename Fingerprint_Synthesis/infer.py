import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from random import choice
from VAE_Attia_et_al import ConvVAE
from torchvision.transforms.functional import to_pil_image

dataset_path = os.path.join(".", "Fingerprint_Synthesis", "dataset", "half", "fp")
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def denormalize(tensor):
    return (tensor * 0.5) + 0.5


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name):
    # Recreate the model
    model = ConvVAE().to(DEVICE)
    model.load_state_dict(
        torch.load(
            f"./Masters_PracWork/Fingerprint_Synthesis/model/{model_name}",
            map_location=DEVICE,
        )
    )
    model.eval()
    return model


def process_image(model, img_path):
    print(f"Opening {img_path}")
    img = Image.open(img_path)

    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        reconstructed, mu, logvar = model(x)

    reconstructed = denormalize(reconstructed.squeeze(0).cpu())
    original = denormalize(x.squeeze(0).cpu())

    return to_pil_image(original), to_pil_image(reconstructed)
