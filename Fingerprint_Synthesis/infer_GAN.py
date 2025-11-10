from GAN_Minaee_et_al import TVGan
import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "Oct09_20:24_Generator"
torch.manual_seed(0)


def load_model(model_name):
    # Recreate the model
    model = TVGan(image_channels=1).to(DEVICE)
    model.load_state_dict(
        torch.load(
            f"./Masters_PracWork/Fingerprint_Synthesis/model/GANs/{model_name}",
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


if __name__ == "__main__":
    n = int(input("Please enter the number of images you wish to generate: "))
    model = load_model(MODEL_NAME)
    # img = generate_image(model)
    # # img.show()
    # img.save("generated_fingerprint.png")

    imgs = generate_batch(model, n)
    for index, img in enumerate(imgs):
        to_pil_image(img).save(
            f"Masters_PracWork/Fingerprint_Synthesis/dataset/generated_by_GAN/Generated_{index}.png"
        )
    print(f"Successfully generated {n} images!")
    grid = make_grid(imgs, nrow=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
