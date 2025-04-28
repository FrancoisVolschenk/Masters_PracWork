import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
import pickle

def draw_circle(radius, center_x = 0.5, center_y = 0.5, size = 28):
    circle = plt.Circle((center_x, center_y), radius, color='k', fill=False)
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.add_patch(circle)
    ax.axis('off')
    buf = fig.canvas.print_to_buffer()
    plt.close()

    return np.array(Image.frombuffer('RGBA', buf[1], buf[0]).convert('L').resize((int(size), int(size))))

def gen_circles(n, size=28):
    print("Generating random coordiantes for the circles")
    center_x = np.random.uniform(0.0, 0.03, size=n).reshape(-1, 1)+.5
    center_y = np.random.uniform(0.0, 0.03, size=n).reshape(-1, 1)+.5

    print("Generating radii for the circles")
    radius = np.random.uniform(0.03, 0.47, size=n).reshape(-1, 1)
    sizes = np.ones((n, 1)) * size

    print("Grouping the data and generating the circles")
    coords = np.concatenate([radius, center_x, center_y, sizes], axis=1)
    circles = np.apply_along_axis(func1d=lambda v: draw_circle(*v), axis=1, arr=coords)
    return circles, radius

def create_and_prep_dataset():
    np.random.seed(42)

    circles, radius = gen_circles(1000)

    print("Creating the dataset")
    circles_ds = TensorDataset(torch.as_tensor(circles).unsqueeze(1).float()/255, torch.as_tensor(radius))
    print("Creating the dataLoader")
    circles_dl = DataLoader(circles_ds, batch_size=32, shuffle=True, drop_last=True)
    return circles_ds, circles_dl

def load_and_prep_dataset():
    with open("Dataset.bin", "rb") as dataset_file:
        dataset = pickle.load(dataset_file)

    print("Dataset successfully retrieved, creating loader")
    circles_dl = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    return dataset, circles_dl

if __name__ == "__main__":
    dataset, dataloader = create_and_prep_dataset()

    print("Storing the dataset")
    with open("Dataset.bin", "wb") as dataset_file:
        pickle.dump(dataset, dataset_file)
