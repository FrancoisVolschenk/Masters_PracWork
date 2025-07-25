from PIL import Image
import numpy as np

width = 256
height = 144


with open("fingerprint.raw", "rb") as f:
    raw_data = f.read()

# if len(raw_data) != width * height:
#     raise ValueError(f"Unexpected file size: got {len(raw_data)} bytes, expected {width * height}")

img_array = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width))
img = Image.fromarray(img_array, mode='L')

# cropped = img.crop((0, 0, width // 2, height))

# cropped.save("fingerprint.png")
# cropped.show()

img.save("fingerprint.png")
img.show()