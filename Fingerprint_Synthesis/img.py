from PIL import Image
import os
from pathlib import Path

input_dir = Path("./Masters_PracWork/Fingerprint_Synthesis/dataset/Cross_Fp_Processed")
output_dir = Path(
    "./Masters_PracWork/Fingerprint_Synthesis/dataset/Cross_Fp_Processed_64x64/fp"
)
# target_size = (512, 512)
target_size = (64, 64)

supported_exts = [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"]

output_dir.mkdir(parents=True, exist_ok=True)

counter = 0

for path in input_dir.rglob("*"):
    if path.suffix.lower() in supported_exts:
        print(f"Processing {path}")
        img = Image.open(path).convert("L")
        img = img.resize(target_size, Image.LANCZOS)
        out_path = output_dir / f"{counter:05d}.png"
        img.save(out_path)
        counter += 1

print(f"Processed {counter} images into {output_dir}")
