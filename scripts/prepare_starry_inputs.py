import json
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

DATASET_DIR = Path("output/dataset")
REFERENCE_JSON = Path("reference.json")
OUT_DIR = Path("starry_inputs")

IMAGE_PATHS = [
    "videos/observation.images.fpv/chunk-000/frame_000000.jpg",
    "videos/observation.images.fpv/chunk-000/frame_000227.jpg",
]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def main():
    OUT_DIR.mkdir(exist_ok=True)

    with open(REFERENCE_JSON, "r", encoding="utf-8") as f:
        ref = json.load(f)

    state = np.array(ref["cases"][0]["state_norm"], dtype=np.float32).reshape(1, 2)
    latent = np.array(ref["latent_mu"], dtype=np.float32).reshape(1, 32)

    state.tofile(OUT_DIR / "state.bin")
    latent.tofile(OUT_DIR / "latent.bin")

    for rel_path in IMAGE_PATHS:
        img = Image.open(DATASET_DIR / rel_path).convert("RGB")
        arr = IMAGE_TRANSFORM(img).unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)
        name = Path(rel_path).name.replace(".jpg", "_input.bin")
        arr.tofile(OUT_DIR / name)

    print("saved to", OUT_DIR)

if __name__ == "__main__":
    main()
