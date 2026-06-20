import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

DATASET_DIR = Path("output/dataset")
REFERENCE_JSON = Path("reference.json")
OUT_DIR = Path("artifacts/tpu_mlir_npz")
CALI_DIR = OUT_DIR / "act_cali_dataset"
CALI_LIMIT = 128

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

def rel_case_path(case_path: str) -> Path:
    p = Path(case_path)
    if p.is_absolute():
        try:
            return p.resolve().relative_to(DATASET_DIR.resolve())
        except ValueError:
            return Path(p.name)
    try:
        return p.relative_to(DATASET_DIR)
    except ValueError:
        return p

def frame_tag(path: Path) -> str:
    m = re.search(r"(\d{6})", path.stem)
    return m.group(1) if m else path.stem

def preprocess(rel_path: Path):
    img = Image.open(DATASET_DIR / rel_path).convert("RGB").resize((224, 224))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = (arr - MEAN) / STD
    return arr.reshape(1, 1, 3, 224, 224).astype(np.float32)

def main():
    with open(REFERENCE_JSON, "r", encoding="utf-8") as f:
        ref = json.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CALI_DIR.mkdir(parents=True, exist_ok=True)

    state = np.asarray(ref["cases"][0]["state_norm"], dtype=np.float32).reshape(1, 2)
    latent = np.asarray(ref["latent_mu"], dtype=np.float32).reshape(1, 32)

    manifest = {"test_cases": [], "calibration_cases": []}

    for case in ref["cases"]:
        rel = rel_case_path(case["image_path"])
        tag = frame_tag(rel)
        out = OUT_DIR / f"act_test_{tag}.npz"
        np.savez(out, image=preprocess(rel), state=state, latent=latent)
        manifest["test_cases"].append({
            "image": rel.as_posix(),
            "npz": str(out),
            "direction": case.get("direction"),
        })

    images = sorted(DATASET_DIR.glob("videos/observation.images.fpv/**/*.jpg"))
    if CALI_LIMIT > 0:
        images = images[:CALI_LIMIT]

    for img_path in images:
        rel = img_path.relative_to(DATASET_DIR)
        tag = frame_tag(rel)
        out = CALI_DIR / f"cali_{tag}.npz"
        np.savez(out, image=preprocess(rel), state=state, latent=latent)
        manifest["calibration_cases"].append({
            "image": rel.as_posix(),
            "npz": str(out),
        })

    with open(OUT_DIR / "act_npz_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"test npz files  : {len(manifest['test_cases'])}")
    print(f"calibration npz : {len(manifest['calibration_cases'])}")
    print(f"output dir      : {OUT_DIR}")

if __name__ == "__main__":
    main()
