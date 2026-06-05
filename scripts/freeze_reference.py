import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from act.configuration_act import ACTConfig
from act.modeling_act import ACTModel

MODEL_PATH = Path("output/train/model.pt")
DATASET_DIR = Path("output/dataset")
IMAGE_PATHS = [
    "videos/observation.images.fpv/chunk-000/frame_000000.jpg",
    "videos/observation.images.fpv/chunk-000/frame_000227.jpg",
]
STATE = [0.0, 0.0]
OUT_JSON = Path("reference.json")

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def load_checkpoint(path, device):
    ckpt = torch.load(str(path), map_location=device, weights_only=False)

    if "model_state_dict" not in ckpt:
        raise RuntimeError("checkpoint 缺少 model_state_dict")
    if "config" not in ckpt:
        raise RuntimeError("checkpoint 缺少 config")

    model = ACTModel(ACTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state_dict"])

    inf_mu = ckpt.get("inference_latent_mu")
    inf_log_sigma = ckpt.get("inference_latent_log_sigma")
    if inf_mu is None or inf_log_sigma is None:
        raise RuntimeError("checkpoint 缺少 inference_latent_mu / inference_latent_log_sigma")

    model.set_inference_latent(inf_mu, inf_log_sigma)
    model = model.to(device).eval()
    return model

def load_stats(dataset_dir):
    stats_path = dataset_dir / "meta" / "stats.json"
    with open(stats_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return {
        "state_q01": raw["observation.state"]["q01"],
        "state_q99": raw["observation.state"]["q99"],
        "action_q01": raw["action"]["q01"],
        "action_q99": raw["action"]["q99"],
    }

def process_image(path, device):
    img = Image.open(path).convert("RGB")
    t = IMAGE_TRANSFORM(img)
    return t.unsqueeze(0).unsqueeze(0).to(device)

def normalize_state(state, stats, device):
    q01 = torch.tensor(stats["state_q01"], dtype=torch.float32, device=device)
    q99 = torch.tensor(stats["state_q99"], dtype=torch.float32, device=device)
    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    d = q99 - q01
    d = torch.where(d == 0, torch.tensor(1e-8, device=device), d)
    return 2 * (s - q01) / d - 1

def denormalize_action(action, stats, device):
    q01 = torch.tensor(stats["action_q01"], dtype=torch.float32, device=device)
    q99 = torch.tensor(stats["action_q99"], dtype=torch.float32, device=device)
    d = q99 - q01
    d = torch.where(d == 0, torch.tensor(1e-8, device=device), d)
    return (action + 1) / 2 * d + q01

def turn_direction(first_step):
    lv, rv = first_step[0], first_step[1]
    if lv < rv:
        return "left"
    if lv > rv:
        return "right"
    return "straight"

def main():
    device = "cpu"
    torch.manual_seed(0)
    torch.set_num_threads(1)

    model = load_checkpoint(MODEL_PATH, device)
    stats = load_stats(DATASET_DIR)

    model._sample_latent = lambda mu, log_sigma_x2: mu

    latent_mu = model._inference_latent_mu.detach().cpu().reshape(-1).tolist()
    results = {
        "model_path": str(MODEL_PATH),
        "state_raw": STATE,
        "latent_mode": "use_inference_latent_mu_directly",
        "latent_mu": latent_mu,
        "cases": [],
    }

    state_tensor = normalize_state(STATE, stats, device)
    state_norm = state_tensor[0].detach().cpu().tolist()

    for rel_path in IMAGE_PATHS:
        image_path = DATASET_DIR / rel_path
        image_tensor = process_image(image_path, device)

        with torch.no_grad():
            output = model.forward(
                image_tensor,
                state_tensor,
                action_target=None,
                infer_cvae=True,
            )
            action_norm = output["action"][0]
            action_denorm = denormalize_action(action_norm, stats, device)

        first_step = action_denorm[0].detach().cpu().tolist()
        direction = turn_direction(first_step)

        case = {
            "image_path": str(image_path),
            "state_norm": state_norm,
            "action_norm": action_norm.detach().cpu().tolist(),
            "action_denorm": action_denorm.detach().cpu().tolist(),
            "first_step": {
                "left_vel": first_step[0],
                "right_vel": first_step[1],
                "gripper_target": first_step[2],
            },
            "direction": direction,
        }
        results["cases"].append(case)

        print(
            f"{image_path.name}: "
            f"left_vel={first_step[0]:+.6f}, "
            f"right_vel={first_step[1]:+.6f}, "
            f"gripper={first_step[2]:+.6f}, "
            f"direction={direction}"
        )

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nreference saved to: {OUT_JSON}")

if __name__ == "__main__":
    main()
