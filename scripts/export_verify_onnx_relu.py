import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
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
REFERENCE_JSON = Path("reference.json")
ONNX_PATH = Path("proj57_relu.onnx")
STATE = [0.0, 0.0]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class ACTDeterministicWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, state, latent):
        batch_size = images.shape[0]

        vision_features = self.model.vision_encoder(images)
        state_features = self.model.state_encoder(state)
        latent_features = self.model.latent_proj(latent).unsqueeze(1)

        encoder_in = torch.cat([latent_features, state_features, vision_features], dim=1)

        seq_len = encoder_in.shape[1]
        pos_embed = self.model.encoder_pos_embed.weight[:seq_len].unsqueeze(0)

        encoder_out = self.model.encoder(encoder_in, pos_embed=pos_embed)

        decoder_pos_embed = self.model.decoder_pos_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        decoder_in = torch.zeros(
            batch_size,
            self.model.config.action_chunk_size,
            self.model.config.hidden_dim,
            device=images.device,
        ) + decoder_pos_embed

        decoder_out = self.model.decoder(
            decoder_in,
            encoder_out,
            decoder_pos_embed=decoder_pos_embed,
            encoder_pos_embed=pos_embed,
        )

        action_pred = self.model.action_head(decoder_out)
        return action_pred

def load_checkpoint(path, device):
    ckpt = torch.load(str(path), map_location=device, weights_only=False)

    model = ACTModel(ACTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    latent_mu = ckpt["inference_latent_mu"]
    if latent_mu.ndim == 1:
        latent_mu = latent_mu.unsqueeze(0)
    latent_mu = latent_mu.to(device=device, dtype=torch.float32)

    return model, latent_mu

def replace_gelu_with_relu(module):
    import torch.nn as nn
    import torch.nn.functional as F

    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, nn.ReLU())
        else:
            replace_gelu_with_relu(child)

    if hasattr(module, "activation") and module.activation is F.gelu:
        module.activation = F.relu

def load_stats(dataset_dir):
    stats_path = dataset_dir / "meta" / "stats.json"
    with open(stats_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return {
        "state_q01": np.array(raw["observation.state"]["q01"], dtype=np.float32),
        "state_q99": np.array(raw["observation.state"]["q99"], dtype=np.float32),
        "action_q01": np.array(raw["action"]["q01"], dtype=np.float32),
        "action_q99": np.array(raw["action"]["q99"], dtype=np.float32),
    }

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    t = IMAGE_TRANSFORM(img)
    t = t.unsqueeze(0).unsqueeze(0)
    return t.numpy().astype(np.float32)

def normalize_state(state, stats):
    s = np.array(state, dtype=np.float32).reshape(1, -1)
    d = stats["state_q99"] - stats["state_q01"]
    d = np.where(d == 0, 1e-8, d)
    return (2.0 * (s - stats["state_q01"]) / d - 1.0).astype(np.float32)

def denormalize_action(action, stats):
    d = stats["action_q99"] - stats["action_q01"]
    d = np.where(d == 0, 1e-8, d)
    return ((action + 1.0) / 2.0 * d + stats["action_q01"]).astype(np.float32)

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

    model, latent_mu = load_checkpoint(MODEL_PATH, device)
    replace_gelu_with_relu(model)
    wrapper = ACTDeterministicWrapper(model).eval()

    dummy_image = torch.zeros(1, 1, 3, 224, 224, dtype=torch.float32, device=device)
    dummy_state = torch.zeros(1, 2, dtype=torch.float32, device=device)
    dummy_latent = latent_mu.clone()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_image, dummy_state, dummy_latent),
            str(ONNX_PATH),
            input_names=["image", "state", "latent"],
            output_names=["action"],
            opset_version=18,
            do_constant_folding=True,
        )

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print(f"exported: {ONNX_PATH}")

    stats = load_stats(DATASET_DIR)

    with open(REFERENCE_JSON, "r", encoding="utf-8") as f:
        reference = json.load(f)

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    state_np = normalize_state(STATE, stats)
    latent_np = latent_mu.detach().cpu().numpy().astype(np.float32)

    all_ok = True

    for case in reference["cases"]:
        image_path = Path(case["image_path"])
        image_np = preprocess_image(image_path)

        action_norm = sess.run(
            ["action"],
            {
                "image": image_np,
                "state": state_np,
                "latent": latent_np,
            },
        )[0][0]

        action_denorm = denormalize_action(action_norm, stats)
        ref_denorm = np.array(case["action_denorm"], dtype=np.float32)

        first_step = action_denorm[0]
        direction = turn_direction(first_step)
        max_abs_diff = float(np.max(np.abs(action_denorm - ref_denorm)))

        print(
            f"{image_path.name}: "
            f"left_vel={first_step[0]:+.6f}, "
            f"right_vel={first_step[1]:+.6f}, "
            f"gripper={first_step[2]:+.6f}, "
            f"direction={direction}, "
            f"max_abs_diff={max_abs_diff:.8f}"
        )

        if direction != case["direction"] or max_abs_diff > 1e-3:
            all_ok = False

    if not all_ok:
        raise SystemExit("ONNX verification failed")

    print("onnx verification passed")

if __name__ == "__main__":
    main()
