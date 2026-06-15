from pathlib import Path
import numpy as np

root = Path("/workspace/proj57")
inp = root / "starry_inputs"
out = root / "tpu_bf16_relu" / "workspace"
out.mkdir(parents=True, exist_ok=True)

cases = {
    "000000": "frame_000000_input.bin",
    "000227": "frame_000227_input.bin",
}

for tag, image_file in cases.items():
    image = np.fromfile(inp / image_file, dtype=np.float32).reshape(1, 1, 3, 224, 224)
    state = np.fromfile(inp / "state.bin", dtype=np.float32).reshape(1, 2)
    latent = np.fromfile(inp / "latent.bin", dtype=np.float32).reshape(1, 32)

    np.savez(out / f"act_{tag}_input.npz", image=image, state=state, latent=latent)

    print(tag, image.shape, state.shape, latent.shape)
