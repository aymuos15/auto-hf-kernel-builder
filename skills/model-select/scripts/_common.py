from __future__ import annotations

import glob
from pathlib import Path


def model_slug(model_id):
    return model_id.replace("/", "__")


def first_asset(assets_dir):
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        hits = sorted(glob.glob(str(Path(assets_dir) / ext)))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(f"no image asset in {assets_dir}")


def load_model_and_inputs(cfg):
    import torch
    from transformers import AutoModel, AutoProcessor
    from PIL import Image

    mid = cfg["model"]["id"]
    assets = Path(cfg["model"].get("example_input", {}).get("assets_dir", "assets"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(mid)
    model = AutoModel.from_pretrained(mid, dtype=torch.float32).to(device).eval()
    image = Image.open(first_asset(assets)).convert("RGB")
    return model, processor(images=image, return_tensors="pt").to(device), device


def run_forward(model, inputs):
    import torch

    with torch.no_grad():
        return model.get_image_embeddings(inputs["pixel_values"])
