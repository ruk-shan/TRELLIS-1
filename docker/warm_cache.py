"""Warm offline caches at image-build time.

Runs inside `docker build`, with network available. Bakes the auxiliary
models the TRELLIS image-to-3D pipeline pulls on first use:

- DINOv2 (dinov2_vitl14_reg): fetched via torch.hub by
  trellis/pipelines/trellis_image_to_3d.py. Cached under TORCH_HOME.
- rembg u2net: fetched on first `rembg.new_session('u2net')` call from the
  same pipeline. Cached under U2NET_HOME.

The TRELLIS-image-large checkpoints are baked in via `COPY . /workspace/`,
so no download is needed for them.
"""
import os

os.environ.setdefault("TORCH_HOME", "/workspace/.cache/torch")
os.environ.setdefault("U2NET_HOME", "/workspace/.cache/u2net")
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
os.makedirs(os.environ["U2NET_HOME"], exist_ok=True)

import torch  # noqa: E402

torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True)

import rembg  # noqa: E402

rembg.new_session("u2net")

print("[warm-cache] DINOv2 + rembg u2net cached under /workspace/.cache")
