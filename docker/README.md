# Docker

Portable, offline-capable build of TRELLIS.

Pinned stack: **CUDA 11.8 / cuDNN 8 / Ubuntu 22.04 / Python 3.10 / PyTorch 2.4.0+cu118**.
GPU archs targeted: `7.5;8.0;8.6;8.9;9.0+PTX` (Turing → Hopper).

## Files

- [Dockerfile](Dockerfile) — full image definition.
- [Dockerfile.dockerignore](Dockerfile.dockerignore) — build-context exclusions (BuildKit auto-discovers it next to the Dockerfile).
- [warm_cache.py](warm_cache.py) — pre-downloads DINOv2 + rembg `u2net` into `/workspace/.cache` during the image build.

## Prerequisites

- Docker 20.10+ with BuildKit enabled.
- NVIDIA Container Toolkit (`nvidia-docker2` / `nvidia-container-toolkit`).
- Host NVIDIA driver supporting CUDA 11.8 runtime (≥ 525.60.13).
- TRELLIS-image-large weights present at [../models_weights/TRELLIS-image-large/](../models_weights/TRELLIS-image-large/) — they are baked into the image during build.
- Network access **at build time** (apt, PyPI, torch.hub, GitHub). Not required at run time.

## Build

Run from the repo root so the build context is the project directory:

```sh
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t trellis:cu118 .
```

First build is slow — expect **30–60 min**. Most of it is `flash-attn`, `nvdiffrast`, `diffoctreerast`, `diff-gaussian-rasterization`, and `vox2seq` compiling for five GPU architectures. `MAX_JOBS=4` is set inside the Dockerfile to keep RAM bounded; bump it if your host has plenty of memory.

A GPU is **not required at build time** (`FORCE_CUDA=1` lets nvcc target archs without a device). It **is** required at run time.

### Build-arg overrides

```sh
# Different CUDA base (must still be a *-devel image)
docker build -f docker/Dockerfile \
  --build-arg CUDA_VERSION=12.1.1 \
  -t trellis:cu121 .
```

Note that switching CUDA versions also requires changing the torch / xformers / spconv lines inside the Dockerfile to matching wheels.

## Run

The image runs **fully offline** — TRELLIS weights, DINOv2, and rembg `u2net` are all baked in. `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set so any accidental network call fails fast instead of hanging.

### Single-image Gradio demo (default `CMD`)

```sh
docker run --rm --gpus all -p 7860:7860 \
  -v $PWD/input:/workspace/input \
  -v $PWD/output:/workspace/output \
  trellis:cu118
```

Open `http://localhost:7860`.

### Multi-image Gradio demo

```sh
docker run --rm --gpus all -p 7860:7860 \
  -v $PWD/input:/workspace/input \
  -v $PWD/output:/workspace/output \
  trellis:cu118 \
  python multi_image_app.py
```

### CLI: folder of multi-view images → textured GLB

```sh
docker run --rm --gpus all \
  -v $PWD/input:/workspace/input \
  -v $PWD/output:/workspace/output \
  trellis:cu118 \
  python generate_glb.py input/screw_A -o output/screw_A.glb --preview
```

See [../generate_glb.py](../generate_glb.py) for the full flag list.

### Verify it's truly offline

```sh
docker run --rm --gpus all --network=none trellis:cu118 \
  python example.py
```

`--network=none` cuts off all networking. If anything in the pipeline still tries to phone home, it will error here.

## Overriding the baked weights

If you want to point at a different checkpoint set without rebuilding:

```sh
docker run --rm --gpus all -p 7860:7860 \
  -v /path/to/other-weights:/workspace/models_weights \
  -v $PWD/input:/workspace/input \
  -v $PWD/output:/workspace/output \
  trellis:cu118
```

The mount shadows the baked `/workspace/models_weights/`.

## Image layout

| path | what's there |
|---|---|
| `/opt/venv/` | Python 3.10 venv with all deps |
| `/workspace/` | TRELLIS source (everything `COPY`'d from repo root) |
| `/workspace/models_weights/TRELLIS-image-large/` | baked checkpoints (~3.1 GB) |
| `/workspace/.cache/torch/` | DINOv2 hub cache |
| `/workspace/.cache/u2net/` | rembg ONNX |
| `/usr/local/cuda/` | CUDA 11.8 toolkit (from the base image) |

## Troubleshooting

**`could not select device driver "" with capabilities: [[gpu]]`** — NVIDIA Container Toolkit isn't installed or the Docker daemon wasn't restarted after install.

**Build OOMs during flash-attn / vox2seq compile** — lower `MAX_JOBS` in the Dockerfile (default 4). Each parallel `nvcc` invocation can use several GB of RAM.

**`undefined symbol: ... ABI ...` at run time** — usually a torch/extension ABI mismatch. The Dockerfile pins torch, xformers, spconv, and kaolin to one consistent CUDA — if you edited any of those lines, make sure the CUDA tag matches across all of them.

**Want the image to run on a different GPU arch** — edit `TORCH_CUDA_ARCH_LIST` in the Dockerfile before building. Adding more archs increases compile time roughly linearly.

**Build fails at `warm_cache.py` with no network** — `docker build` requires outbound network; only the *runtime* is offline. Build on a connected machine, then `docker save` / `docker load` the image onto the air-gapped target.
