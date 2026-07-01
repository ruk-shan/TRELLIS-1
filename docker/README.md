# Docker

Portable, offline-capable build of TRELLIS (Blackwell / RTX 50-series capable).

Pinned stack: **CUDA 12.8 / cuDNN 9 / Ubuntu 22.04 / Python 3.10 / PyTorch 2.7.0+cu128**.
GPU arch targeted: `12.0+PTX` (Blackwell / sm_120). Broaden `TORCH_CUDA_ARCH_LIST`
in the Dockerfile to build a more portable image (at the cost of compile time).

## Files

- [Dockerfile](Dockerfile) — full image definition.
- [Dockerfile.dockerignore](Dockerfile.dockerignore) — build-context exclusions (BuildKit auto-discovers it next to the Dockerfile).
- [warm_cache.py](warm_cache.py) — pre-downloads DINOv2 + rembg `u2net` into `/workspace/.cache` during the image build.

## Prerequisites

- Docker 20.10+ with BuildKit enabled.
- NVIDIA Container Toolkit (`nvidia-docker2` / `nvidia-container-toolkit`).
- Host NVIDIA driver supporting CUDA 12.8 runtime (≥ 525, Blackwell needs ≥ 570).
- TRELLIS-image-large weights downloaded to [../models_weights/TRELLIS-image-large/](../models_weights/TRELLIS-image-large/) — they are **mounted at run time**, not baked into the image (the build context excludes `models_weights`). Download them with:
  ```sh
  huggingface-cli download microsoft/TRELLIS-image-large \
    --local-dir models_weights/TRELLIS-image-large
  ```
- Network access **at build time** (apt, PyPI, torch.hub, GitHub). Not required at run time.

## Quick start (Docker Compose)

The fastest path. From the repo root, with the weights downloaded to `./models_weights`:

```sh
docker compose up --build      # first time: builds the image, then starts the demo
docker compose up              # subsequent runs: reuse the built image
```

Open `http://localhost:7860`. Stop with `Ctrl-C`, or `docker compose down` to remove
the container. Compose (see [../docker-compose.yml](../docker-compose.yml)) wires up the
GPU, publishes port 7860, and bind-mounts `models_weights/`, `input/`, and `output/`.

Run a one-off CLI command through the same service:

```sh
docker compose run --rm trellis python generate_glb.py input/screw_A -o output/screw_A.glb
```

The sections below cover the equivalent raw `docker build` / `docker run` commands.

## Build

Run from the repo root so the build context is the project directory:

```sh
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t trellis:cu128 .
```

First build is slow — expect **20–45 min**. Most of it is `nvdiffrast`, `diffoctreerast`, `diff-gaussian-rasterization`, and `vox2seq` compiling CUDA from source. `MAX_JOBS=4` is set inside the Dockerfile to keep RAM bounded; bump it if your host has plenty of memory.

> `flash-attn` is intentionally **not** installed — TRELLIS runs the `xformers` attention backend, and a mismatched flash-attn version makes `xformers` refuse to import. xformers uses its own kernels, so nothing is lost.

A GPU is **not required at build time** (`FORCE_CUDA=1` lets nvcc target archs without a device). It **is** required at run time.

### Build-arg overrides

```sh
# Different CUDA base (must still be a *-devel image)
docker build -f docker/Dockerfile \
  --build-arg CUDA_VERSION=12.6.3 \
  -t trellis:cu126 .
```

Note that switching CUDA versions also requires changing the torch / xformers / spconv lines inside the Dockerfile to matching wheels.

## Run

The image runs **offline** — DINOv2 and rembg `u2net` are baked in, and the TRELLIS
weights are supplied via a runtime mount (they are not in the image). `HF_HUB_OFFLINE=1`
and `TRANSFORMERS_OFFLINE=1` are set so any accidental network call fails fast instead
of hanging. Mount `models_weights` on every run.

### Gradio demo (default `CMD`)

The `app.py` demo handles both single-image and multi-image (multi-view) input via tabs.

```sh
docker run --rm --gpus all -p 7860:7860 \
  -v $PWD/models_weights:/workspace/models_weights \
  -v $PWD/input:/workspace/input \
  -v $PWD/output:/workspace/output \
  trellis:cu128
```

Open `http://localhost:7860`.

### CLI: folder of multi-view images → textured GLB/OBJ

```sh
docker run --rm --gpus all \
  -v $PWD/models_weights:/workspace/models_weights \
  -v $PWD/input:/workspace/input \
  -v $PWD/output:/workspace/output \
  trellis:cu128 \
  python generate_glb.py input/screw_A -o output/screw_A.glb --preview
```

See [../generate_glb.py](../generate_glb.py) for the full flag list.

### Verify it's truly offline

```sh
docker run --rm --gpus all --network=none \
  -v $PWD/models_weights:/workspace/models_weights \
  trellis:cu128 \
  python example.py
```

`--network=none` cuts off all networking. The weights mount must still be present.
If anything in the pipeline tries to phone home, it will error here.

## Pointing at a different checkpoint set

The weights always come from a runtime mount, so just change the host path:

```sh
docker run --rm --gpus all -p 7860:7860 \
  -v /path/to/other-weights:/workspace/models_weights \
  -v $PWD/input:/workspace/input \
  -v $PWD/output:/workspace/output \
  trellis:cu128
```

## Image layout

| path | what's there |
|---|---|
| `/opt/venv/` | Python 3.10 venv with all deps |
| `/workspace/` | TRELLIS source (everything `COPY`'d from repo root) |
| `/workspace/models_weights/TRELLIS-image-large/` | checkpoints — **mounted at run time**, not in the image |
| `/workspace/.cache/torch/` | DINOv2 hub cache |
| `/workspace/.cache/u2net/` | rembg ONNX |
| `/usr/local/cuda/` | CUDA 11.8 toolkit (from the base image) |

## Troubleshooting

**`could not select device driver "" with capabilities: [[gpu]]`** — NVIDIA Container Toolkit isn't installed or the Docker daemon wasn't restarted after install.

**Build OOMs during vox2seq / diffoctreerast compile** — lower `MAX_JOBS` in the Dockerfile (default 4). Each parallel `nvcc` invocation can use several GB of RAM.

**`undefined symbol: ... ABI ...` at run time** — usually a torch/extension ABI mismatch. The Dockerfile pins torch, xformers, spconv, and kaolin to one consistent CUDA — if you edited any of those lines, make sure the CUDA tag matches across all of them.

**`no kernel image is available for execution on the device`** — an extension lacks SASS for your GPU's arch. Make sure `TORCH_CUDA_ARCH_LIST` includes your arch (Blackwell/RTX 50-series = `12.0`) and rebuild. If it traces back to `spconv` specifically, the prebuilt `spconv-cu126` wheel may not cover `sm_120`; build spconv from source against your arch as a fallback.

**Want the image to run on a different GPU arch** — edit `TORCH_CUDA_ARCH_LIST` in the Dockerfile before building. Adding more archs increases compile time roughly linearly.

**Build fails at `warm_cache.py` with no network** — `docker build` requires outbound network; only the *runtime* is offline. Build on a connected machine, then `docker save` / `docker load` the image onto the air-gapped target.
