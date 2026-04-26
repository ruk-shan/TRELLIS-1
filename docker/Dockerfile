# syntax=docker/dockerfile:1.7
#
# TRELLIS — portable CUDA build.
# Pinned: CUDA 11.8, cuDNN 8, Ubuntu 22.04, Python 3.10, PyTorch 2.4.0.
#
# Build (run from the repo root — context is `.`, Dockerfile lives under docker/):
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t trellis:cu118 .
#
# The image is built to run fully offline:
#   - TRELLIS-image-large checkpoints baked in via COPY (~3.1 GB)
#   - DINOv2 + rembg u2net pre-downloaded into /workspace/.cache during build
#   - HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE set so any stray network call fails
#     fast instead of hanging
# Build-time network IS required (apt, pip, torch.hub). Runtime network is not.
#
# Run (offline; mount input/output if you want host-visible I/O):
#   docker run --rm --gpus all -p 7860:7860 \
#     -v $PWD/input:/workspace/input \
#     -v $PWD/output:/workspace/output \
#     trellis:cu118
#
# Build-context exclusions live in docker/Dockerfile.dockerignore — BuildKit
# auto-discovers <dockerfile>.dockerignore next to the Dockerfile.

ARG CUDA_VERSION=11.8.0
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/root/.local/bin:/opt/venv/bin:${PATH} \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX" \
    MAX_JOBS=4 \
    TORCH_HOME=/workspace/.cache/torch \
    U2NET_HOME=/workspace/.cache/u2net

# System deps: build chain, Python 3.10, GL libs for open3d/nvdiffrast/pyvista, ffmpeg for imageio.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ninja-build cmake git ca-certificates curl \
        python3.10 python3.10-dev python3.10-venv python3-pip \
        libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 \
        libegl1 libgles2 libglu1-mesa \
        ffmpeg libjpeg-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# uv as the installer/resolver.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv

# Dedicated venv so the image isn't tied to system-Python conventions.
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv -p 3.10 ${VIRTUAL_ENV}

WORKDIR /workspace

# ---- PyTorch (cu118) -------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.4.0 torchvision==0.19.0

# ---- Pure-Python deps (--basic + --demo from setup.sh) ---------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        "numpy<2" packaging wheel setuptools \
        pillow imageio imageio-ffmpeg tqdm easydict \
        opencv-python-headless scipy ninja rembg onnxruntime \
        trimesh open3d xatlas pyvista pymeshfix igraph transformers \
        gradio==4.44.1 gradio_litmodel3d==0.0.1 \
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# ---- Prebuilt CUDA wheels --------------------------------------------------
# xformers wheel matched to torch 2.4.0 + cu118 (per setup.sh).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-url https://download.pytorch.org/whl/cu118 \
        xformers==0.0.27.post2

# spconv for CUDA 11.x (native algo used by the pipeline).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install spconv-cu118

# Kaolin: upstream publishes a torch-2.4.0 wheel only against cu121.
# It loads fine against torch+cu118 in practice — same path setup.sh takes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# ---- CUDA extensions built from source (need torch + nvcc visible) ---------
# `--no-build-isolation` lets setup.py see the torch we just installed.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-build-isolation flash-attn

RUN git clone --depth 1 https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast \
    && uv pip install --no-build-isolation /tmp/nvdiffrast \
    && rm -rf /tmp/nvdiffrast

RUN git clone --depth 1 --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast \
    && uv pip install --no-build-isolation /tmp/diffoctreerast \
    && rm -rf /tmp/diffoctreerast

RUN git clone --depth 1 https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting \
    && uv pip install --no-build-isolation /tmp/mip-splatting/submodules/diff-gaussian-rasterization/ \
    && rm -rf /tmp/mip-splatting

# ---- Project source --------------------------------------------------------
# Copied last so editing app code doesn't bust the heavy CUDA layers above.
COPY . /workspace/

# Local extension shipped inside this repo.
RUN uv pip install --no-build-isolation /workspace/extensions/vox2seq

# ---- Warm offline caches ---------------------------------------------------
# Pre-download DINOv2 (torch.hub) + rembg u2net so the container never reaches
# out at run time. Build-time network is required for this layer.
RUN python /workspace/docker/warm_cache.py

# Lock down stray network access at run time. Any HF / transformers code that
# tries to phone home will fail fast instead of hanging on a DNS timeout.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 7860
CMD ["python", "app.py"]
