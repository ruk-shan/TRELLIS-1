# syntax=docker/dockerfile:1.7
#
# TRELLIS — portable CUDA build (Blackwell / RTX 50-series capable).
# Pinned: CUDA 12.8, cuDNN 9, Ubuntu 22.04, Python 3.10, PyTorch 2.7.0+cu128.
#
# Build (run from the repo root — context is `.`, Dockerfile lives under docker/):
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t trellis:cu128 .
#
# The image runs with auxiliary models baked in, but the TRELLIS checkpoints are
# NOT in the image — they are mounted at runtime:
#   - TRELLIS-image-large checkpoints are EXCLUDED from the build context (see
#     docker/Dockerfile.dockerignore) and must be mounted at run time (~3.1 GB)
#   - DINOv2 + rembg u2net pre-downloaded into /workspace/.cache during build
#   - HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE set so any stray network call fails
#     fast instead of hanging
# Build-time network IS required (apt, pip, torch.hub). Runtime network is not.
#
# Run (mount the weights + input/output so they stay host-local):
#   docker run --rm --gpus all -p 7860:7860 \
#     -v $PWD/models_weights:/workspace/models_weights \
#     -v $PWD/input:/workspace/input \
#     -v $PWD/output:/workspace/output \
#     trellis:cu128
#
# Build-context exclusions live in docker/Dockerfile.dockerignore — BuildKit
# auto-discovers <dockerfile>.dockerignore next to the Dockerfile.

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_HTTP_TIMEOUT=600 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/root/.local/bin:/opt/venv/bin:${PATH} \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="12.0+PTX" \
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

# ---- PyTorch (cu128) -------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.0 torchvision==0.22.0

# ---- Pure-Python deps (--basic + --demo from setup.sh) ---------------------
# The gradio 4.44.1 demo stack is pinned to its 2024-era companions. Left
# unpinned, uv resolves bleeding-edge versions that break the pinned gradio:
#   - transformers 5.x → huggingface_hub >=1.5, which dropped `HfFolder` (gradio's
#     oauth.py imports it at load time → ImportError).
#   - pydantic >=2.11 emits boolean `additionalProperties` in JSON schemas, which
#     gradio_client 1.3.0 can't parse → "argument of type 'bool' is not iterable"
#     500 on every request. fastapi is pinned alongside for schema-gen compat.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        "numpy<2" packaging wheel setuptools \
        pillow imageio imageio-ffmpeg tqdm easydict \
        opencv-python-headless scipy ninja rembg onnxruntime \
        trimesh open3d xatlas pyvista pymeshfix igraph \
        "transformers==4.46.3" "huggingface_hub==0.25.2" \
        gradio==4.44.1 gradio_litmodel3d==0.0.1 \
        "pydantic==2.10.6" "fastapi==0.115.6" \
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# ---- Prebuilt CUDA wheels --------------------------------------------------
# xformers wheel matched to torch 2.7.0 + cu128.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --index-url https://download.pytorch.org/whl/cu128 \
        xformers==0.0.30

# spconv: no cu128 wheel is published — spconv-cu126 is the newest CUDA-12.x
# build and works against the 12.8 runtime via CUDA minor-version forward-compat.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install spconv-cu126

# Kaolin: upstream publishes a torch-2.7.0 wheel for cu128.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html

# ---- CUDA extensions built from source (need torch + nvcc visible) ---------
# `--no-build-isolation` lets setup.py see the torch we just installed.
#
# NOTE: flash-attn is intentionally NOT installed. Every TRELLIS entry point runs
# the `xformers` attention backend (ATTN_BACKEND=xformers), and trellis only imports
# flash_attn when ATTN_BACKEND=flash_attn (see trellis/modules/attention/full_attn.py).
# Worse, xformers 0.0.30 *refuses to import* if an out-of-range flash-attn is present
# (it wants <=2.7.4; current source builds produce 2.8.x). Omitting it avoids the
# conflict and the costly/uncertain flash-attn build on Blackwell. xformers uses its
# own kernels regardless.

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

# ---- Blackwell (sm_120) attention fix --------------------------------------
# xformers 0.0.30 ships FlashAttention v2/v3 kernels built only up to Hopper
# (sm_90), yet its op-selection gates advertise support for every compute
# capability >= 8.0 / 9.0. On Blackwell (sm_120) the dispatcher therefore picks a
# Flash kernel that has no sm_120 image and the process dies with
# "no kernel image is available for execution on the device".
# Its CUTLASS op DOES have working sm_120 kernels (verified for both dense and
# BlockDiagonalMask/varlen attention — the dense + sparse paths TRELLIS uses).
# Bump both Flash ops' minimum compute capability above sm_120 so the dispatcher
# falls back to CUTLASS. (Patches the xformers installed far above; kept here as a
# late layer so it doesn't bust the heavy CUDA build cache.)
RUN FMHA=/opt/venv/lib/python3.10/site-packages/xformers/ops/fmha \
    && sed -i 's/CUDA_MINIMUM_COMPUTE_CAPABILITY = (8, 0)/CUDA_MINIMUM_COMPUTE_CAPABILITY = (13, 0)/' "$FMHA/flash.py" \
    && sed -i 's/CUDA_MINIMUM_COMPUTE_CAPABILITY = (9, 0)/CUDA_MINIMUM_COMPUTE_CAPABILITY = (13, 0)/' "$FMHA/flash3.py" \
    && grep -q 'CUDA_MINIMUM_COMPUTE_CAPABILITY = (13, 0)' "$FMHA/flash.py" \
    && grep -q 'CUDA_MINIMUM_COMPUTE_CAPABILITY = (13, 0)' "$FMHA/flash3.py"

# Lock down stray network access at run time. Any HF / transformers code that
# tries to phone home will fail fast instead of hanging on a DNS timeout.
# GRADIO_SERVER_NAME binds the demo to 0.0.0.0 so it's reachable through the
# published port (Gradio's default 127.0.0.1 would only be visible inside the
# container).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

EXPOSE 7860
CMD ["python", "app.py"]
