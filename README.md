# TRELLIS — Image to 3D (inference)

A slimmed, inference-only fork of [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) for turning images of an object into a textured 3D model. Generate from a single image or several views, preview it, and export as **GLB** or **OBJ**.

## Clone

This repo uses a git submodule ([FlexiCubes](https://github.com/MaxtirError/FlexiCubes), used for mesh extraction), so clone with `--recurse-submodules`:

```sh
git clone --recurse-submodules <repo-url>
# already cloned without it? run:
git submodule update --init --recursive
```

Without the submodule, `trellis/representations/mesh/flexicubes/` is empty and mesh extraction fails.

## Requirements

- NVIDIA GPU (≥16 GB memory recommended)
- The `TRELLIS-image-large` weights, downloaded locally:
  ```sh
  huggingface-cli download microsoft/TRELLIS-image-large \
    --local-dir models_weights/TRELLIS-image-large
  ```

## Run with Docker (recommended)

Builds a Blackwell / RTX 50-series–capable CUDA 12.8 stack. Requires the NVIDIA Container Toolkit.

```sh
docker compose up --build      # first run: builds, then starts the demo
docker compose up              # later runs: reuse the built image
```

Then open <http://localhost:7860>. See [docker/README.md](docker/README.md) for details.

## Run locally

Activate the virtualenv, then launch:

```sh
source .venv/bin/activate
python app.py
```

Weights are read from [models_weights/TRELLIS-image-large/](models_weights/TRELLIS-image-large/). Attention uses the `xformers` backend and `native` spconv (set automatically).

## Web demo — [app.py](app.py)

The Gradio app handles **single-image** and **multi-image** (multi-view) input via tabs. Upload an image (or a set of views of the same object), click **Generate** to preview, then **Extract** to download the mesh as **GLB** or **OBJ**.

- If an image has an alpha channel it is used as the object mask; otherwise `rembg` removes the background.
- For multi-view input, the views are fused with the multidiffusion / stochastic algorithm (selectable in the settings).

## CLI — [generate_glb.py](generate_glb.py)

Turn a folder of images of one object into a textured mesh:

```sh
python generate_glb.py input/screw_A                 # → input/screw_A.glb

python generate_glb.py input/screw_A \
  -o output/screw_A.obj \
  --format obj \        # glb (default) or obj
  --seed 1 \
  --texture-size 1024 \
  --preview             # also writes an .mp4 preview render
```

Run `python generate_glb.py --help` for the full flag list.

> **Note on OBJ:** GLB is a single self-contained file. OBJ exports three files that must stay together — `.obj` + `.mtl` + a texture `.png`.

## Credits & license

Based on **TRELLIS: Structured 3D Latents for Scalable and Versatile 3D Generation** ([paper](https://arxiv.org/abs/2412.01506), [project page](https://microsoft.github.io/TRELLIS/)). Code is licensed under the [MIT License](LICENSE); some submodules carry their own licenses.

```bibtex
@article{xiang2024structured,
    title   = {Structured 3D Latents for Scalable and Versatile 3D Generation},
    author  = {Xiang, Jianfeng and Lv, Zelong and Xu, Sicheng and Deng, Yu and Wang, Ruicheng and Zhang, Bowen and Chen, Dong and Tong, Xin and Yang, Jiaolong},
    journal = {arXiv preprint arXiv:2412.01506},
    year    = {2024}
}
```
