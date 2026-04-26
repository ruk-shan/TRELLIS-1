"""CLI: generate a textured GLB from a folder of multi-view images.

Usage:
    python generate_glb.py <input_folder> [-o output.glb] [--seed 1] [--texture-size 1024]
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

from PIL import Image  # noqa: E402
from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: E402
from trellis.utils import postprocessing_utils, render_utils  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_images(folder: Path) -> list[Image.Image]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        sys.exit(f"No images found in {folder}")
    print(f"Loaded {len(files)} images: {[p.name for p in files]}")
    return [Image.open(p) for p in files]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_folder", type=Path, help="Folder of multi-view images")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .glb path")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--ss-cfg", type=float, default=7.5)
    parser.add_argument("--slat-steps", type=int, default=12)
    parser.add_argument("--slat-cfg", type=float, default=3.0)
    parser.add_argument(
        "--mode", choices=["stochastic", "multidiffusion"], default="stochastic"
    )
    parser.add_argument("--simplify", type=float, default=0.95)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument(
        "--preview", action="store_true", help="Also save an MP4 preview render"
    )
    args = parser.parse_args()

    images = load_images(args.input_folder)
    out_path = args.output or args.input_folder.parent / f"{args.input_folder.name}.glb"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading TRELLIS-image-large pipeline from models_weights/...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("models_weights/TRELLIS-image-large")
    pipeline.cuda()

    print(f"Running multi-image pipeline (mode={args.mode}, seed={args.seed})...")
    outputs = pipeline.run_multi_image(
        images,
        seed=args.seed,
        formats=["gaussian", "mesh"],
        sparse_structure_sampler_params={
            "steps": args.ss_steps,
            "cfg_strength": args.ss_cfg,
        },
        slat_sampler_params={"steps": args.slat_steps, "cfg_strength": args.slat_cfg},
        mode=args.mode,
    )

    gs = outputs["gaussian"][0]
    mesh = outputs["mesh"][0]

    if args.preview:
        preview_path = out_path.with_suffix(".mp4")
        print(f"Rendering preview video → {preview_path}")
        color = render_utils.render_video(gs, num_frames=120)["color"]
        normal = render_utils.render_video(mesh, num_frames=120)["normal"]
        frames = [np.concatenate([c, n], axis=1) for c, n in zip(color, normal)]
        imageio.mimsave(str(preview_path), frames, fps=15)

    print(f"Extracting textured GLB (texture_size={args.texture_size})...")
    glb = postprocessing_utils.to_glb(
        gs, mesh, simplify=args.simplify, texture_size=args.texture_size, verbose=False
    )
    glb.export(str(out_path))
    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
