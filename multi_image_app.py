"""Multi-image to textured 3D model app powered by TRELLIS.

Run:
    source .venv/bin/activate
    python multi_image_app.py
"""

import os

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import shutil
from typing import List, Tuple

import gradio as gr
from gradio_litmodel3d import LitModel3D
import imageio
import numpy as np
import torch
from PIL import Image
from easydict import EasyDict as edict

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import postprocessing_utils, render_utils


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_multi")
os.makedirs(TMP_DIR, exist_ok=True)

pipeline: TrellisImageTo3DPipeline | None = None


def session_dir(req: gr.Request) -> str:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir, ignore_errors=True)


def preprocess_images(images: List[Tuple[Image.Image, str]] | None) -> List[Image.Image]:
    if not images:
        return []
    pil_images = [img[0] if isinstance(img, tuple) else img for img in images]
    return [pipeline.preprocess_image(img) for img in pil_images]


def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        "gaussian": {
            **gs.init_params,
            "_xyz": gs._xyz.cpu().numpy(),
            "_features_dc": gs._features_dc.cpu().numpy(),
            "_scaling": gs._scaling.cpu().numpy(),
            "_rotation": gs._rotation.cpu().numpy(),
            "_opacity": gs._opacity.cpu().numpy(),
        },
        "mesh": {
            "vertices": mesh.vertices.cpu().numpy(),
            "faces": mesh.faces.cpu().numpy(),
        },
    }


def unpack_state(state: dict) -> Tuple[Gaussian, edict]:
    gs = Gaussian(
        aabb=state["gaussian"]["aabb"],
        sh_degree=state["gaussian"]["sh_degree"],
        mininum_kernel_size=state["gaussian"]["mininum_kernel_size"],
        scaling_bias=state["gaussian"]["scaling_bias"],
        opacity_bias=state["gaussian"]["opacity_bias"],
        scaling_activation=state["gaussian"]["scaling_activation"],
    )
    gs._xyz = torch.tensor(state["gaussian"]["_xyz"], device="cuda")
    gs._features_dc = torch.tensor(state["gaussian"]["_features_dc"], device="cuda")
    gs._scaling = torch.tensor(state["gaussian"]["_scaling"], device="cuda")
    gs._rotation = torch.tensor(state["gaussian"]["_rotation"], device="cuda")
    gs._opacity = torch.tensor(state["gaussian"]["_opacity"], device="cuda")
    mesh = edict(
        vertices=torch.tensor(state["mesh"]["vertices"], device="cuda"),
        faces=torch.tensor(state["mesh"]["faces"], device="cuda"),
    )
    return gs, mesh


def get_seed(randomize: bool, seed: int) -> int:
    return int(np.random.randint(0, MAX_SEED)) if randomize else int(seed)


def generate(
    images: List[Tuple[Image.Image, str]],
    seed: int,
    ss_guidance: float,
    ss_steps: int,
    slat_guidance: float,
    slat_steps: int,
    multiimage_algo: str,
    req: gr.Request,
):
    if not images:
        raise gr.Error("Upload at least 2 views of the same object before generating.")
    if len(images) < 2:
        raise gr.Error("Multi-image mode needs at least 2 images. Add more views.")

    user_dir = session_dir(req)
    pil_images = [img[0] if isinstance(img, tuple) else img for img in images]

    outputs = pipeline.run_multi_image(
        pil_images,
        seed=seed,
        formats=["gaussian", "mesh"],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": int(ss_steps),
            "cfg_strength": float(ss_guidance),
        },
        slat_sampler_params={
            "steps": int(slat_steps),
            "cfg_strength": float(slat_guidance),
        },
        mode=multiimage_algo,
    )

    gs = outputs["gaussian"][0]
    mesh = outputs["mesh"][0]

    color_frames = render_utils.render_video(gs, num_frames=120)["color"]
    normal_frames = render_utils.render_video(mesh, num_frames=120)["normal"]
    video = [np.concatenate([c, n], axis=1) for c, n in zip(color_frames, normal_frames)]
    video_path = os.path.join(user_dir, "preview.mp4")
    imageio.mimsave(video_path, video, fps=15)

    state = pack_state(gs, mesh)
    torch.cuda.empty_cache()
    return state, video_path


def extract_glb(state: dict, simplify: float, texture_size: int, req: gr.Request):
    if state is None:
        raise gr.Error("Generate a 3D preview first.")
    user_dir = session_dir(req)
    gs, mesh = unpack_state(state)
    glb = postprocessing_utils.to_glb(
        gs, mesh, simplify=float(simplify), texture_size=int(texture_size), verbose=False
    )
    glb_path = os.path.join(user_dir, "model.glb")
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path


def example_sets() -> List[List[str]]:
    """Return groups of 3-image example sets from assets/example_multi_image."""
    root = "assets/example_multi_image"
    if not os.path.isdir(root):
        return []
    cases = sorted({fn.split("_")[0] for fn in os.listdir(root)})
    sets = []
    for case in cases:
        files = sorted(
            os.path.join(root, fn)
            for fn in os.listdir(root)
            if fn.startswith(f"{case}_")
        )
        if files:
            sets.append(files)
    return sets


with gr.Blocks(
    title="TRELLIS Multi-Image → Textured 3D",
    delete_cache=(600, 600),
) as demo:
    gr.Markdown(
        """
        # TRELLIS — Multi-Image to Textured 3D Model
        Upload **2 or more images of the same object from different viewpoints** and generate a
        textured 3D mesh. Workflow: **Generate** → preview the rotating render → **Extract GLB** →
        download the textured `.glb` (open in Blender, three.js, Unity, etc.).

        Tip: clean cut-outs work best. Images with an alpha channel are used directly; otherwise
        `rembg` removes the background automatically.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_gallery = gr.Gallery(
                label="Multi-View Images",
                format="png",
                type="pil",
                height=320,
                columns=3,
                interactive=True,
            )

            with gr.Accordion("Generation Settings", open=False):
                with gr.Row():
                    seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                gr.Markdown("**Stage 1 — Sparse Structure**")
                with gr.Row():
                    ss_guidance = gr.Slider(0.0, 10.0, label="Guidance", value=7.5, step=0.1)
                    ss_steps = gr.Slider(1, 50, label="Steps", value=12, step=1)
                gr.Markdown("**Stage 2 — Structured Latent**")
                with gr.Row():
                    slat_guidance = gr.Slider(0.0, 10.0, label="Guidance", value=3.0, step=0.1)
                    slat_steps = gr.Slider(1, 50, label="Steps", value=12, step=1)
                multiimage_algo = gr.Radio(
                    ["stochastic", "multidiffusion"],
                    label="Multi-Image Fusion",
                    value="stochastic",
                )

            with gr.Accordion("GLB Extraction Settings", open=False):
                mesh_simplify = gr.Slider(
                    0.9, 0.98, label="Mesh Simplification", value=0.95, step=0.01
                )
                texture_size = gr.Slider(
                    512, 2048, label="Texture Resolution", value=1024, step=512
                )

            with gr.Row():
                generate_btn = gr.Button("Generate 3D", variant="primary")
                extract_btn = gr.Button("Extract Textured GLB", interactive=False)

        with gr.Column(scale=1):
            video_output = gr.Video(
                label="Preview (color + normals)", autoplay=True, loop=True, height=320
            )
            model_output = LitModel3D(label="Textured GLB", exposure=10.0, height=380)
            glb_download = gr.DownloadButton(label="Download .glb", interactive=False)

    state_buf = gr.State()

    examples = example_sets()
    if examples:
        gr.Examples(
            examples=[[paths] for paths in examples],
            inputs=[image_gallery],
            label="Example multi-view sets",
            examples_per_page=8,
        )

    image_gallery.upload(
        preprocess_images,
        inputs=[image_gallery],
        outputs=[image_gallery],
    )

    generate_btn.click(
        get_seed, inputs=[randomize_seed, seed], outputs=[seed]
    ).then(
        generate,
        inputs=[
            image_gallery,
            seed,
            ss_guidance,
            ss_steps,
            slat_guidance,
            slat_steps,
            multiimage_algo,
        ],
        outputs=[state_buf, video_output],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[extract_btn],
    )

    extract_btn.click(
        extract_glb,
        inputs=[state_buf, mesh_simplify, texture_size],
        outputs=[model_output, glb_download],
    ).then(
        lambda: gr.DownloadButton(interactive=True),
        outputs=[glb_download],
    )

    demo.unload(end_session)


if __name__ == "__main__":
    print("Loading TRELLIS-image-large pipeline (first run downloads ~5GB)...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    print("Pipeline ready.")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
