"""Local-only Gradio reviewer demo; no remote inference or public share link."""

import argparse, tempfile, threading
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image
from rolehos.runtime import load_model, predict, overlay, NAMES, PALETTE
from rolehos import ROOT

LOCK = threading.Lock()


@lru_cache(maxsize=3)
def get_model(seed, device):
    return load_model("full", int(seed), device)[0]


def run_image(image, seed=0, device="cpu"):
    if image is None:
        raise ValueError("Upload an RGB image or choose a test example.")
    rgb = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    with LOCK:
        mask = predict(get_model(int(seed), device), rgb)
    ov, col = overlay(rgb, mask)
    tmp = Path(tempfile.mkdtemp(prefix="rolehos_demo_")) / "mask.png"
    Image.fromarray(mask).save(tmp)
    description = "448 × 448 label mask; 0 BG, 1 LH, 2 RH, 3 LO, 4 RO, 5 TO. "
    description += (
        "Demo latency includes Python/UI overhead and is not the paper FPS benchmark."
    )
    return Image.fromarray(ov), Image.fromarray(col), str(tmp), description


def build_app(device="cpu"):
    import gradio as gr

    examples = sorted((ROOT / "examples/images").glob("*.jpg"))
    with gr.Blocks(title="RoleHOS — reviewer demo") as demo:
        gr.Markdown(
            "# RoleHOS: egocentric hand–object segmentation\nUpload an image or choose a curated test example. Inference is local; no image is sent to an external API."
        )
        with gr.Row():
            source = gr.Image(type="pil", sources=["upload"], label="RGB input")
            output = gr.Image(type="pil", label="Segmentation overlay (448 × 448)")
            color = gr.Image(type="pil", label="Role mask")
        seed = gr.Radio(
            [0, 1, 2], value=0, label="Published checkpoint seed (not an ensemble)"
        )
        button = gr.Button("Run RoleHOS", variant="primary")
        download = gr.File(label="Download uint8 label PNG")
        status = gr.Textbox(label="Output information")
        gr.Markdown(
            "**Legend:** red LH (left hand), blue RH (right hand), magenta LO (left-hand object), cyan RO (right-hand object), green TO (two-hand object); black background."
        )
        if examples:
            gr.Examples(
                [[str(x)] for x in examples], inputs=[source], cache_examples=False
            )
        gr.Markdown(
            "These six examples were selected for good segmentation quality and scene/role diversity. They illustrate successful cases, not representative benchmark performance. See examples/manifest.json for attribution; they are not relicensed under MIT. This RGB demo uses PyTorch FP32, not a Jetson INT8 engine."
        )
        button.click(
            lambda image, s: run_image(image, s, device),
            [source, seed],
            [output, color, download, status],
            concurrency_limit=1,
            api_name="predict",
        )
    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args()
    build_app(a.device).queue(max_size=8).launch(
        server_name="127.0.0.1", server_port=a.port, share=False, inbrowser=False
    )
