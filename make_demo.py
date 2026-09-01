"""
Headless demo generator for Genetic Da Vinci.

Runs the SAME search engine (run_ga) that the Gradio app uses, but with no
server, no browser and no port. It reconstructs the sample target image from
circles / triangles / lines and writes:

    assets/demo_result.png   the final reconstruction
    assets/demo.gif          a short animation of the image "assembling"

Usage:
    python make_demo.py                  # modest budget (fast)
    python make_demo.py --shapes 800     # more shapes = closer, but slower
"""
import argparse
import os
import time

import numpy as np
from PIL import Image
from sobol import run_ga

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
TARGET = os.path.join(ASSETS, "sample_target.jpg")
W = H = 400


def as_rgb_image(obj):
    """Accept a PIL Image or an HxWx3 array and return an RGB PIL Image."""
    if isinstance(obj, Image.Image):
        return obj.convert("RGB")
    arr = np.asarray(obj)
    if arr.size == 0:
        return Image.new("RGB", (W, H))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, "L").convert("RGB")
    return Image.fromarray(arr, "RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", type=int, default=300, help="max number of shapes")
    ap.add_argument("--pop", type=int, default=24, help="population size per batch")
    ap.add_argument("--alpha", type=int, default=120, help="opacity 0-255")
    ap.add_argument("--frame-every", type=int, default=50,
                    help="save a GIF frame every N shapes")
    args = ap.parse_args()

    os.makedirs(ASSETS, exist_ok=True)
    target = Image.open(TARGET).convert("RGBA").resize((W, H), Image.LANCZOS)

    print(f"Running run_ga: shapes={args.shapes} pop={args.pop} alpha={args.alpha}")
    t0 = time.time()
    frames = []
    final = None
    last_shape = 0
    for best_img, loss, shape_count, shapes, _ in run_ga(
        target, "arial.ttf", 64, args.shapes, args.pop, W, H,
        alpha=args.alpha, shape_types=["circle", "triangle", "line"],
        verbose=False,
    ):
        final = best_img
        if not frames or (shape_count - last_shape >= args.frame_every):
            frames.append(as_rgb_image(best_img))
            last_shape = shape_count
        print(f"  shape {shape_count:<5d}  loss={loss:.4f}  ({time.time() - t0:.1f}s)")

    if final is None:
        raise SystemExit("run_ga produced no frames.")

    out_png = os.path.join(ASSETS, "demo_result.png")
    as_rgb_image(final).save(out_png)
    print(f"wrote {out_png}")

    if len(frames) > 1:
        # Downscale to keep the GIF small.
        gframes = [f.resize((200, 200)) for f in frames]
        gif_path = os.path.join(ASSETS, "demo.gif")
        gframes[0].save(
            gif_path,
            save_all=True, append_images=gframes[1:], duration=250, loop=0,
        )
        print(f"wrote {gif_path}")

    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
