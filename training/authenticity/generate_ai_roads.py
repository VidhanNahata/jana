"""
Generate AI images of potholes / roads locally, so the authenticity
classifier sees fakes of exactly what a fraudulent report would contain.

  sd_turbo        -> train/val (stabilityai/sd-turbo, 1 step, 512 px)
  dreamshaper_test -> held-out test generator (Lykon/dreamshaper-8, SD1.5 photoreal finetune)

Output: training/data/raw/authenticity/local_sd/<generator>/*.png

Usage:
    python training/authenticity/generate_ai_roads.py --n-turbo 500 --n-holdout 150
Run when no other training is using the GPU (needs ~3-4 GB).
"""

import argparse
import os
import random

import torch
from diffusers import AutoPipelineForText2Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "authenticity", "local_sd")

SUBJECTS = [
    "a large pothole", "a deep pothole filled with muddy water", "several small potholes", "a cracked pothole",
    "a pothole with broken asphalt edges", "a crumbling road with potholes", "a pothole next to a manhole cover",
    "a waterlogged road with hidden potholes", "a small pothole", "a pothole near a speed breaker",
    "a damaged road surface with a pothole", "a pothole surrounded by gravel",
]
PLACES = [
    "on an Indian city street", "on a Bengaluru road", "on a busy road with auto rickshaws", "on a highway",
    "on a narrow residential lane", "near a bus stop", "on a village road", "at a traffic junction",
    "on a road with parked two-wheelers", "on a flyover",
]
CONDITIONS = [
    "during monsoon rain", "after rain with puddles and reflections", "on a sunny afternoon", "at night under street lights",
    "in harsh midday sun", "on an overcast day", "at dusk", "in dusty dry weather", "early morning with fog",
]
STYLES = [
    "photo taken on a smartphone", "dashcam footage still", "amateur phone photo, slightly tilted",
    "close-up photo looking down at the road", "wide shot, photorealistic", "news photo", "DSLR photo, 35mm",
]
NEGATIVE = "cartoon, painting, illustration, drawing, text, watermark, blurry, lowres"


def prompt():
    return f"{random.choice(STYLES)} of {random.choice(SUBJECTS)} {random.choice(PLACES)} {random.choice(CONDITIONS)}"


def run(model, name, n, steps, guidance, sizes, negative):
    out = os.path.join(OUT, name)
    os.makedirs(out, exist_ok=True)
    # The safety checker isn't downloaded (prompts are road scenes only)
    pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=torch.float16, variant="fp16",
                                                     safety_checker=None, requires_safety_checker=False)
    pipe = pipe.to("mps")
    # No enable_attention_slicing(): on MPS fp16 it produces all-black images
    pipe.set_progress_bar_config(disable=True)
    for i in range(n):
        w, h = random.choice(sizes)  # drawn every index so prompts stay reproducible
        path = os.path.join(out, f"{i:05d}.png")
        if os.path.exists(path):
            continue
        kwargs = {"negative_prompt": NEGATIVE} if negative else {}
        img = pipe(prompt(), num_inference_steps=steps, guidance_scale=guidance, width=w, height=h, **kwargs).images[0]
        img.save(path)
        if i % 25 == 0:
            print(name, i, flush=True)
            torch.mps.empty_cache()
    del pipe
    torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-turbo", type=int, default=500)
    parser.add_argument("--n-holdout", type=int, default=150)
    args = parser.parse_args()
    random.seed(1)
    run("stabilityai/sd-turbo", "sd_turbo", args.n_turbo, steps=2, guidance=0.0,
        sizes=[(512, 512), (512, 640), (640, 512)], negative=False)
    run("Lykon/dreamshaper-8", "dreamshaper_test", args.n_holdout, steps=20, guidance=6.5,
        sizes=[(512, 768), (768, 512), (512, 512)], negative=True)


if __name__ == "__main__":
    main()
