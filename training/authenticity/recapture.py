"""
Simulate photographing an image on a screen (monitor / laptop / phone).

Real recaptures are hard to collect at scale, so training uses a physical
simulation of the artifacts a phone camera records when pointed at a display:

  1. The image is shown on an LCD: each pixel becomes an R|G|B sub-pixel
     triplet with a dark grid between pixels.
  2. The camera views the screen at an angle and distance: a perspective warp
     resamples the sub-pixel grid without anti-aliasing, which produces moire
     interference patterns, the strongest recapture cue.
  3. Optics and sensor: defocus, screen gamma / blue cast, glare, vignetting,
     sensor noise, and optionally a bezel / desk visible at the edges.

Usage (preview a few simulations):
    python training/authenticity/recapture.py input.jpg out_dir/
"""

import os
import random
import sys

import cv2
import numpy as np


def _subpixel_display(img, pixel_size):
    """Render img (HxWx3 uint8 RGB) as an LCD panel, pixel_size sensor px per screen px."""
    h, w = img.shape[:2]
    big = cv2.resize(img, (w * pixel_size, h * pixel_size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    mask = np.zeros((pixel_size, pixel_size, 3), np.float32)
    third = max(1, pixel_size // 3)
    for c in range(3):  # vertical R|G|B stripes
        mask[:, c * third:(c + 1) * third, c] = 1.0
    gap = max(1, pixel_size // 6)  # black matrix between pixels
    mask[-gap:, :, :] *= 0.15
    mask = np.tile(mask, (h, w, 1))
    # Sub-pixel stripes lose ~2/3 of the light; scale back up. The camera lens
    # and sensor low-pass the grid, so only part of its contrast survives.
    strength = random.uniform(0.35, 0.85)
    panel = big * (1 - strength) + np.clip(big * mask * 2.4, 0, 255) * strength
    sigma = pixel_size * random.uniform(0.1, 0.25)
    return cv2.GaussianBlur(panel, (0, 0), sigma)


def _perspective(panel, out_w, out_h, max_tilt):
    """View the panel at a random angle, landing it inside an out_w x out_h frame."""
    ph, pw = panel.shape[:2]
    margin = random.uniform(-0.1, 0.1)  # negative: screen overfills the frame
    # Destination corners: a jittered quad covering most of the frame
    def jitter(x, y):
        return [x + random.uniform(-max_tilt, max_tilt) * out_w, y + random.uniform(-max_tilt, max_tilt) * out_h]
    dst = np.float32([
        jitter(margin * out_w, margin * out_h),
        jitter((1 - margin) * out_w, margin * out_h),
        jitter((1 - margin) * out_w, (1 - margin) * out_h),
        jitter(margin * out_w, (1 - margin) * out_h),
    ])
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    m = cv2.getPerspectiveTransform(src, dst)
    # INTER_LINEAR without prefiltering aliases the sub-pixel grid -> moire
    background = random.choice([(15, 15, 18), (40, 40, 45), (120, 110, 100), (200, 195, 190)])
    return cv2.warpPerspective(panel, m, (out_w, out_h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=background)


def _optics(img):
    h, w = img.shape[:2]
    # Screen tone: gamma, contrast loss, cool/warm cast
    img = 255 * (img / 255) ** random.uniform(0.8, 1.25)
    img = img * random.uniform(0.75, 1.0) + random.uniform(5, 30)
    img *= np.array([random.uniform(0.9, 1.0), random.uniform(0.95, 1.02), random.uniform(1.0, 1.12)], np.float32)
    # Glare: soft bright blob
    if random.random() < 0.5:
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = random.uniform(0, w), random.uniform(0, h)
        r = random.uniform(0.2, 0.6) * max(h, w)
        img += random.uniform(20, 90) * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))[..., None]
    # Vignetting
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    img *= (1 - random.uniform(0.0, 0.35) * np.clip(d - 0.4, 0, None))[..., None]
    # Slight defocus / motion
    if random.random() < 0.6:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), random.uniform(0.3, 1.2))
    # Sensor noise
    img += np.random.normal(0, random.uniform(1, 6), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def simulate_recapture(img, out_size=None):
    """img: HxWx3 uint8 RGB. Returns a simulated phone photo of it on a screen."""
    h, w = img.shape[:2]
    if out_size is None:
        out_size = (w, h)
    out_w, out_h = out_size
    # Screen resolution relative to the camera: 2-5 sensor px per screen pixel
    # (closer / lower-dpi screens give bigger pixels and stronger moire).
    pixel_size = random.choice([3, 3, 4, 5, 6])
    # Panel rendered 0.9-2x the output size, so one screen pixel lands on
    # ~1.5-6 camera pixels: periods near 2 px alias into strong moire.
    f = random.uniform(0.9, 2.0)
    screen_w = max(32, int(out_w * f / pixel_size))
    screen_h = max(32, int(screen_w * h / w))
    shown = cv2.resize(img, (screen_w, screen_h), interpolation=cv2.INTER_AREA)
    panel = _subpixel_display(shown, pixel_size)
    viewed = _perspective(panel, out_w, out_h, max_tilt=random.uniform(0.0, 0.06))
    return _optics(viewed.astype(np.float32))


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    img = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
    for i in range(4):
        sim = simulate_recapture(img)
        cv2.imwrite(os.path.join(out, f"recapture_{i}.jpg"), cv2.cvtColor(sim, cv2.COLOR_RGB2BGR))
    print("wrote", out)
