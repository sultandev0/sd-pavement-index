# 04_pipeline_design/v1_mask_compare.py
# Viability check V1, follow-up. The first run (smoke_v1_models.py) loaded and ran both models on mps,
# but the road mask was poor: 6% of the frame, most of the near road missed, the hood called "road".
# This script tests two likely causes on the SAME frame and saves a 2x2 comparison image:
#   (1) preprocessing - the default processor squashes the 16:9 frame into a square, and the hood +
#       burned-in text strip is included. Fix tried: crop the hood strip, resize keeping the aspect ratio.
#   (2) model size - SegFormer-B0 is the smallest variant. Compared: B0, B2, B5 (same Cityscapes training).
# It writes only reports/figures/smoke/<frame>_mask_compare.jpg. First run downloads B2 (~110 MB)
# and B5 (~340 MB) to ~/.cache/huggingface.
# Usage (project root, pavement env active):
#   python 04_pipeline_design/v1_mask_compare.py data/frames_test/pb_sample.jpg
import sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT     = Path(__file__).resolve().parent.parent
img_path = Path(sys.argv[1])
full     = Image.open(img_path).convert("RGB")
W, H     = full.size

HOOD_FRAC = 0.10                       # bottom 10% of the frame = hood + burned-in speed/GPS/clock text
WORK_W    = 2048                       # Cityscapes training images are 2048 px wide
NEAR_BOX  = (0.80, 0.90, 0.15, 0.60)   # rough near-road check box (rows 80-90%, x 15-60%), picked by eye on pb_sample
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet normalization (SegFormer's processor)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def mult32(v):                         # SegFormer downsamples by 32; keep sizes divisible by 32
    return max(32, int(round(v / 32)) * 32)

keep_h  = int(H * (1 - HOOD_FRAC))
cropped = full.crop((0, 0, W, keep_h))
SQUARE  = (1024, 1024)
ASPECT  = (WORK_W, mult32(keep_h * WORK_W / W))
RUNS = [("B0 | whole frame squashed to a square (like the 1st run)", "b0", full,    SQUARE),
        ("B0 | hood cropped, aspect ratio kept",                     "b0", cropped, ASPECT),
        ("B2 | hood cropped, aspect ratio kept",                     "b2", cropped, ASPECT),
        ("B5 | hood cropped, aspect ratio kept",                     "b5", cropped, ASPECT)]

def to_full(mask):
    """pad a mask of the hood-cropped image back to full-frame size (hood rows = not road)"""
    out = np.zeros((H, W), dtype=bool)
    out[: mask.shape[0]] = mask
    return out

def scores(m):
    y0, y1, x0, x1 = NEAR_BOX
    near = m[int(y0 * H): int(y1 * H), int(x0 * W): int(x1 * W)]
    return {"road": m.mean(), "near": near.mean(), "hood": m[keep_h:].mean()}

def panel(m, title, sc, secs):
    rgb = np.asarray(full).copy()
    rgb[m] = (0.5 * rgb[m] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    pw = 960
    im = Image.fromarray(rgb).resize((pw, int(pw * H / W)), Image.BILINEAR)
    d  = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:                    # older Pillow
        font = ImageFont.load_default()
    y0, y1, x0, x1 = NEAR_BOX
    d.rectangle([x0 * pw, y0 * im.height, x1 * pw, y1 * im.height], outline=(255, 255, 0), width=3)
    d.line([0, keep_h * im.height / H, pw, keep_h * im.height / H], fill=(0, 255, 255), width=2)
    d.rectangle([0, 0, pw, 52], fill=(0, 0, 0))
    d.text((8, 4), title, fill=(255, 255, 255), font=font)
    d.text((8, 28), f"road {sc['road']:.0%} of frame | near box {sc['near']:.0%} | hood strip {sc['hood']:.0%}"
                    f" | {secs:.2f} s/frame", fill=(255, 255, 0), font=font)
    return im

def grid(panels):
    pw, ph = panels[0].size
    g = Image.new("RGB", (2 * pw, 2 * ph))
    for i, p in enumerate(panels):
        g.paste(p, ((i % 2) * pw, (i // 2) * ph))
    return g

if __name__ == "__main__":
    import torch
    from transformers import SegformerForSemanticSegmentation
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    def sync():
        if dev == "mps":
            torch.mps.synchronize()

    def road_mask(model, img, size):
        """resize img to size=(w, h), run SegFormer, return a road mask at img's own size"""
        x = np.asarray(img.resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
        x = torch.from_numpy((x - MEAN) / STD).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad():
            lab = model(pixel_values=x).logits.argmax(1, keepdim=True).float()   # class per pixel, 1/4 res
            lab = torch.nn.functional.interpolate(lab, size=img.size[::-1], mode="nearest")
        return (lab[0, 0] == 0).cpu().numpy()                                     # Cityscapes class 0 = road

    print(f"frame {W}x{H} | device {dev} | hood strip = bottom {HOOD_FRAC:.0%} | aspect-kept size {ASPECT}")
    panels, models = [], {}
    for title, variant, img, size in RUNS:
        if variant not in models:
            models = {variant: SegformerForSemanticSegmentation.from_pretrained(
                f"nvidia/segformer-{variant}-finetuned-cityscapes-1024-1024").to(dev).eval()}
            if dev == "mps":
                torch.mps.empty_cache()
        model = models[variant]
        road_mask(model, img, size); sync()                     # warm-up (not timed)
        t = time.time(); m = road_mask(model, img, size); sync(); secs = time.time() - t
        m = to_full(m) if img is cropped else m
        sc = scores(m)
        print(f"{title:58s} road {sc['road']:4.0%} | near box {sc['near']:4.0%} | "
              f"hood strip {sc['hood']:4.0%} | {secs:.2f} s/frame")
        panels.append(panel(m, title, sc, secs))

    out = ROOT / "reports/figures/smoke" / f"{img_path.stem}_mask_compare.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    grid(panels).save(out, quality=85)
    print(f"saved: {out}")
