# 04_pipeline_design/smoke_v1_models.py
# Viability check V1: do the two frozen models load and run on the Mac GPU (mps), and does the
# road mask find the road in a REAL dashcam frame?
#   SegFormer-B0 (Cityscapes street scenes; class 0 = road)  -> road mask
#   DINOv2-small                                              -> one embedding per frame
# Usage (project root, pavement env active):
#   python 04_pipeline_design/smoke_v1_models.py reports/figures/smoke/img_01.jpg
# First run downloads ~100 MB of model weights to ~/.cache/huggingface (one time).
import sys, time
from pathlib import Path
import numpy as np
import torch, transformers
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, SegformerForSemanticSegmentation

ROOT     = Path(__file__).resolve().parent.parent
img_path = Path(sys.argv[1])
img      = Image.open(img_path).convert("RGB")
dev      = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"torch {torch.__version__} | transformers {transformers.__version__} | device {dev} | frame {img.size}")

def sync():                                   # GPU work is asynchronous; wait before reading the clock
    if dev == "mps":
        torch.mps.synchronize()

# 1) road mask
SEG_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
sproc  = AutoImageProcessor.from_pretrained(SEG_ID)
segm   = SegformerForSemanticSegmentation.from_pretrained(SEG_ID).to(dev).eval()
with torch.no_grad():
    t = time.time()
    logits = segm(**sproc(images=img, return_tensors="pt").to(dev)).logits          # (1, 19, H/4, W/4)
    logits = torch.nn.functional.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False)
    road   = (logits.argmax(1)[0] == 0).cpu().numpy()                                 # True where "road"
    sync(); mask_s = time.time() - t
print(f"road = {road.mean():.0%} of the frame | mask {mask_s:.2f}s (first call includes warm-up)")

# save an overlay: road tinted magenta, so you can judge the mask by eye
out_dir = ROOT / "reports/figures/smoke"
out_dir.mkdir(parents=True, exist_ok=True)
rgb = np.asarray(img).copy()
rgb[road] = (0.5 * rgb[road] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
overlay = out_dir / f"{img_path.stem}_roadmask.jpg"
Image.fromarray(rgb).save(overlay, quality=85)

# 2) crop to the road's bounding box, then embed (16 copies = a realistic batch, for the speed number)
ys, xs = np.where(road)
crop = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)) if road.any() else img
DINO_ID = "facebook/dinov2-small"
dproc   = AutoImageProcessor.from_pretrained(DINO_ID)
dino    = AutoModel.from_pretrained(DINO_ID).to(dev).eval()
batch   = dproc(images=[crop] * 16, return_tensors="pt").to(dev)
with torch.no_grad():
    dino(**batch); sync()                                                              # warm-up
    t = time.time(); emb = dino(**batch).pooler_output; sync()
fps = 16 / (time.time() - t)
print(f"road crop {crop.size} -> DINOv2-small embedding {tuple(emb.shape)} | {fps:.0f} frames/s (batch 16)")
print(f"saved overlay: {overlay}")
# NOTE: the default DINOv2 processor center-crops to 224x224, which cuts the sides off a wide road crop.
# Fine for a smoke test; the real pipeline will keep the aspect ratio (Viability check V3).
