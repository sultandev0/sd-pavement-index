# 04_pipeline_design/road_masks.py
# Viability check V3, part A (+ the mask half of V4's throughput).
#
# Runs the frozen SegFormer-B0 road model on every sampled frame ONCE and caches the result, so every later
# decision (strip rows, side edges, tile rule, end trim) is a filter on saved tables, never a re-run.
#
# Preprocessing = exactly what passed V1 (v1_mask_compare.py), plus the settled QA/QC item M6:
#   crop the bottom 10% (hood + burned-in text) -> resize to 2048 x 1024 (aspect ratio kept, multiple of 32)
#   -> ImageNet normalization -> SegFormer-B0 (Cityscapes) -> logits upsampled BILINEARLY to 2048 x 1024 (M6)
#   -> argmax = one class per pixel (Cityscapes ids 0-18; 0 = road).
#
# Writes (all under data/, git-ignored):
#   data/masks/<SESSION>/<frame stem>.png    class map, 2048 x 1024, 8-bit (one Cityscapes id per pixel)
#   data/processed/masks_<SESSION>.csv       per frame: road share of the frame / of the lower half, seconds
#   data/processed/road_heatmap.npz          share of frames in which each pixel is road (all drives)
#   reports/figures/v3/road_heatmap.png      the heatmap with a 5% grid -> used to choose the strip rows/edges
#   data/processed/mask_log.csv              timing per run (V4 = mask + embed; extraction is timed elsewhere)
# Resumable: frames whose class map already exists are skipped (their maps still count in the heatmap).
#
# Usage (project root, pavement env active):
#   python 04_pipeline_design/road_masks.py --selftest      # ~10 s: synthetic frame, checks the whole path
#   python 04_pipeline_design/road_masks.py                 # all frames_<SESSION>.csv tables
#   python 04_pipeline_design/road_masks.py --drive 11-11-2026-Pacific-Beach
#   python 04_pipeline_design/road_masks.py --bench 8      # time batch sizes / upsampling on 8 real frames, saves nothing

import argparse, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

HOOD_FRAC = 0.10                        # bottom 10% = hood + burned-in overlay (V1)
WORK_W    = 2048                        # Cityscapes width (V1)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet normalization (V1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SEG_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
BATCH  = 1                              # frames per forward pass (set by --batch; 4 was 5x slower on mps)
ROAD   = 0                              # Cityscapes class id for road
CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
           "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
           "motorcycle", "bicycle"]

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
PROC = ROOT / "data/processed"


def stop(msg):
    sys.exit(f"STOP: {msg}")


def mult32(v):
    return max(32, int(round(v / 32)) * 32)


def work_size(w, h):
    """(width, height) SegFormer sees, after the hood crop: 3840 x 2160 -> 2048 x 1024"""
    keep_h = int(h * (1 - HOOD_FRAC))
    return keep_h, (WORK_W, mult32(keep_h * WORK_W / w))


def prep(path):
    """JPEG -> normalized float32 array (H, W, 3) at the working size, same steps as V1"""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    keep_h, size = work_size(w, h)
    x = np.asarray(im.crop((0, 0, w, keep_h)).resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
    return (x - MEAN) / STD, (w, h)


def class_map_png(lab):
    """uint8 class map -> PNG (lossless)"""
    return Image.fromarray(lab.astype(np.uint8))


def readable(path):
    """True if a saved class map exists and decodes completely"""
    if not path.exists():
        return False
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def mask_stem(file):
    return Path(file).stem


def load_frames(drive):
    tables = sorted(PROC.glob("frames_*.csv"))
    if drive:
        tables = [t for t in tables if t.stem == f"frames_{drive}"]
    if not tables:
        stop("no frames_<SESSION>.csv found (or --drive does not match one)")
    return pd.concat([pd.read_csv(t) for t in tables], ignore_index=True)


def heatmap_figure(heat, n, out):
    """heatmap of 'share of frames where this pixel is road', with a 5% grid and row/column labels"""
    h, w = heat.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = (heat * 255).astype(np.uint8)              # red = always road, black = never
    rgb[..., 1] = (heat * 140).astype(np.uint8)
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    for k in range(1, 20):                                    # 5% grid; full-frame percentages
        y = int(k * 0.05 / (1 - HOOD_FRAC) * h)               # row k*5% of the FULL frame
        if y < h:
            d.line([0, y, w, y], fill=(80, 160, 255) if k % 2 == 0 else (40, 80, 128), width=1)
            d.text((4, y + 2), f"{k * 5}%", fill=(255, 255, 255))
        x = int(k * 0.05 * w)
        d.line([x, 0, x, h], fill=(40, 80, 128), width=1)
        d.text((x + 2, 4), f"{k * 5}%", fill=(255, 255, 255))
    d.text((8, h - 20), f"share of {n} frames where the pixel is road (bright = always); rows/cols = % of full frame",
           fill=(255, 255, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)


def profile(heat):
    """printable table: mean road share per full-frame row band (2%) x column band (10%)"""
    h, w = heat.shape
    rows = []
    for lo in range(50, 90, 2):                               # rows 50-90% of the full frame
        y0, y1 = int(lo / 100 / (1 - HOOD_FRAC) * h), int((lo + 2) / 100 / (1 - HOOD_FRAC) * h)
        band = heat[y0:min(y1, h)]
        rows.append([f"{lo}-{lo + 2}%"] + [f"{band[:, int(c / 10 * w):int((c + 1) / 10 * w)].mean():.2f}"
                                           for c in range(10)])
    return pd.DataFrame(rows, columns=["rows"] + [f"x{c * 10}-{c * 10 + 10}" for c in range(10)]).to_string(index=False)


def run(frames, predict, dev, label):
    """frames: table with 'drive' and 'file'; predict(batch array NHWC) -> uint8 class maps (N, h, w)"""
    t_all = time.time()
    todo, stats = [], []
    for r in frames.itertuples(index=False):
        out = ROOT / "data/masks" / r.drive / f"{mask_stem(r.file)}.png"
        if not (ROOT / r.file).exists():
            stop(f"frame missing on disk: {r.file}")
        todo.append((r.drive, r.file, out))
    need = [t for t in todo if not readable(t[2])]
    print(f"{label}: {len(todo):,} frames | {len(todo) - len(need):,} already masked | {len(need):,} to do | device {dev}")

    t_prep = t_model = 0.0
    batches = [need[i:i + BATCH] for i in range(0, len(need), BATCH)]
    with ThreadPoolExecutor(max_workers=BATCH + 2) as pool:     # decode the next batch while the GPU works
        load = lambda b: [f.result() for f in [pool.submit(prep, ROOT / fl) for _, fl, _ in b]]
        ahead = pool.submit(load, batches[0]) if batches else None   # at most ONE batch decoded ahead (memory)
        for k, b in enumerate(batches):
            prepped = ahead.result()
            ahead = pool.submit(load, batches[k + 1]) if k + 1 < len(batches) else None
            t0 = time.time()
            x = np.stack([p[0] for p in prepped])
            labs = predict(x)
            t_model += time.time() - t0
            for (drive, f, out), lab in zip(b, labs):
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".tmp.png")
                class_map_png(lab).save(tmp)
                tmp.replace(out)                                   # atomic: a stopped run never leaves half a file
            if (k + 1) % 25 == 0 or k == len(batches) - 1:
                done = min((k + 1) * BATCH, len(need))
                el = time.time() - t_all
                print(f"  {done:,}/{len(need):,} frames | {el:.0f} s | {el / done:.2f} s/frame")
    secs = time.time() - t_all

    # per-frame stats from ALL saved maps (new and old); heatmap from training drives only (not <SESSION>__stale)
    heat, n = np.zeros((1, 1)), 0
    for drive, f, out in todo:
        lab = np.asarray(Image.open(out))
        road = lab == ROAD
        if "__stale" not in drive:
            heat = road.astype(np.float64) if n == 0 else heat + road
            n += 1
        h = lab.shape[0]
        stats.append(dict(drive=drive, file=f, road_share=road.mean(), road_share_lower=road[h // 2:].mean()))
    st = pd.DataFrame(stats)
    for drive, g in st.groupby("drive"):
        g.to_csv(PROC / f"masks_{drive}.csv", index=False)
    if n:
        heat /= n
        np.savez_compressed(PROC / "road_heatmap.npz", heat=heat.astype(np.float32), n=n)
        heatmap_figure(heat, n, ROOT / "reports/figures/v3/road_heatmap.png")
    log = PROC / "mask_log.csv"
    pd.DataFrame([dict(when=time.strftime("%Y-%m-%d %H:%M"), label=label, device=dev, frames_masked=len(need),
                       seconds=round(secs, 1), s_per_frame=round(secs / max(len(need), 1), 3),
                       model_s_per_frame=round(t_model / max(len(need), 1), 3))]).to_csv(
        log, mode="a", header=not log.exists(), index=False)

    print(f"\nmasked {len(need):,} frames in {secs:.0f} s"
          + (f" ({secs / len(need):.2f} s/frame; model only {t_model / len(need):.2f})" if need else ""))
    print(f"road share of the frame: median {st['road_share'].median():.1%} | lower half: median "
          f"{st['road_share_lower'].median():.1%} | frames with < 5% road in the lower half: "
          f"{int((st['road_share_lower'] < 0.05).sum())}")
    if n:
        print(f"\nROAD HEATMAP PROFILE ({n:,} training frames; stale frames excluded): share of frames where the band is road "
              "(rows = % of full frame, cols = % of width)")
        print(profile(heat))
    print(f"\nsaved: data/masks/<drive>/*.png, masks_<drive>.csv, road_heatmap.npz, "
          f"reports/figures/v3/road_heatmap.png, mask_log.csv")


def make_predict(dev, upsample="bilinear_logits"):
    import torch
    from transformers import SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(SEG_ID).to(dev).eval()

    def predict(x):
        t = torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)
        with torch.no_grad():
            logits = model(pixel_values=t).logits                                   # (N, 19, h/4, w/4)
            if upsample == "bilinear_logits":                                       # M6 (production)
                logits = torch.nn.functional.interpolate(logits, size=x.shape[1:3], mode="bilinear",
                                                         align_corners=False)
                lab = logits.argmax(1).to(torch.uint8).cpu().numpy()
            else:                                                                   # V1 style, bench only
                lab = torch.nn.functional.interpolate(logits.argmax(1, keepdim=True).float(),
                                                      size=x.shape[1:3], mode="nearest")[:, 0]
                lab = lab.to(torch.uint8).cpu().numpy()
        return lab
    return predict


def bench(dev, n):
    """time the mask step on n real frames for several settings; nothing is saved"""
    global BATCH
    import torch
    fr = load_frames(None).sample(n, random_state=0)
    xs = [prep(ROOT / f)[0] for f in fr["file"]]
    print(f"BENCH on {n} real frames, device {dev} (decode excluded; first call per setting = warm-up, not timed)")
    for up in ("bilinear_logits", "nearest_after_argmax"):
        predict = make_predict(dev, up)
        for b in (1, 2, 4):
            predict(np.stack(xs[:b]))
            if dev == "mps":
                torch.mps.synchronize()
            t = time.time()
            for i in range(0, n, b):
                predict(np.stack(xs[i:i + b]))
            secs = (time.time() - t) / n
            print(f"  upsample {up:20s} batch {b}: {secs:.2f} s/frame")
        del predict
        if dev == "mps":
            torch.mps.empty_cache()
    t = time.time()
    for f in fr["file"]:
        prep(ROOT / f)
    print(f"  JPEG decode + crop + resize (one thread): {(time.time() - t) / n:.2f} s/frame")


def selftest(predict, dev):
    """synthetic 3840 x 2160 frame in a temp 'project': whole path end to end, then cleaned up"""
    global ROOT, PROC
    real_root = ROOT
    with tempfile.TemporaryDirectory() as tmp:
        ROOT, PROC = Path(tmp), Path(tmp) / "data/processed"
        PROC.mkdir(parents=True)
        (ROOT / "data/frames/TEST").mkdir(parents=True)
        rng = np.random.default_rng(0)
        files = []
        for i in range(5):
            a = np.full((2160, 3840, 3), 120, np.uint8)
            a[:1000] = (135, 180, 235)                                             # "sky"
            a[1200:] = rng.integers(60, 90, (960, 3840, 3), dtype=np.uint8)        # grey "asphalt"
            p = ROOT / f"data/frames/TEST/clip_n{i:05d}.jpg"
            Image.fromarray(a).save(p, quality=90)
            files.append(str(p.relative_to(ROOT)))
        fr = pd.DataFrame(dict(drive="TEST", file=files))
        run(fr, predict, dev, "SELFTEST")
        lab = np.asarray(Image.open(ROOT / "data/masks/TEST/clip_n00000.png"))
        assert lab.shape == (1024, 2048) and lab.dtype == np.uint8 and lab.max() <= 18, lab.shape
        run(fr, predict, dev, "SELFTEST resume")                                   # must skip all 5
        print(f"\nSELFTEST OK | class map {lab.shape} | classes present: "
              f"{', '.join(CLASSES[c] for c in np.unique(lab))} | nothing written to {real_root.name}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="V3a: SegFormer-B0 class maps for every sampled frame")
    ap.add_argument("--drive", help="only this SESSION (default: all frames_*.csv)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--batch", type=int, default=BATCH, help="frames per forward pass (default 1)")
    ap.add_argument("--bench", type=int, metavar="N", help="time settings on N real frames; saves nothing")
    args = ap.parse_args()
    BATCH = args.batch
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if args.bench:
        bench(dev, args.bench)
        sys.exit()
    predict = make_predict(dev)
    if args.selftest:
        selftest(predict, dev)
    else:
        run(load_frames(args.drive), predict, dev, args.drive or "all drives")
