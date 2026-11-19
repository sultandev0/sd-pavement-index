# 04_pipeline_design/embed_tiles.py
# Viability check V3, part C + the embed half of V4. QA/QC item M1.
#
# For every sampled frame: cut the near-field strip into 252 x 252 px tiles at NATIVE 4K resolution (no resize,
# no center crop), normalize with ImageNet mean/std by hand, and run frozen DINOv2-small on all tiles of the
# frame in one forward pass. EVERY tile is embedded, kept or not; its road share is stored so the M1b rule
# (keep a tile if >= 98% road, a frame if >= 2 tiles) is a filter applied later, never a re-run.
# In the same pass, rung 3's hand-crafted features are computed on the ROAD pixels of each tile:
# grayscale mean, grayscale std, Laplacian variance (edge sharpness).
#
# Strip = the one that passed V3 on 2026-11-14 (review_config.json): 1 tile row, bottom at 90% of the frame,
# x 10-60% -> 7 tiles per frame. The script stops if the review config says otherwise.
#
# Writes (git-ignored):
#   data/embeddings/tiles_<SESSION>.npz   cls (n_tiles x 384), pmean (n_tiles x 384), float32
#   data/embeddings/tiles_<SESSION>.csv   one row per tile: frame_id, file, seg_key, tile, road_share,
#                                         gray_mean, gray_std, lap_var  (row i = embedding row i)
#   data/processed/embed_log.csv          timing (V4 = mask + embed; extraction is timed elsewhere)
# Usage (project root, pavement env active):
#   python 04_pipeline_design/embed_tiles.py --selftest     # ~15 s, synthetic frames, checks shapes + batching
#   python 04_pipeline_design/embed_tiles.py --bench 16     # time it on 16 real frames, saves nothing
#   python 04_pipeline_design/embed_tiles.py                # all drives; skips drives already embedded

import argparse, json, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

TILE = 252                                  # 18 x 18 DINOv2 patches of 14 px
STRIP = dict(rows=1, bottom=0.90, x_left=0.10, x_right=0.60)   # passed V3 50/50 (2026-11-14)
HOOD_FRAC = 0.10                            # class maps cover the top 90% of the frame
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)       # ImageNet (DINOv2's own normalization)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DINO_ID = "facebook/dinov2-small"
DIM = 384

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
PROC = ROOT / "data/processed"
EMB = ROOT / "data/embeddings"


def stop(msg):
    sys.exit(f"STOP: {msg}")


def tile_boxes(W, H):
    """same geometry as v3_review.py: whole tiles laid upward from bottom*H, centered between the x limits"""
    yb = int(round(STRIP["bottom"] * H))
    x0, x1 = int(STRIP["x_left"] * W), int(STRIP["x_right"] * W)
    ncol = (x1 - x0) // TILE
    off = x0 + ((x1 - x0) - ncol * TILE) // 2
    return [(off + c * TILE, yb - (r + 1) * TILE, off + (c + 1) * TILE, yb - r * TILE)
            for r in range(STRIP["rows"]) for c in range(ncol)]


def check_review_config():
    f = ROOT / "data/v3_review/review_config.json"
    if not f.exists():
        print("note: data/v3_review/review_config.json not found; using the strip written in this script")
        return
    c = json.loads(f.read_text())
    mine = dict(rows=STRIP["rows"], bottom=STRIP["bottom"], x_left=STRIP["x_left"], x_right=STRIP["x_right"], tile=TILE)
    theirs = {k: c.get(k) for k in mine}
    if any(abs(float(mine[k]) - float(theirs[k])) > 1e-9 for k in mine):
        stop(f"strip in this script {mine} differs from the reviewed strip {theirs}")


def road_tile(lab, box, W, H):
    """road pixels of one tile at native size (class map nearest-upsampled), and the tile's road share"""
    mh, mw = lab.shape
    keep_h = int(H * (1 - HOOD_FRAC))
    x0, y0, x1, y1 = box
    r0, r1 = int(round(y0 * mh / keep_h)), int(round(y1 * mh / keep_h))
    c0, c1 = int(round(x0 * mw / W)), int(round(x1 * mw / W))
    sub = lab[r0:r1, c0:c1]
    share = float((sub == 0).mean())                      # identical to v3_review.py's tile share
    road = np.asarray(Image.fromarray((sub == 0).astype(np.uint8) * 255).resize((TILE, TILE), Image.NEAREST)) > 0
    return road, share


def handcrafted(tile, road):
    """rung 3: grayscale mean/std and Laplacian variance over road pixels (NaN if < 100 road pixels)"""
    g = tile.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if road.sum() < 100:
        return np.nan, np.nan, np.nan
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    inner = road[1:-1, 1:-1] & road[:-2, 1:-1] & road[2:, 1:-1] & road[1:-1, :-2] & road[1:-1, 2:]
    lv = float(lap[inner].var()) if inner.sum() >= 100 else np.nan
    return float(g[road].mean()), float(g[road].std()), lv


def prep_frame(path_img, path_lab):
    """one frame -> (normalized tiles n x 252 x 252 x 3, road shares, hand-crafted rows)"""
    im = np.asarray(Image.open(path_img).convert("RGB"))
    H, W = im.shape[:2]
    lab = np.asarray(Image.open(path_lab))
    xs, shares, hc = [], [], []
    for box in tile_boxes(W, H):
        x0, y0, x1, y1 = box
        t = im[y0:y1, x0:x1]
        road, share = road_tile(lab, box, W, H)
        xs.append((t.astype(np.float32) / 255.0 - MEAN) / STD)
        shares.append(share)
        hc.append(handcrafted(t, road))
    return np.stack(xs), shares, hc


def load_frames(drive=None):
    tables = sorted(PROC.glob("frames_*.csv"))
    if drive:
        tables = [t for t in tables if t.stem == f"frames_{drive}"]
    if not tables:
        stop("no frames_<SESSION>.csv found")
    return {t.stem[len("frames_"):]: pd.read_csv(t) for t in tables}


def mask_path(drive, file):
    return ROOT / "data/masks" / drive / f"{Path(file).stem}.png"


def embed_drive(drive, fr, embed, dev):
    out_npz, out_csv = EMB / f"tiles_{drive}.npz", EMB / f"tiles_{drive}.csv"
    if out_npz.exists() and out_csv.exists():
        old = pd.read_csv(out_csv, usecols=["frame_id"])
        if set(old["frame_id"]) == set(fr["frame_id"]):
            print(f"{drive}: already embedded ({len(old):,} tiles), skipped")
            return None
        print(f"{drive}: frames table changed since the last run -> re-embedding")
    miss = [f for f in fr["file"] if not (ROOT / f).exists()] + \
           [f for f in fr["file"] if not mask_path(drive, f).exists()]
    if miss:
        stop(f"{drive}: {len(miss)} frames or class maps missing (first: {miss[0]}); run road_masks.py")

    rows, cls_all, pm_all = [], [], []
    t_all, t_model = time.time(), 0.0
    recs = list(fr.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=2) as pool:            # prepare the next frame while the GPU works
        ahead = pool.submit(prep_frame, ROOT / recs[0].file, mask_path(drive, recs[0].file))
        for k, r in enumerate(recs):
            x, shares, hc = ahead.result()
            if k + 1 < len(recs):
                ahead = pool.submit(prep_frame, ROOT / recs[k + 1].file, mask_path(drive, recs[k + 1].file))
            t0 = time.time()
            cls, pm = embed(x)
            t_model += time.time() - t0
            cls_all.append(cls); pm_all.append(pm)
            for i, (s, (gm, gs, lv)) in enumerate(zip(shares, hc)):
                rows.append(dict(frame_id=r.frame_id, file=r.file, seg_key=r.seg_key, tile=i, road_share=round(s, 5),
                                 gray_mean=gm, gray_std=gs, lap_var=lv))
            if (k + 1) % 100 == 0 or k == len(recs) - 1:
                el = time.time() - t_all
                print(f"  {drive}: {k + 1:,}/{len(recs):,} frames | {el:.0f} s | {el / (k + 1):.3f} s/frame")
    secs = time.time() - t_all
    EMB.mkdir(parents=True, exist_ok=True)
    cls_all, pm_all = np.concatenate(cls_all).astype(np.float32), np.concatenate(pm_all).astype(np.float32)
    tab = pd.DataFrame(rows)
    assert len(tab) == len(cls_all) == len(pm_all)
    tmp = EMB / f".tmp_tiles_{drive}.npz"
    np.savez(tmp, cls=cls_all, pmean=pm_all)
    tmp.replace(out_npz)
    tab.to_csv(out_csv, index=False)
    print(f"{drive}: {len(fr):,} frames -> {len(tab):,} tiles x {cls_all.shape[1]} dims in {secs:.0f} s "
          f"({secs / len(fr):.3f} s/frame, model {t_model / len(fr):.3f}) | tiles >= 98% road: "
          f"{(tab['road_share'] >= 0.98).mean():.0%}")
    return dict(drive=drive, frames=len(fr), tiles=len(tab), seconds=round(secs, 1),
                s_per_frame=round(secs / len(fr), 3), model_s_per_frame=round(t_model / len(fr), 3), device=dev)


def make_embed(dev):
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(DINO_ID).to(dev).eval()

    def embed(x):
        t = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(dev)        # n x 3 x 252 x 252
        with torch.no_grad():
            h = model(pixel_values=t).last_hidden_state                          # n x (1 + 324) x 384, final norm
            cls, pm = h[:, 0], h[:, 1:].mean(1)
        return cls.float().cpu().numpy(), pm.float().cpu().numpy()
    return embed


def run_all(drive, embed, dev):
    check_review_config()
    t = time.time()
    logs = [embed_drive(d, fr, embed, dev) for d, fr in load_frames(drive).items()]
    logs = [l for l in logs if l]
    if logs:
        log = PROC / "embed_log.csv"
        pd.DataFrame(logs).assign(when=time.strftime("%Y-%m-%d %H:%M")).to_csv(
            log, mode="a", header=not log.exists(), index=False)
        n = sum(l["frames"] for l in logs)
        print(f"\nEMBED DONE: {n:,} frames in {time.time() - t:.0f} s ({(time.time() - t) / n:.3f} s/frame)")
    m = PROC / "mask_log.csv"
    if m.exists() and logs:
        ml = pd.read_csv(m)
        ml = ml[ml["frames_masked"] > 50]
        if len(ml):
            mask_spf = float((ml["seconds"].sum()) / ml["frames_masked"].sum())
            emb_spf = sum(l["seconds"] for l in logs) / sum(l["frames"] for l in logs)
            print(f"V4 (mask + embed): mask {mask_spf:.2f} + embed {emb_spf:.2f} = {mask_spf + emb_spf:.2f} s/frame "
                  f"-> 1,440 frames {1440 * (mask_spf + emb_spf) / 60:.0f} min | 3,000 frames "
                  f"{3000 * (mask_spf + emb_spf) / 60:.0f} min (pass: whole dataset < 60 min)")


def bench(dev, n):
    import torch
    embed = make_embed(dev)
    frs = load_frames()
    d, fr = next(iter(frs.items()))
    fr = fr.sample(min(n, len(fr)), random_state=0)
    t = time.time()
    xs = [prep_frame(ROOT / f, mask_path(d, f))[0] for f in fr["file"]]
    prep_s = (time.time() - t) / len(xs)
    embed(xs[0])
    for label, groups in [("1 frame (7 tiles) per pass", [[x] for x in xs]),
                          ("4 frames (28 tiles) per pass", [xs[i:i + 4] for i in range(0, len(xs), 4)])]:
        embed(np.concatenate(groups[0]))                                         # warm-up for this shape
        if dev == "mps":
            torch.mps.synchronize()
        t = time.time()
        for g in groups:
            embed(np.concatenate(g))
        print(f"  {label}: {(time.time() - t) / len(xs):.3f} s/frame")
    print(f"  frame prep (JPEG decode + tiles + hand-crafted), one thread: {prep_s:.3f} s/frame")


def selftest(embed, dev):
    """synthetic frames + class maps in a temp project; checks shapes, files, resume, and batch consistency"""
    global ROOT, PROC, EMB
    real, saved = ROOT, (ROOT, PROC, EMB)
    try:
        _selftest_body(embed, dev, real)
    finally:
        ROOT, PROC, EMB = saved                                                  # never leave paths pointing at temp


def _selftest_body(embed, dev, real):
    global ROOT, PROC, EMB
    with tempfile.TemporaryDirectory() as tmp:
        ROOT, PROC, EMB = Path(tmp), Path(tmp) / "data/processed", Path(tmp) / "data/embeddings"
        PROC.mkdir(parents=True)
        rng = np.random.default_rng(0)
        files = []
        for i in range(6):
            a = rng.integers(40, 120, (2160, 3840, 3), dtype=np.uint8)
            p = ROOT / f"data/frames/TEST/c_n{i:05d}.jpg"; p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(a).save(p, quality=90); files.append(str(p.relative_to(ROOT)))
            lab = np.zeros((1024, 2048), np.uint8); lab[:, 1000 + 40 * i:] = 1         # road on the left part
            m = ROOT / f"data/masks/TEST/c_n{i:05d}.png"; m.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(lab).save(m)
        pd.DataFrame(dict(frame_id=range(6), file=files, seg_key=["S1"] * 3 + ["S2"] * 3)).to_csv(
            PROC / "frames_TEST.csv", index=False)
        run_all(None, embed, dev)
        z = np.load(EMB / "tiles_TEST.npz"); tab = pd.read_csv(EMB / "tiles_TEST.csv")
        assert z["cls"].shape == (42, DIM) and z["pmean"].shape == (42, DIM) and len(tab) == 42, z["cls"].shape
        assert np.isfinite(z["cls"]).all() and np.isfinite(z["pmean"]).all()
        x = prep_frame(ROOT / files[0], ROOT / "data/masks/TEST/c_n00000.png")[0]
        one = np.concatenate([embed(x[i:i + 1])[0] for i in range(len(x))])
        many = embed(x)[0]
        diff = float(np.abs(one - many).max() / (np.abs(one).max() + 1e-9))
        run_all(None, embed, dev)                                                    # must skip
        print(f"\nSELFTEST {'OK' if diff < 1e-2 else 'WARNING'} | 6 frames -> {len(tab)} tiles x {DIM} | "
              f"one-by-one vs batched CLS: max relative difference {diff:.1e} | road shares "
              f"{tab['road_share'].min():.2f}-{tab['road_share'].max():.2f} | nothing written to {real.name}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DINOv2-small embeddings of every near-field tile (+ rung-3 features)")
    ap.add_argument("--drive")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", type=int, metavar="N")
    a = ap.parse_args()
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if a.bench:
        bench(dev, a.bench)
    elif a.selftest:
        selftest(make_embed(dev), dev)
    else:
        run_all(a.drive, make_embed(dev), dev)
