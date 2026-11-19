# 04_pipeline_design/v3_review.py
# Viability check V3, part B: choose and FREEZE the near-field strip, the tile rule (M1, M1b) and the
# end-of-segment trim, from the cached class maps (road_masks.py). Reads only; writes review files under data/.
#
# Geometry (M1): tiles are 252 x 252 px at NATIVE 4K resolution (18 x 18 DINOv2 patches of 14 px), laid in
# whole rows upward from the strip bottom, centered between the left/right limits. A tile's road share is
# read from the 2048 x 1024 class map (which covers rows 0-90% of the frame).
#
# Subcommands (project root, pavement env active):
#   survey                          compare candidate strips: tiles kept, frames/segments surviving, by PCI band
#   sheets --rows R --bottom B --x L R2 [--seed 592]
#                                   50 random frames -> 5 review pages of 10 strips each (blind: no PCI, no
#                                   distance-to-end shown); you list the bad-mask and paint frame numbers
#   score  --mask-bad 7,23 --paint 3,12,40     V3 pass/fail (>= 45/50) + paint frequency by distance to the end ahead
#   audit  --rows R --bottom B --x L R2 [--tag T]   survivorship audit (M1b): rejection vs PCI band, width, drive;
#                                   which classes take the non-road pixels; a worst-PCI sheet to inspect by eye
#                                   (--tag T writes audit_frames_T.csv / audit_worst_pci_T.jpg, keeping earlier audits)
#   V3-local (Day 4C, pre-registered 2026-11-14): the same blind review on ONE new drive
#   sheets --drive S --seed N       20 random clean, non-alley frames of drive S -> 2 pages; frozen strip; seed must be
#                                   the drive's pre-registered seed (or + 20 for a re-draw after a logged fix)
#   score  --drive S --seed N --mask-bad .. --paint ..   pass >= 18/20; files in data/v3_review/local/<S>_seed<N>/
# Every subcommand reads CLEAN frames only: the <S>__stale tables (stale set, H3/V7) are never part of V3.
# Writes: data/v3_review/*.jpg, *.csv (git-ignored)

import argparse, json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

TILE = 252                      # px at native resolution (multiple of 14)
ROAD_MIN = 0.98                 # M1b starting value: keep a tile if >= 98% road
MIN_TILES = 2                   # M1b starting value: keep a frame if >= 2 tiles survive
MIN_FRAMES = 3                  # plan section 6: usable segment needs >= 3 frames
HOOD_FRAC = 0.10                # class maps cover the top 90% of the frame (road_masks.py)
SEED = 592                      # V3 draw (V0 = 590, V2b = 591); fixed before the run
N_REVIEW = 50
PASS_AT = 45
# V3-local, pre-registered 2026-11-14 (Day 4C) before any class map of these drives existed: same blind rules as V3,
# 20 random frames from the drive's clean NON-ALLEY frames (alleys are out of every analysis set), pass >= 18/20.
# A re-draw after a logged fix uses the registered seed + 20. Seed = the drive's V2 seed + 10.
LOCAL_SEEDS = {"11-12-2026-Golden-Hill": 603, "11-13-2026-SE-SD": 604, "11-14-2026-Encanto": 606,
               "11-15-2026-Uptown": 607}   # Uptown: pre-registered 2026-11-15 (Day 5) with V2 seed 597, before any frame existed
LOCAL_REGISTERED = {"11-15-2026-Uptown": "2026-11-15 (Day 5)"}   # default for the others: "2026-11-14 (Day 4C)"
LOCAL_N, LOCAL_PASS, LOCAL_REDRAW = 20, 18, 20
FROZEN_STRIP = (1, 0.90, 0.10, 0.60)   # rows, bottom, x-left, x-right: pipeline_rules.json, frozen 2026-11-14
LEGACY_STRIP = (1, 0.90, 0.05, 0.65)   # this script's original CLI defaults (kept, so earlier commands behave the same)
KEY = "_review_key_DO_NOT_OPEN.csv"
CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
           "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
           "motorcycle", "bicycle"]
BANDS = ["Good", "Satisfactory", "Fair", "Poor", "Very Poor", "Serious", "Failed"]
BAND3 = {"Good": "Good/Sat", "Satisfactory": "Good/Sat", "Fair": "Fair/Poor", "Poor": "Fair/Poor",
         "Very Poor": "VPoor-Failed", "Serious": "VPoor-Failed", "Failed": "VPoor-Failed"}
# candidate strips for the survey: (tile rows, strip bottom as share of frame height, x-left, x-right).
# Chosen from the Run A road heatmap (2026-11-14): road is reliable (>= 0.87 of frames) in rows 80-90% from the
# left edge to ~60% of the width; the right side (curb, parked cars, hood curve) and rows above ~78% are not.
CANDIDATES = [(1, 0.90, 0.05, 0.65), (1, 0.90, 0.10, 0.60), (1, 0.90, 0.05, 0.75), (1, 0.90, 0.00, 0.70),
              (1, 0.90, 0.15, 0.85), (1, 0.88, 0.05, 0.65), (2, 0.90, 0.05, 0.65)]

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
PROC = ROOT / "data/processed"
OUT = ROOT / "data/v3_review"


def stop(msg):
    sys.exit(f"STOP: {msg}")


# ------------------------------------------------------------------------------------------- geometry
def tile_boxes(W, H, rows, bottom, xl, xr):
    """native-pixel boxes (x0, y0, x1, y1) of the tile grid; rows stack upward from bottom*H"""
    yb = int(round(bottom * H))
    if yb > int(H * (1 - HOOD_FRAC)):
        stop(f"strip bottom {bottom:.0%} is inside the hood crop (> {1 - HOOD_FRAC:.0%})")
    x0, x1 = int(xl * W), int(xr * W)
    ncol = (x1 - x0) // TILE
    off = x0 + ((x1 - x0) - ncol * TILE) // 2
    return [(off + c * TILE, yb - (r + 1) * TILE, off + (c + 1) * TILE, yb - r * TILE)
            for r in range(rows) for c in range(ncol)]


def tile_stats(lab, boxes, W, H):
    """per tile: road share + class counts of the non-road pixels, read from the class map"""
    mh, mw = lab.shape
    keep_h = int(H * (1 - HOOD_FRAC))
    out = []
    for (x0, y0, x1, y1) in boxes:
        r0, r1 = int(round(y0 * mh / keep_h)), int(round(y1 * mh / keep_h))
        c0, c1 = int(round(x0 * mw / W)), int(round(x1 * mw / W))
        t = lab[r0:r1, c0:c1]
        out.append((float((t == 0).mean()), np.bincount(t.ravel(), minlength=19)))
    return out


# ------------------------------------------------------------------------------------- frames + masks
def ahead_behind(fr, fx):
    """distance (m) to the segment end the car drives toward / away from, per frame, by interpolating Gate 0's
    per-fix position along the segment (along_m) between the two fixes that bracket the frame in time"""
    secs = lambda s: ((pd.to_datetime(s) - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).to_numpy(float)
    fx = fx.assign(t=secs(fx["local"])).sort_values("t").reset_index(drop=True)
    tt = fx["t"].to_numpy()
    ft = secs(fr["local"])
    i = np.clip(np.searchsorted(tt, ft, side="right") - 1, 0, len(fx) - 2)
    a, b = fx.iloc[i].reset_index(drop=True), fx.iloc[i + 1].reset_index(drop=True)
    f = np.clip((ft - tt[i]) / np.maximum(tt[i + 1] - tt[i], 1e-9), 0, 1)
    same = (a["seg_key"].to_numpy() == fr["seg_key"].to_numpy()) & (b["seg_key"].to_numpy() == fr["seg_key"].to_numpy())
    along = a["along_m"] + f * (b["along_m"] - a["along_m"])
    L = a["seg_len_m"]
    fwd = (b["along_m"] >= a["along_m"]).to_numpy()
    ahead = np.where(fwd, L - along, along)
    behind = np.where(fwd, along, L - along)
    return np.where(same, ahead, np.nan), np.where(same, behind, np.nan)


def load_frames(drive=None):
    """CLEAN frames (the V3 population): every frames_<S>.csv except the <S>__stale pseudo-drives (stale labels may
    describe the street before a repave, and those frames never train). With a drive: only frames_<drive>.csv,
    minus alley frames (V3-local population: frames that can enter an analysis set)."""
    if drive:
        if drive.endswith("__stale"):
            stop("V3-local reviews clean frames only, not a __stale table")
        tables = [PROC / f"frames_{drive}.csv"]
        if not tables[0].exists():
            stop(f"missing data/processed/frames_{drive}.csv: run sample_frames.py plan {drive} first")
    else:
        found = sorted(PROC.glob("frames_*.csv"))
        tables = [t for t in found if not t.stem.endswith("__stale")]
        if len(tables) < len(found):
            print(f"note: {len(found) - len(tables)} __stale table(s) left out (V3 reads clean frames only)")
    if not tables:
        stop("no frames_<SESSION>.csv found")
    parts = []
    for t in tables:
        fr = pd.read_csv(t)
        name = t.stem[len("frames_"):]
        fx_f = PROC / f"gate0_fixes_{name}.csv"
        if fx_f.exists():
            fr["ahead_m"], fr["behind_m"] = ahead_behind(fr, pd.read_csv(fx_f, index_col=0))
        else:
            fr["ahead_m"] = fr["behind_m"] = np.nan
            print(f"note: {fx_f.name} missing -> no distance-to-end for {name}")
        parts.append(fr)
    fr = pd.concat(parts, ignore_index=True)
    if drive:                                   # frozen alley rule (M4): rd20full contains the word ALLEY
        alley = fr["rd20full"].astype(str).str.contains(r"\bALLEY\b", case=False, regex=True)
        print(f"{drive}: {len(fr):,} clean frames, {int(alley.sum()):,} on alleys left out -> {int((~alley).sum()):,} to draw from")
        fr = fr[~alley].reset_index(drop=True)
    fr["mask"] = [str(ROOT / "data/masks" / d / f"{Path(f).stem}.png") for d, f in zip(fr["drive"], fr["file"])]
    have = np.array([Path(m).exists() for m in fr["mask"]])
    if not have.all():
        stop(f"{int((~have).sum())} of {len(fr)} frames have no class map yet: run road_masks.py first")
    return fr


def read_lab(path):
    return np.asarray(Image.open(path))


def frame_size(fr):
    W, H = Image.open(ROOT / fr["file"].iloc[0]).size
    return W, H


# --------------------------------------------------------------------------------------------- survey
def evaluate(fr, shares, W, H, cand):
    rows, bottom, xl, xr = cand
    boxes = tile_boxes(W, H, rows, bottom, xl, xr)
    kept = shares >= ROAD_MIN                                                           # frames x tiles
    nk = kept.sum(1)
    d = fr.assign(n_kept=nk, frame_ok=nk >= MIN_TILES)
    seg = d[d["frame_ok"]].groupby("seg_key").size()
    prim = d[d["is_primary"] == 1]
    pseg_all = prim["seg_key"].nunique()
    pseg_ok = int((prim[prim["frame_ok"]].groupby("seg_key").size() >= MIN_FRAMES).sum())
    res = dict(strip=f"{rows} row{'s' if rows > 1 else ''}, rows {bottom - rows * TILE / H:.0%}-{bottom:.0%}, x {xl:.0%}-{xr:.0%}",
               tiles_per_frame=len(boxes), tiles_kept=f"{kept.mean():.0%}",
               median_kept=int(np.median(nk)), frames_ok=f"{d['frame_ok'].mean():.0%}",
               primary_segs_ge3=f"{pseg_ok}/{pseg_all}")
    for b3 in ["Good/Sat", "Fair/Poor", "VPoor-Failed"]:
        m = d["pci23_d"].map(BAND3).eq(b3).to_numpy()
        res[f"kept {b3}"] = f"{kept[m].mean():.0%}" if m.any() else "-"
    return res


def survey():
    fr = load_frames()
    W, H = frame_size(fr)
    print(f"{len(fr):,} frames ({W}x{H}) | reading class maps (one pass) ...")
    boxes = [tile_boxes(W, H, *c) for c in CANDIDATES]
    shares = [np.zeros((len(fr), len(b))) for b in boxes]
    for i, m in enumerate(fr["mask"]):
        lab = read_lab(m)
        for k, b in enumerate(boxes):
            shares[k][i] = [s for s, _ in tile_stats(lab, b, W, H)]
    res = pd.DataFrame([evaluate(fr, shares[k], W, H, c) for k, c in enumerate(CANDIDATES)])
    print(f"\nSTRIP SURVEY (tile {TILE} px native; tile kept if >= {ROAD_MIN:.0%} road; frame kept if >= "
          f"{MIN_TILES} tiles; segment usable if >= {MIN_FRAMES} kept frames)")
    print(res.to_string(index=False))
    print("\n'kept <band>' = share of tiles kept, by PCI band group (survivorship early look; not a model result)")
    a = fr["ahead_m"]
    if a.notna().any():
        prim = fr[fr["is_primary"] == 1]
        print("\nEND TRIM: frames dropped if the end the car drives toward is closer than T m (primary frames)")
        for T in (15, 20, 25, 30, 35):
            k = prim[prim["ahead_m"] >= T]
            print(f"  T={T:2d} m: {len(k):,}/{len(prim):,} frames | primary segments with >= {MIN_FRAMES} frames: "
                  f"{int((k.groupby('seg_key').size() >= MIN_FRAMES).sum())}")
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "strip_survey.csv", index=False)
    print(f"\nsaved: data/v3_review/strip_survey.csv")


# --------------------------------------------------------------------------------------------- sheets
def font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def strip_panel(path_img, lab, boxes, W, H, width=1920):
    """the strip (+ half a tile of context above) with the road tinted magenta and tiles outlined
    green = kept (>= ROAD_MIN road) / red = rejected; road % printed in each tile"""
    im = Image.open(path_img).convert("RGB")
    st = tile_stats(lab, boxes, W, H)
    keep_h = int(H * (1 - HOOD_FRAC))
    y_top = max(0, min(b[1] for b in boxes) - TILE // 2)
    y_bot = max(b[3] for b in boxes)
    x_l, x_r = 0, W
    road = np.asarray(Image.fromarray((lab == 0).astype(np.uint8) * 255).resize((W, keep_h), Image.NEAREST)) > 0
    a = np.asarray(im)[y_top:y_bot].copy()
    m = road[y_top:y_bot]
    a[m] = (0.65 * a[m] + 0.35 * np.array([255, 0, 255])).astype(np.uint8)
    p = Image.fromarray(a)
    d = ImageDraw.Draw(p)
    f = font(44)
    for (x0, y0, x1, y1), (s, _) in zip(boxes, st):
        col = (0, 255, 0) if s >= ROAD_MIN else (255, 40, 40)
        d.rectangle([x0, y0 - y_top, x1 - 1, y1 - y_top - 1], outline=col, width=6)
        d.text((x0 + 10, y0 - y_top + 8), f"{100 * s:.0f}", fill=col, font=f)
    sc = width / W
    return p.resize((width, int(p.height * sc)), Image.BILINEAR), st


def review_setup(drive, seed):
    """V3 (all clean frames, 50, >= 45, data/v3_review/) or V3-local (one drive, 20, >= 18, its own folder)"""
    if drive is None:
        return dict(label="V3", n=N_REVIEW, pass_at=PASS_AT, out=OUT)
    if drive not in LOCAL_SEEDS:
        stop(f"{drive} has no pre-registered V3-local seed (registered: {', '.join(LOCAL_SEEDS)})")
    base = LOCAL_SEEDS[drive]
    if seed not in (base, base + LOCAL_REDRAW):
        stop(f"--seed for {drive} must be its pre-registered {base} (or {base + LOCAL_REDRAW} only for a re-draw "
             f"after a logged fix); got {seed}")
    return dict(label=f"V3-LOCAL {drive}", n=LOCAL_N, pass_at=LOCAL_PASS, out=OUT / "local" / f"{drive}_seed{seed}")


def sheets(rows, bottom, xl, xr, seed, drive=None):
    rv = review_setup(drive, seed)
    out, n = rv["out"], rv["n"]
    if drive and (out / KEY).exists():
        stop(f"{out.relative_to(ROOT)} already holds this draw: a pre-registered draw is made once")
    fr = load_frames(drive)
    if len(fr) < n:
        stop(f"only {len(fr)} frames to draw from (need {n})")
    W, H = frame_size(fr)
    boxes = tile_boxes(W, H, rows, bottom, xl, xr)
    pick = fr.sample(n, random_state=seed).reset_index(drop=True)
    pick.index = range(1, n + 1)
    out.mkdir(parents=True, exist_ok=True)
    thumb_w, f_big = 400, font(48)
    for page in range(n // 10):
        rows_img = []
        for k in range(page * 10 + 1, page * 10 + 11):
            r = pick.loc[k]
            lab = read_lab(r["mask"])
            sp, _ = strip_panel(ROOT / r["file"], lab, boxes, W, H)
            th = Image.open(ROOT / r["file"]).convert("RGB").resize((thumb_w, int(thumb_w * H / W)))
            dt = ImageDraw.Draw(th)
            s = thumb_w / W
            dt.rectangle([min(b[0] for b in boxes) * s, min(b[1] for b in boxes) * s,
                          max(b[2] for b in boxes) * s, max(b[3] for b in boxes) * s], outline=(0, 255, 0), width=2)
            h = max(sp.height, th.height)
            row = Image.new("RGB", (thumb_w + sp.width, h + 8), (30, 30, 30))
            row.paste(th, (0, 0)); row.paste(sp, (thumb_w, 0))
            ImageDraw.Draw(row).rectangle([0, 0, 110, 60], fill=(0, 0, 0))
            ImageDraw.Draw(row).text((8, 4), f"#{k}", fill=(255, 255, 0), font=f_big)
            rows_img.append(row)
        pg = Image.new("RGB", (rows_img[0].width, sum(x.height for x in rows_img)))
        y = 0
        for x in rows_img:
            pg.paste(x, (0, y)); y += x.height
        pg.save(out / f"review_page_{page + 1}.jpg", quality=88)
    cfg = dict(rows=rows, bottom=bottom, x_left=xl, x_right=xr, tile=TILE, road_min=ROAD_MIN, seed=seed)
    if drive:
        cfg.update(drive=drive, n=n, pass_at=rv["pass_at"], population="clean non-alley frames of the drive",
                   preregistered=LOCAL_REGISTERED.get(drive, "2026-11-14 (Day 4C)"))
    (out / "review_config.json").write_text(json.dumps(cfg, indent=1))
    pick[["drive", "file", "seg_key", "pci23", "pci23_d", "ahead_m", "behind_m"]].to_csv(out / KEY, index_label="num")
    where = out.relative_to(ROOT)
    if drive:
        print(f"{rv['label']} REVIEW (seed {seed}, pre-registered): {n // 10} pages x 10 frames in "
              f"{where}/review_page_1-{n // 10}.jpg  (pass >= {rv['pass_at']}/{n})")
    else:
        print(f"V3 REVIEW (seed {seed}, fixed before the run): 5 pages x 10 frames in data/v3_review/review_page_1-5.jpg")
    print(f"strip: {rows} tile row(s), bottom {bottom:.0%}, x {xl:.0%}-{xr:.0%}; {len(boxes)} tiles per frame")
    print("Left: whole frame (green box = strip). Right: the strip, road tinted magenta, tiles green = kept / red = rejected.")
    print("For each #, answer two questions (pre-registered rules):")
    print("  MASK BAD if a GREEN tile contains anything that is not the road surface (car, curb, sidewalk, driveway")
    print("     apron, hood, vegetation, a windshield reflection hiding the road), OR the magenta misses a large piece")
    print("     of road (about a tile or more) inside the strip.")
    print("  PAINT if any GREEN tile contains intersection paint: a STOP line, crosswalk, or STOP / word legend")
    print("     (lane lines along the street, e.g. the center line, do NOT count; note them separately if you like).")
    print("  Unsure whether a mask is acceptable -> count it BAD (tie-break).")
    if drive:
        print(f"Then:  python 04_pipeline_design/v3_review.py score --drive {drive} --seed {seed} "
              f"--mask-bad <nums> --paint <nums>   (use 0 for none)")
    else:
        print("Then:  python 04_pipeline_design/v3_review.py score --mask-bad <nums> --paint <nums>   (use 0 for none)")


# ---------------------------------------------------------------------------------------------- score
def nums(s, n=N_REVIEW):
    v = [int(x) for x in str(s).replace(" ", "").split(",") if x not in ("", "0")]
    if any(not 1 <= x <= n for x in v):
        stop(f"frame numbers must be 1-{n}")
    return sorted(set(v))


def score(mask_bad, paint, drive=None, seed=None):
    rv = review_setup(drive, seed)
    out, n, pass_at = rv["out"], rv["n"], rv["pass_at"]
    if not (out / KEY).exists():
        stop(f"no review key in {out.relative_to(ROOT)}: run sheets first")
    if drive and (out / "v3_result.csv").exists():
        stop(f"{out.relative_to(ROOT)}/v3_result.csv exists: this draw is already scored. To correct a typo, rename "
             f"the old v3_result.csv first (keep both on record), then score again")
    key = pd.read_csv(out / KEY, index_col="num")
    if len(key) != n:
        stop(f"the key holds {len(key)} frames, expected {n}")
    bad, pnt = nums(mask_bad, n), nums(paint, n)
    key["mask_ok"] = ~key.index.isin(bad)
    key["paint"] = key.index.isin(pnt)
    ok = int(key["mask_ok"].sum())
    print(f"{rv['label']} MASK CHECK: {ok}/{n} acceptable -> {'PASS' if ok >= pass_at else 'FAIL'} (pre-registered >= {pass_at}/{n})")
    print(f"\nINTERSECTION PAINT in a kept tile: {len(pnt)}/{n} frames ({len(pnt) / n:.0%})")
    if key["ahead_m"].notna().any():
        bins = [0, 20, 25, 30, 40, 1e9]
        key["ahead_bin"] = pd.cut(key["ahead_m"], bins, right=False,
                                  labels=["15-20", "20-25", "25-30", "30-40", "40+"])
        t = key.groupby("ahead_bin", observed=False).agg(frames=("paint", "size"), with_paint=("paint", "sum"))
        print("by distance to the segment end AHEAD (m):")
        print(t.to_string())
        if len(pnt):
            print(f"paint frames' distance ahead (m): {', '.join(f'{x:.0f}' for x in key.loc[key['paint'], 'ahead_m'])}")
    print("\nbad-mask frames (for the record):", ", ".join(f"#{i} {key.at[i, 'pci23_d']}" for i in bad) or "none")
    key.to_csv(out / "v3_result.csv", index_label="num")
    print(f"saved: {out.relative_to(ROOT)}/v3_result.csv")


# ---------------------------------------------------------------------------------------------- audit
def audit(rows, bottom, xl, xr, tag=None):
    sfx = f"_{tag}" if tag else ""
    fr = load_frames()
    W, H = frame_size(fr)
    boxes = tile_boxes(W, H, rows, bottom, xl, xr)
    recs, nonroad = [], np.zeros(19)
    nonroad_band = {b: np.zeros(19) for b in BANDS}
    for r in fr.itertuples(index=False):
        st = tile_stats(read_lab(r.mask), boxes, W, H)
        sh = np.array([s for s, _ in st])
        for s, cnt in st:
            nonroad += cnt * (np.arange(19) != 0)
            if r.pci23_d in nonroad_band:
                nonroad_band[r.pci23_d] += cnt * (np.arange(19) != 0)
        recs.append(dict(tiles_kept=float((sh >= ROAD_MIN).mean()), frame_ok=int((sh >= ROAD_MIN).sum() >= MIN_TILES),
                         mean_road=float(sh.mean())))
    d = pd.concat([fr.reset_index(drop=True), pd.DataFrame(recs)], axis=1)

    sc_f = PROC / "segments_clean.csv"
    if sc_f.exists():
        sc = pd.read_csv(sc_f, low_memory=False)
        key = next((c for c in ("seg_key", "iamfloc") if c in sc.columns), None)
        if key and "pwidth" in sc.columns:
            w = sc[[key, "pwidth"]].drop_duplicates(key).rename(columns={key: "seg_key"})
            w["seg_key"] = w["seg_key"].astype(str).str.strip().str.upper()
            d = d.merge(w, on="seg_key", how="left")
    print(f"SURVIVORSHIP AUDIT (M1b) | strip {rows} row(s), bottom {bottom:.0%}, x {xl:.0%}-{xr:.0%} | {len(d):,} frames")
    by = d.groupby("pci23_d").agg(frames=("frame_ok", "size"), tiles_kept=("tiles_kept", "mean"),
                                  frames_kept=("frame_ok", "mean"), mean_road_share=("mean_road", "mean"))
    print("\nby PCI band:")
    print(by.reindex([b for b in BANDS if b in by.index]).round(3).to_string())
    rho = d[["pci23", "tiles_kept"]].corr(method="spearman").iloc[0, 1]
    print(f"Spearman(PCI, share of tiles kept) over frames: {rho:+.2f}  (near 0 = rejection does not track condition)")
    if "pwidth" in d.columns and d["pwidth"].notna().any():
        d["width_bin"] = pd.qcut(d["pwidth"], 3, duplicates="drop")
        print("\nby street width (ft, tertiles):")
        print(d.groupby("width_bin", observed=True).agg(frames=("frame_ok", "size"), tiles_kept=("tiles_kept", "mean"),
                                                         frames_kept=("frame_ok", "mean")).round(3).to_string())
    else:
        print("\n(no segments_clean.csv with pwidth found -> width audit skipped)")
    print("\nby drive:")
    print(d.groupby("drive").agg(frames=("frame_ok", "size"), tiles_kept=("tiles_kept", "mean"),
                                 frames_kept=("frame_ok", "mean")).round(3).to_string())
    seg = d[d["is_primary"] == 1].groupby(["seg_key", "pci23_d"])["frame_ok"].sum().reset_index()
    seg["usable"] = seg["frame_ok"] >= MIN_FRAMES
    t = seg.groupby("pci23_d")["usable"].agg(["size", "sum"])
    print("\nprimary segments usable (>= 3 kept frames) by band:")
    print(t.reindex([b for b in BANDS if b in t.index]).rename(columns={"size": "segments", "sum": "usable"}).to_string())

    def top(v):
        v = v / max(v.sum(), 1)
        o = np.argsort(v)[::-1][:4]
        return ", ".join(f"{CLASSES[i]} {v[i]:.0%}" for i in o if v[i] > 0.005)
    print("\nwhat the NON-road pixels inside the strip are called (all frames):", top(nonroad))
    for b in BANDS:
        if nonroad_band[b].sum():
            print(f"  {b:12s}: {top(nonroad_band[b])}")
    print("(if bad bands show more 'sidewalk'/'terrain' inside the lane, SegFormer may be calling distress non-road)")

    worst = d[d["pci23_d"].isin(["Serious", "Failed", "Very Poor"])]
    if len(worst):
        pick = worst.sample(min(10, len(worst)), random_state=SEED)
        imgs = []
        for i, r in enumerate(pick.itertuples(index=False), 1):
            sp, _ = strip_panel(ROOT / r.file, read_lab(r.mask), boxes, W, H)
            ImageDraw.Draw(sp).text((10, 6), f"W{i} {r.pci23_d} {r.pci23:.0f}", fill=(255, 255, 0), font=font(40))
            imgs.append(sp)
        pg = Image.new("RGB", (imgs[0].width, sum(x.height for x in imgs)))
        y = 0
        for x in imgs:
            pg.paste(x, (0, y)); y += x.height
        OUT.mkdir(parents=True, exist_ok=True)
        pg.save(OUT / f"audit_worst_pci{sfx}.jpg", quality=88)
        print(f"\nsaved: data/v3_review/audit_worst_pci{sfx}.jpg  (look: are cracks/potholes/patches left un-tinted or in red tiles?)")
    d.drop(columns=["mask"]).to_csv(OUT / f"audit_frames{sfx}.csv", index=False)
    print(f"saved: data/v3_review/audit_frames{sfx}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="V3: strip survey, review sheets, scoring, survivorship audit")
    ap.add_argument("step", choices=["survey", "sheets", "score", "audit"])
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--bottom", type=float, default=None)
    ap.add_argument("--x", type=float, nargs=2, default=None, metavar=("LEFT", "RIGHT"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mask-bad", default="0")
    ap.add_argument("--paint", default="0")
    ap.add_argument("--drive", help="V3-local: review ONE new drive (sheets / score; pre-registered seed required)")
    ap.add_argument("--tag", help="audit only: suffix for the output files, so earlier audits are kept")
    a = ap.parse_args()
    if a.tag and a.step != "audit":
        stop("--tag is for audit only")
    if a.drive:
        if a.step not in ("sheets", "score"):
            stop("--drive is for sheets and score only")
        if a.seed is None:
            stop(f"--drive needs --seed (the drive's pre-registered seed)")
        given = (a.rows, a.bottom) + (tuple(a.x) if a.x else (None, None))
        if any(g is not None and abs(g - f) > 1e-9 for g, f in zip(given, FROZEN_STRIP)):
            stop(f"V3-local uses the frozen strip {FROZEN_STRIP} (rows, bottom, x-left, x-right); leave the strip options out")
        rows, bottom, xl, xr = FROZEN_STRIP
        seed = a.seed
    else:
        rows = a.rows if a.rows is not None else LEGACY_STRIP[0]
        bottom = a.bottom if a.bottom is not None else LEGACY_STRIP[1]
        xl, xr = tuple(a.x) if a.x else LEGACY_STRIP[2:]
        seed = a.seed if a.seed is not None else SEED
    if a.step == "survey":
        survey()
    elif a.step == "sheets":
        sheets(rows, bottom, xl, xr, seed, a.drive)
    elif a.step == "score":
        score(a.mask_bad, a.paint, a.drive, seed if a.drive else None)
    else:
        audit(rows, bottom, xl, xr, a.tag)
