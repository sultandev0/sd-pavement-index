# 04_pipeline_design/build_segment_table.py
# Applies the FROZEN rules (04_pipeline_design/pipeline_rules.json) to the tile embeddings and builds the
# modeling tables: kept frames with their vectors, and one row per segment (the "bag" of its kept frames).
#
#   tile kept    : road share >= tile_rule.road_share_min (0.98)
#   frame kept   : >= tile_rule.min_tiles_per_frame (2) kept tiles AND distance to the segment end ahead
#                  >= end_trim_ahead_m (30 m; --trim overrides for the ablations 15 / 37)
#   frame vector : mean of its kept tiles, stored as cls (384) and patch_mean (384); hand-crafted = mean of kept tiles
#   segment      : usable if >= min_frames_per_segment (3) kept frames
#   sets         : alleys excluded everywhere; primary = clean, residential, non-alley; secondary = clean, non-alley
#   stale set    : frames from <SESSION>__stale tables (segments repaired on/after 2023-02-01) go through the SAME
#                  rules into separate stale_* files; set_stale = usable, residential, non-alley. Never trained on.
#
# Reads : pipeline_rules.json, data/embeddings/tiles_<SESSION>.csv/.npz, data/processed/frames_<SESSION>.csv,
#         data/processed/gate0_fixes_<SESSION>.csv, data/processed/clean_driven_segments.csv
# Writes: data/modeling/trim<T>/frames.csv       one row per kept frame (row i = vector row i)
#         data/modeling/trim<T>/frame_vectors.npz cls, patch_mean (n_frames x 384 each, float32)
#         data/modeling/trim<T>/segments.csv      one row per clean driven segment, with usable / set flags
#         data/modeling/trim<T>/stale_frames.csv, stale_frame_vectors.npz, stale_segments.csv   (V7 / H3)
# Usage (project root, pavement env active):
#   python 04_pipeline_design/build_segment_table.py            # frozen trim (30 m)
#   python 04_pipeline_design/build_segment_table.py --trim 15  # ablation tables, separate folder

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v3_review import ahead_behind                     # one definition of "distance to the end ahead"

ROOT = next(p for p in HERE.parents if (p / ".git").exists())
PROC, EMB = ROOT / "data/processed", ROOT / "data/embeddings"
BANDS = ["Good", "Satisfactory", "Fair", "Poor", "Very Poor", "Serious", "Failed"]


def stop(msg):
    sys.exit(f"STOP: {msg}")


def load_rules():
    f = HERE / "pipeline_rules.json"
    if not f.exists():
        stop(f"missing {f.relative_to(ROOT)}")
    r = json.loads(f.read_text())
    cfg = ROOT / "data/v3_review/review_config.json"            # the strip the embeddings were made with
    if cfg.exists():
        c, s = json.loads(cfg.read_text()), r["strip"]
        pairs = [(c["rows"], s["rows"]), (c["bottom"], s["bottom_frac"]), (c["x_left"], s["x_left_frac"]),
                 (c["x_right"], s["x_right_frac"]), (c["tile"], s["tile_px"])]
        if any(abs(float(a) - float(b)) > 1e-9 for a, b in pairs):
            stop("pipeline_rules.json strip differs from the reviewed strip (data/v3_review/review_config.json)")
    return r


def per_drive(drive, rules, trim):
    tab = pd.read_csv(EMB / f"tiles_{drive}.csv")
    z = np.load(EMB / f"tiles_{drive}.npz")
    cls, pm = z["cls"], z["pmean"]
    if not (len(tab) == len(cls) == len(pm)):
        stop(f"{drive}: tile table and vectors have different lengths")
    fr = pd.read_csv(PROC / f"frames_{drive}.csv")
    base = drive.split("__")[0]                                 # <SESSION>__stale uses the session's Gate 0 fixes
    fr["ahead_m"], _ = ahead_behind(fr, pd.read_csv(PROC / f"gate0_fixes_{base}.csv", index_col=0))
    if set(tab["file"]) != set(fr["file"]):
        stop(f"{drive}: embedded frames differ from frames_{drive}.csv; re-run embed_tiles.py")
    per = tab.groupby("file").size()
    if per.nunique() != 1 or per.iloc[0] != rules["strip"]["tiles_per_frame"]:
        stop(f"{drive}: expected {rules['strip']['tiles_per_frame']} tiles per frame, found {sorted(per.unique())}")

    tab["kept"] = tab["road_share"] >= rules["tile_rule"]["road_share_min"]
    k = tab["kept"].to_numpy()
    g = tab.groupby("file", sort=False)
    files = list(g.groups.keys())
    idx = [g.indices[f] for f in files]
    n_kept = np.array([k[i].sum() for i in idx])

    def mean_kept(arr):
        out = np.full((len(files), arr.shape[1]), np.nan, dtype=np.float32)
        for j, i in enumerate(idx):
            ii = i[k[i]]
            if len(ii):
                out[j] = arr[ii].mean(0)
        return out

    hc = {c: np.array([tab[c].to_numpy()[i[k[i]]].mean() if k[i].any() else np.nan for i in idx])
          for c in ("gray_mean", "gray_std", "lap_var")}
    f = pd.DataFrame(dict(file=files, n_tiles_kept=n_kept, **hc))
    f = f.merge(fr[["file", "drive", "seg_key", "rd20full", "pci23", "pci23_d", "cpname", "is_primary", "ahead_m"]],
                on="file", how="left", validate="one_to_one")
    f["tile_ok"] = f["n_tiles_kept"] >= rules["tile_rule"]["min_tiles_per_frame"]
    f["trim_ok"] = f["ahead_m"] >= trim                    # NaN distance -> dropped (counted below)
    f["kept"] = f["tile_ok"] & f["trim_ok"]
    return f, mean_kept(cls), mean_kept(pm)


def main(trim_arg):
    rules = load_rules()
    trim = float(rules["end_trim_ahead_m"] if trim_arg is None else trim_arg)
    min_fr = int(rules["min_frames_per_segment"])
    drives = sorted(p.stem[len("tiles_"):] for p in EMB.glob("tiles_*.npz"))
    if not drives:
        stop("no data/embeddings/tiles_<SESSION>.npz found: run embed_tiles.py")
    stale_drives = [d for d in drives if "__stale" in d]
    drives = [d for d in drives if "__stale" not in d]
    if not drives:
        stop("no training drives embedded")
    parts = [per_drive(d, rules, trim) for d in drives]
    f = pd.concat([p[0] for p in parts], ignore_index=True)
    cls = np.concatenate([p[1] for p in parts]); pm = np.concatenate([p[2] for p in parts])

    n = len(f)
    print(f"RULES: pipeline_rules.json (frozen {rules['frozen']}) | tile >= {rules['tile_rule']['road_share_min']:.0%} road, "
          f">= {rules['tile_rule']['min_tiles_per_frame']} tiles/frame | end trim {trim:g} m"
          f"{'' if trim_arg is None else ' (ABLATION, overrides the frozen value)'} | >= {min_fr} frames/segment")
    print(f"drives: {', '.join(drives)}")
    print(f"\nFRAMES: {n:,} sampled")
    print(f"  dropped by the tile rule (< {rules['tile_rule']['min_tiles_per_frame']} kept tiles): {int((~f['tile_ok']).sum()):,}")
    print(f"  dropped by the end trim (< {trim:g} m to the end ahead): {int((f['tile_ok'] & ~f['trim_ok']).sum()):,}"
          f"  (no distance available: {int(f['ahead_m'].isna().sum())})")
    print(f"  KEPT: {int(f['kept'].sum()):,} ({f['kept'].mean():.0%})")

    cl = pd.read_csv(PROC / "clean_driven_segments.csv")
    cl["seg_key"] = cl["seg_key"].astype(str).str.strip().str.upper()
    cl = (cl.sort_values("drive").groupby("seg_key", as_index=False)
            .agg(rd20full=("rd20full", "first"), pci23=("pci23", "first"), pci23_d=("pci23_d", "first"),
                 cpname=("cpname", "first"), pav_length=("pav_length", "first"),
                 last_repair_end=("last_repair_end", "first"), in_clean_set=("in_clean_set", "max"),
                 is_residential=("is_residential", "max"), is_alley=("is_alley", "max"),
                 drives=("drive", lambda s: ";".join(sorted(set(s))))))
    kf = f[f["kept"]].reset_index(drop=True)
    kv_cls, kv_pm = cls[f["kept"].to_numpy()], pm[f["kept"].to_numpy()]
    cnt = kf.groupby("seg_key").size().rename("n_frames_kept")
    seg = cl.merge(cnt, left_on="seg_key", right_index=True, how="left").fillna({"n_frames_kept": 0})
    seg["n_frames_kept"] = seg["n_frames_kept"].astype(int)
    seg["usable"] = seg["n_frames_kept"] >= min_fr
    clean, nonalley = seg["in_clean_set"].eq(1), seg["is_alley"].eq(0)
    seg["set_primary"] = seg["usable"] & clean & nonalley & seg["is_residential"].eq(1)
    seg["set_secondary"] = seg["usable"] & clean & nonalley
    unknown = set(kf["seg_key"]) - set(seg["seg_key"])
    if unknown:
        stop(f"{len(unknown)} kept-frame segments are not in clean_driven_segments.csv")

    print(f"\nSEGMENTS (clean driven, Gate 0): {len(seg)} | alleys excluded: {int(seg['is_alley'].eq(1).sum())}")
    print(f"  usable (>= {min_fr} kept frames): {int(seg['usable'].sum())} | secondary set (clean, non-alley): "
          f"{int(seg['set_secondary'].sum())} | PRIMARY (clean, residential, non-alley): {int(seg['set_primary'].sum())}")
    p = seg[seg["set_primary"]]
    lost = seg[clean & nonalley & seg["is_residential"].eq(1) & ~seg["usable"]]
    print(f"  primary candidates not usable: {len(lost)} "
          + (f"({', '.join(f'{r.rd20full} {r.pci23_d} {r.n_frames_kept} fr' for r in lost.itertuples())})" if len(lost) else ""))
    print("\nPRIMARY by band:  " + " | ".join(f"{b} {int((p['pci23_d'] == b).sum())}" for b in BANDS))
    print("PRIMARY by area:  " + " | ".join(f"{a} {c}" for a, c in p["cpname"].value_counts().items()))
    pf = p["n_frames_kept"]
    print(f"kept frames per primary segment: min {pf.min()}, median {pf.median():.0f}, max {pf.max()}")

    areas = p["cpname"].value_counts()
    ok_areas = int((areas >= 20).sum())
    gate = len(p) >= 150 and ok_areas >= 2
    print(f"\nV5 as pre-registered (Gate 0 clean): {len(seg)} clean segments, {seg['cpname'].nunique()} areas -> "
          f"{'PASS' if len(seg) >= 150 and seg['cpname'].nunique() >= 2 else 'FAIL'} (needs >= 150, >= 2 areas)")
    print(f"V6/V7 gate (frozen): {len(p)} primary usable, {ok_areas} areas with >= 20 -> "
          f"{'MET: V6/V7 may run' if gate else 'NOT MET: V6/V7 wait (need >= 150 across >= 2 areas with >= 20 each)'}")

    out = ROOT / f"data/modeling/trim{trim:g}"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["drive", "file", "seg_key", "rd20full", "cpname", "pci23", "pci23_d", "is_primary", "ahead_m",
            "n_tiles_kept", "gray_mean", "gray_std", "lap_var"]
    kf[cols].to_csv(out / "frames.csv", index_label="row")
    np.savez(out / "frame_vectors.npz", cls=kv_cls.astype(np.float32), patch_mean=kv_pm.astype(np.float32))
    seg.to_csv(out / "segments.csv", index=False)
    assert len(kf) == len(kv_cls) and np.isfinite(kv_cls).all() and np.isfinite(kv_pm).all()
    print(f"\nsaved: {out.relative_to(ROOT)}/frames.csv ({len(kf):,} rows), frame_vectors.npz "
          f"({kv_cls.shape[0]:,} x {kv_cls.shape[1]} cls + patch_mean), segments.csv ({len(seg)} rows)")

    if stale_drives:
        stale_set(stale_drives, rules, trim, min_fr, out)
    else:
        print("\nSTALE SET: no <SESSION>__stale embeddings yet (run sample_frames plan + extract <SESSION>__stale, "
              "road_masks, embed_tiles)")


def stale_set(stale_drives, rules, trim, min_fr, out):
    """same rules on the stale frames -> stale_* files (V7 / H3); never part of training"""
    parts = [per_drive(d, rules, trim) for d in stale_drives]
    f = pd.concat([p[0] for p in parts], ignore_index=True)
    cls = np.concatenate([p[1] for p in parts]); pm = np.concatenate([p[2] for p in parts])
    k = f["kept"].to_numpy()
    kf = f[k].reset_index(drop=True)
    sc = pd.read_csv(PROC / "segments_clean.csv", usecols=["iamfloc", "rd20full", "pci23", "pci23_d", "cpname",
                                                           "is_stale", "is_residential", "is_alley",
                                                           "last_repair_end", "pav_length"])
    sc["iamfloc"] = sc["iamfloc"].astype(str).str.strip().str.upper()
    keys = sorted(set(f["seg_key"].astype(str).str.strip().str.upper()))
    seg = sc[sc["iamfloc"].isin(keys)].drop_duplicates("iamfloc").rename(columns={"iamfloc": "seg_key"})
    if len(seg) != len(keys):
        stop("some stale-frame segments are missing from segments_clean.csv")
    if not seg["is_stale"].eq(1).all():
        stop("a segment in a __stale table is not stale in segments_clean.csv")
    cnt = kf.groupby("seg_key").size().rename("n_frames_kept")
    seg = seg.merge(cnt, left_on="seg_key", right_index=True, how="left").fillna({"n_frames_kept": 0})
    seg["n_frames_kept"] = seg["n_frames_kept"].astype(int)
    seg["usable"] = seg["n_frames_kept"] >= min_fr
    seg["set_stale"] = seg["usable"] & seg["is_residential"].eq(1) & seg["is_alley"].eq(0)
    seg["repaired_after_survey_end"] = pd.to_datetime(seg["last_repair_end"]) > pd.Timestamp("2023-08-31")
    s = seg[seg["set_stale"]]
    print(f"\nSTALE SET (V7 / H3; same rules, never trained on): {len(f):,} frames -> {int(k.sum()):,} kept | "
          f"{len(seg)} stale segments, {int(seg['usable'].sum())} usable, {len(s)} residential non-alley "
          f"(of which repaired after the survey ended, Aug 2023: {int(s['repaired_after_survey_end'].sum())})")
    print("STALE by band (2023 label): " + " | ".join(f"{b} {int((s['pci23_d'] == b).sum())}" for b in BANDS))
    kf.to_csv(out / "stale_frames.csv", index_label="row")
    np.savez(out / "stale_frame_vectors.npz", cls=cls[k].astype(np.float32), patch_mean=pm[k].astype(np.float32))
    seg.to_csv(out / "stale_segments.csv", index=False)
    print(f"saved: {out.relative_to(ROOT)}/stale_frames.csv ({len(kf):,} rows), stale_frame_vectors.npz, stale_segments.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="apply the frozen rules -> kept frames + segment bags")
    ap.add_argument("--trim", type=float, help="end-trim ablation (m); default = the frozen value")
    main(ap.parse_args().trim)
