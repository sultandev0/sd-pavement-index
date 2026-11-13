# 04_pipeline_design/v0_blind_test.py
# Viability check V0 (pre-registered): can a human tell good from bad pavement in these frames?
#   Take the 6 best and 6 worst CLEAN pilot segments (Mira Mesa), grab one frame from the middle of each,
#   shuffle them, and have Wael rate each frame Good or Bad WITHOUT seeing the City's score.
#   Pass if >= 10 of 12 are right (guessing averages 6).
# Usage (project root, pavement env active):
#   python 04_pipeline_design/v0_blind_test.py make     -> writes data/v0_blind/ (12 frames + rating sheet)
#   ... open the frames, fill in my_ratings.csv (G or B for each) ...
#   python 04_pipeline_design/v0_blind_test.py score    -> compares your ratings with the City's scores
# Writes only inside data/v0_blind/ (git-ignored: frames show plates and GPS text).
import sys, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT    = Path(__file__).resolve().parent.parent
SESSION = "10-25-2026-Mira-Mesa"
FOOTAGE = Path("/Users/waelsultan/dashcam_footage") / SESSION
OUT     = ROOT / "data/v0_blind"
KEY     = OUT / "_answer_key_DO_NOT_OPEN.csv"
SHEET   = OUT / "my_ratings.csv"
N_EACH, HOOD_FRAC, SHOW_W, SEED = 6, 0.10, 1920, 590

def make():
    if SHEET.exists():
        sys.exit(f"{SHEET} already exists - delete data/v0_blind/ first if you really want new frames")
    # 1) the 6 best and 6 worst clean pilot segments
    clean = pd.read_csv(ROOT / "data/processed/clean_driven_segments.csv", index_col=0)
    clean = clean[clean["drive"] == SESSION].sort_values("pci23")
    pick = pd.concat([clean.head(N_EACH).assign(truth="B"), clean.tail(N_EACH).assign(truth="G")])

    # 2) for each, the usable GPS fix closest to the middle of the segment
    fx  = pd.read_csv(ROOT / f"data/processed/gate0_fixes_{SESSION}.csv", index_col=0)
    trk = pd.read_csv(ROOT / f"data/gps_logs/{SESSION}/_track.csv")
    assert (fx["clip"].values == trk.loc[fx.index, "clip"].values).all(), "fix rows don't line up with the track"
    fx["video_t_s"] = trk.loc[fx.index, "video_t_s"]
    ok = fx[(fx["reason"] == "ok") & fx["seg_key"].isin(pick.index)].copy()
    ok["off_mid"] = (ok["along_m"] - ok["seg_len_m"] / 2).abs()
    mid = ok.sort_values("off_mid").drop_duplicates("seg_key").set_index("seg_key")
    pick = pick.join(mid[["clip", "video_t_s", "local", "speed_mph"]])
    assert pick["clip"].notna().all(), "a picked segment has no usable fix"

    # 3) shuffle, extract, crop the hood + burned-in text strip, save
    pick = pick.sample(frac=1, random_state=SEED)
    pick["frame"] = [f"frame_{i:02d}.jpg" for i in range(1, len(pick) + 1)]
    OUT.mkdir(parents=True, exist_ok=True)
    for seg, r in pick.iterrows():
        clip = next((FOOTAGE / f"{r['clip']}{ext}" for ext in (".MP4", ".mp4") if (FOOTAGE / f"{r['clip']}{ext}").exists()), None)
        assert clip is not None, f"video not found for clip {r['clip']} in {FOOTAGE}"
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "f.jpg"
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{max(r['video_t_s'], 0):.2f}",
                            "-i", str(clip), "-frames:v", "1", "-q:v", "2", str(tmp)], check=True)
            im = Image.open(tmp).convert("RGB")
        w, h = im.size
        im = im.crop((0, 0, w, int(h * (1 - HOOD_FRAC)))).resize((SHOW_W, int(SHOW_W * h * (1 - HOOD_FRAC) / w)))
        im.save(OUT / r["frame"], quality=90)

    pick.reset_index(names="seg_key")[["frame", "truth", "seg_key", "rd20full", "pci23", "pci23_d", "clip",
                                       "video_t_s", "local", "speed_mph"]].to_csv(KEY, index=False)
    pd.DataFrame({"frame": sorted(pick["frame"]), "my_rating": ""}).to_csv(SHEET, index=False)
    print(f"wrote {len(pick)} frames + {SHEET.name} to {OUT}")
    print("Now: open each frame, write G (good pavement) or B (bad pavement) in my_ratings.csv. Don't open the key.")

def score():
    key = pd.read_csv(KEY)
    me  = pd.read_csv(SHEET, dtype=str).fillna("")
    me["my_rating"] = me["my_rating"].str.strip().str.upper().str[:1]
    missing = me.loc[~me["my_rating"].isin(["G", "B"]), "frame"].tolist()
    if missing:
        sys.exit(f"rate every frame with G or B first - missing: {missing}")
    r = key.merge(me, on="frame")
    r["right"] = r["truth"] == r["my_rating"]
    print(r[["frame", "my_rating", "truth", "pci23", "pci23_d", "rd20full", "right"]].to_string(index=False))
    n = int(r["right"].sum())
    print(f"\nV0: {n}/12 correct (guessing averages 6) -> {'PASS' if n >= 10 else 'FAIL'}  (pre-registered: >= 10/12)")
    r.to_csv(OUT / "v0_result.csv", index=False)
    print(f"saved: {OUT / 'v0_result.csv'}")

if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "make"]()
