# 04_pipeline_design/v0_check_misses.py
# V0 follow-up (diagnosis only; the V0 result itself stays 9/12 = FAIL).
# For each missed frame, plus one correctly rated frame from the same street when there is one:
#   1) re-extract the frame UNCROPPED, so the camera's burned-in GPS text is visible, and print the
#      GPS fix we used -> tells us whether the frame really shows the intended street (timing check)
#   2) print the segment's repair history -> a very low score on a street that looks fine may mean
#      a repave the City's repair log never recorded
# Writes only data/v0_blind/check_<frame>.jpg. Usage: python 04_pipeline_design/v0_check_misses.py
import subprocess
from pathlib import Path
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
SESSION = "10-25-2026-Mira-Mesa"
FOOTAGE = Path("/Users/waelsultan/dashcam_footage") / SESSION
B       = ROOT / "data/v0_blind"

res = pd.read_csv(B / "v0_result.csv")
key = pd.read_csv(B / "_answer_key_DO_NOT_OPEN.csv")[["frame", "clip", "video_t_s", "local"]]
res = res.merge(key, on="frame", suffixes=("", "_k"))
miss = res[~res["right"]]
same_street_ok = res[res["right"] & res["rd20full"].isin(miss["rd20full"])].drop_duplicates("rd20full")
look = pd.concat([miss, same_street_ok])

trk = pd.read_csv(ROOT / f"data/gps_logs/{SESSION}/_track.csv")[["clip", "local", "latitude", "longitude", "speed_mph"]]
look = look.merge(trk, on=["clip", "local"], how="left", suffixes=("", "_trk"))
cl = pd.read_csv(ROOT / "data/processed/segments_clean.csv")
cl["seg_key"] = cl["iamfloc"].astype("string").str.strip().str.upper()
cols = [c for c in ["has_prior_repair", "years_since_repair", "last_repair_pre_survey", "last_repair_end",
                    "pwidth", "lane_no"] if c in cl.columns]
look = look.merge(cl[["seg_key"] + cols], on="seg_key", how="left")

pd.set_option("display.width", 200)
show = ["frame", "right", "my_rating", "truth", "pci23", "rd20full", "latitude", "longitude", "speed_mph"] + cols
print(look[show].to_string(index=False))

for _, r in look.iterrows():
    clip = next((FOOTAGE / f"{r['clip']}{e}" for e in (".MP4", ".mp4") if (FOOTAGE / f"{r['clip']}{e}").exists()), None)
    out = B / f"check_{r['frame']}"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{max(r['video_t_s'], 0):.2f}",
                    "-i", str(clip), "-frames:v", "1", "-vf", "scale=1920:-2", "-q:v", "3", str(out)], check=True)
print("\nsaved uncropped frames:", ", ".join(f"check_{f}" for f in look["frame"]))
print("Compare each frame's burned-in N:/W: text (bottom-left) with the latitude/longitude printed above.")
