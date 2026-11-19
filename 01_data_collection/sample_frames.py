# 01_data_collection/sample_frames.py
# Viability check V2: distance-based frame sampler (QA/QC items M2 and M3).
#
# Turns one drive into still frames, about one every SPACING_M meters of road, each tagged with the
# clean street segment it shows. Three subcommands, run from the project root with the pavement env active:
#
#   python 01_data_collection/sample_frames.py plan    <SESSION>   # seconds; writes two CSV tables, no video work
#   python 01_data_collection/sample_frames.py extract <SESSION>   # minutes; frames -> JPEGs
#          [--method onepass|perframe]  one ffmpeg pass per clip (default) or one short call per frame
#          [--no-hwaccel]               decode in software instead of the Mac's video hardware
#   python 01_data_collection/sample_frames.py timetest <SESSION>  # M2: time per-frame calls vs one pass
#   python 01_data_collection/sample_frames.py spotcheck [--seed N] [--drive SESSION]  # V2 check: 10 random frames
#   python 01_data_collection/sample_frames.py extract <SESSION>__stale   # frames on STALE segments (V7 / H3 set)
#          extract [--prune]  deletes JPEGs in the drive's folder that are no longer in the plan
#
# How a frame gets its segment (reuses Gate 0, no new map matching):
#   Gate 0 already snapped every 1 Hz GPS fix to a segment and kept it only if it passed every rule
#   (within 20 m, heading within 30 deg, >= 15 m from the segment ends, 3-60 mph). A frame between two
#   consecutive fixes inherits their segment only when BOTH fixes passed and snapped to the SAME segment,
#   and that segment is a clean usable segment (data/processed/clean_driven_segments.csv).
#
# Clock (M3): the camera names each clip by its own clock. Per clip, the offset between the first GPS fix
#   and the filename time is split into a whole-hour clock error (0 or -3600 s: the camera missed the
#   daylight-saving change) and a start lag (0-3 s: first fix arrives after recording starts). Anything
#   else stops the script. True video position of a fix = video_t_s - clock error.
#   Clock-sync correction (added after V2 failed 8/10, 2026-11-13): a clip whose start lag differs from the
#   drive's usual lag by more than 1 s was named while the camera clock was still unsynced (seen on the
#   first clip of the Pacific Beach drive: filename 2 s fast, burned-in clock jumps back 2 s mid-clip).
#   Its video positions are shifted by (usual lag - its lag). The expected burned-in clock is computed from
#   GPS time (what the camera shows once synced), not from the filename.
#   --min-lag (added 2026-11-14, Day 4C): `plan <SESSION> --min-lag -1` accepts start lags down to -1 s for that
#   run only (the 11-14 drive: every clip's first fix is 1 s BEFORE its filename time, i.e. the camera clock ran
#   ahead of GPS). The default range 0-3 s is unchanged for every other run; -1 is the lowest value accepted.
#   The burned-in clock cannot reveal a whole-drive camera-clock error (filename and overlay share that clock);
#   V2's N/W check decides whether such a drive's frames are placed correctly.
#   --clock-shift (added 2026-11-14, Day 4C): `plan <SESSION> --clock-shift S` moves every frame of that run S whole
#   seconds later in the video and expects the burned-in clock to read S s ahead (camera clock ahead of GPS for the
#   whole drive). S is measured, never guessed: the 11-14 drive failed V2 4/10 (seed 596); its burned-in N/W sat
#   -1.9 s (median) from the control drives (Golden Hill, SE-SD) -> S = +2; the fresh V2 draw uses seed 616.
#   Default 0 leaves every other run unchanged.
#
# Stale set (added 2026-11-14 for V7/H3): frames on driven segments that are labeled but STALE (repaired on/after
#   2023-02-01, is_stale in segments_clean.csv) are written to a separate table, frames_<SESSION>__stale.csv, and
#   extracted to data/frames/<SESSION>__stale/. Same sampling rules; never used for training. The clean table
#   frames_<SESSION>.csv is unchanged by this addition.
# Reads : data/gps_logs/<SESSION>/_track.csv, data/processed/gate0_fixes_<SESSION>.csv,
#         data/processed/clean_driven_segments.csv, ~/dashcam_footage/<SESSION>/*F.MP4
# Writes: data/processed/clip_offsets_<SESSION>.csv, data/processed/frames_<SESSION>.csv,
#         data/frames/<SESSION>/<clip>_n<frame>.jpg, data/frames/_v2_spotcheck/ (spotcheck only)
# Everything it writes is under data/ (git-ignored: frames show plates and the GPS overlay).

import argparse, math, shutil, subprocess, sys, tempfile, time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SPACING_M   = 7.0     # one frame every ~7 m of road (tunable; recorded in the frames table)
MAX_GAP_S   = 1.5     # GPS is 1 Hz: never interpolate across a longer gap between fixes
LAG_RANGE_S = (0, 3)  # allowed start lag after removing the whole-hour clock error (M3)
CLOCK_ERRORS_S = (0, -3600)   # allowed whole-hour clock errors: none, or camera still on daylight time
JPEG_Q      = 2       # ffmpeg JPEG quality (2 = near-lossless; keeps fine crack detail)
SEED        = 590     # spot-check draw (same seed as V0)

ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
PROC = ROOT / "data/processed"
FOOTAGE = Path.home() / "dashcam_footage"


def stop(msg):
    sys.exit(f"STOP: {msg}")


def camera_start(clip):
    """2026_1111_134009_003201F -> datetime of the filename (camera clock, no time zone)"""
    y, md, hms = clip.split("_")[:3]
    return datetime.strptime(f"{y}{md} {hms}", "%Y%m%d %H%M%S")


def probe(path):
    """frame rate, frame count, size of the video stream (ffprobe)"""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate,nb_frames,width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")
    width, height, rate, nb = out[0], out[1], out[2], out[3]
    num, den = rate.split("/")
    return float(num) / float(den), int(nb), int(width), int(height)


def haversine_m(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(h))


# ---------------------------------------------------------------------------------------------- plan
def plan(session, spacing, min_lag=None, clock_shift=0):
    lag_lo = LAG_RANGE_S[0] if min_lag is None else min_lag
    if not -1 <= lag_lo <= LAG_RANGE_S[0]:
        stop(f"--min-lag must be between -1 and {LAG_RANGE_S[0]} s")
    if clock_shift != int(clock_shift) or not -3 <= clock_shift <= 3:
        stop("--clock-shift must be a whole number of seconds between -3 and 3")
    trk_f, fix_f = ROOT / f"data/gps_logs/{session}/_track.csv", PROC / f"gate0_fixes_{session}.csv"
    cln_f, vid_d = PROC / "clean_driven_segments.csv", FOOTAGE / session
    for f in (trk_f, fix_f, cln_f, vid_d):
        if not f.exists():
            stop(f"missing {f}")

    trk = pd.read_csv(trk_f)
    fx = pd.read_csv(fix_f, index_col=0)
    need = {"clip", "local", "seg_key", "reason"}
    if not need <= set(fx.columns):
        stop(f"{fix_f.name} lacks columns {sorted(need - set(fx.columns))}")
    trk = trk.merge(fx[["clip", "local", "seg_key", "reason"]], on=["clip", "local"], how="left",
                    validate="one_to_one")
    if trk["reason"].isna().any():
        stop(f"{int(trk['reason'].isna().sum())} track fixes have no Gate 0 result; re-run gate0_drive.ipynb")
    trk = trk.sort_values("utc").reset_index(drop=True)

    # ---- M3: per-clip clock offset
    rows = []
    for clip, g in trk.groupby("clip", sort=True):
        first = float(g["video_t_s"].iloc[0])
        err = int(round(first / 3600.0)) * 3600
        lag = round(first - err, 1)
        path = vid_d / f"{clip}.MP4"
        if not path.exists():
            stop(f"video not found: {path}")
        fps, nb, w, h = probe(path)
        rows.append(dict(clip=clip, n_fixes=len(g), first_fix_video_t_s=first, clock_error_s=err,
                         start_lag_s=lag, fps=fps, nb_frames=nb, width=w, height=h,
                         camera_start=camera_start(clip).strftime("%Y-%m-%d %H:%M:%S")))
    off = pd.DataFrame(rows)
    print(f"{session}: {len(off)} clips | per-clip offsets (M3):")
    print(off[["clip", "n_fixes", "first_fix_video_t_s", "clock_error_s", "start_lag_s", "fps",
               "nb_frames"]].to_string(index=False))
    bad = off[~off["clock_error_s"].isin(CLOCK_ERRORS_S) | ~off["start_lag_s"].between(lag_lo, LAG_RANGE_S[1])]
    if len(bad):
        stop(f"unexpected clock offset in {list(bad['clip'])}: expected error in {CLOCK_ERRORS_S} s "
             f"and lag in {(lag_lo, LAG_RANGE_S[1])} s; nothing written")
    if lag_lo < LAG_RANGE_S[0]:
        print(f"NOTE: start lags down to {lag_lo:g} s accepted for this run only (--min-lag; default {LAG_RANGE_S}). "
              f"A whole-drive camera-clock error is invisible to the burned-in clock: V2's N/W check decides.")
    if off["clock_error_s"].nunique() > 1:
        stop("clock error differs between clips of one drive; nothing written")
    usual = float(off["start_lag_s"].mode().max())          # the drive's usual start lag
    off["sync_correction_s"] = np.where((off["start_lag_s"] - usual).abs() > 1, usual - off["start_lag_s"], 0.0)
    fixed = off[off["sync_correction_s"] != 0]
    print(f"usual start lag: {usual:g} s | clips needing a clock-sync correction: "
          + (", ".join(f"{c} ({d:+g} s)" for c, d in zip(fixed["clip"], fixed["sync_correction_s"])) or "none"))
    if clock_shift:                                      # whole-drive camera-clock error measured in V2 (Day 4C)
        off["clock_shift_s"] = float(clock_shift)
        print(f"NOTE: whole-drive clock shift {clock_shift:+g} s for this run (--clock-shift): every frame moves "
              f"{clock_shift:+g} s in the video, and the expected burned-in clock reads {clock_shift:+g} s (camera clock "
              f"ahead of GPS). Measured from V2's burned-in N/W against the control drives.")
    off.to_csv(PROC / f"clip_offsets_{session}.csv", index=False)

    # true video position of each fix
    trk = trk.merge(off[["clip", "clock_error_s", "sync_correction_s", "fps", "nb_frames"]], on="clip")
    trk["video_t"] = trk["video_t_s"] - trk["clock_error_s"] + trk["sync_correction_s"] + clock_shift
    trk["t_utc"] = pd.to_datetime(trk["utc"])

    # ---- distance-based targets between consecutive fixes
    a, b = trk.iloc[:-1].reset_index(drop=True), trk.iloc[1:].reset_index(drop=True)
    step = haversine_m(a["latitude"].to_numpy(), a["longitude"].to_numpy(),
                       b["latitude"].to_numpy(), b["longitude"].to_numpy())
    cum = np.concatenate([[0.0], np.cumsum(step)])          # meters driven at each fix
    dt = (b["t_utc"] - a["t_utc"]).dt.total_seconds().to_numpy()

    out = []
    for i in range(len(a)):
        k0, k1 = math.floor(cum[i] / spacing) + 1, math.floor(cum[i + 1] / spacing)
        if k1 < k0 or step[i] <= 0:
            continue                                         # no target in this interval (or not moving)
        if a.at[i, "clip"] != b.at[i, "clip"]:
            why = "clip_boundary"
        elif dt[i] > MAX_GAP_S:
            why = "gps_gap"
        elif a.at[i, "reason"] != "ok" or b.at[i, "reason"] != "ok":
            why = "fix_not_ok"
        elif a.at[i, "seg_key"] != b.at[i, "seg_key"]:
            why = "segment_change"
        else:
            why = "candidate"
        for k in range(k0, k1 + 1):
            f = (k * spacing - cum[i]) / step[i]             # fraction of the way from fix a to fix b
            vt = a.at[i, "video_t"] + f * (b.at[i, "video_t"] - a.at[i, "video_t"])
            out.append(dict(clip=a.at[i, "clip"], video_t=vt, why=why, seg_key=a.at[i, "seg_key"],
                            dist_m=k * spacing,
                            latitude=a.at[i, "latitude"] + f * (b.at[i, "latitude"] - a.at[i, "latitude"]),
                            longitude=a.at[i, "longitude"] + f * (b.at[i, "longitude"] - a.at[i, "longitude"]),
                            speed_mph=a.at[i, "speed_mph"] + f * (b.at[i, "speed_mph"] - a.at[i, "speed_mph"]),
                            track_deg=a.at[i, "track_deg"] if f < 0.5 else b.at[i, "track_deg"],
                            local=a.at[i, "t_utc"] + timedelta(seconds=f * dt[i]), fps=a.at[i, "fps"],
                            nb_frames=a.at[i, "nb_frames"], clock_error_s=a.at[i, "clock_error_s"],
                            sync_correction_s=a.at[i, "sync_correction_s"]))
    fr = pd.DataFrame(out)

    # keep frames on clean usable segments, inside the video
    cl = pd.read_csv(cln_f)
    cl["seg_key"] = cl["seg_key"].astype("string").str.strip().str.upper()
    cl = cl.drop_duplicates("seg_key").set_index("seg_key")
    fr["seg_key"] = fr["seg_key"].astype("string").str.strip().str.upper()
    fr["n"] = np.rint(fr["video_t"] * fr["fps"]).astype(int)           # frame number inside the clip
    is_cand = fr["why"] == "candidate"
    # stale set (V7/H3): candidates on labeled, stale segments; decided before the clean relabelling below
    stale_keys = set()
    sc_f = PROC / "segments_clean.csv"
    if sc_f.exists():
        sc = pd.read_csv(sc_f, usecols=["iamfloc", "is_stale", "pci23", "pci23_d", "rd20full", "cpname",
                                        "is_residential", "is_alley"])
        sc["iamfloc"] = sc["iamfloc"].astype("string").str.strip().str.upper()
        sc = sc.drop_duplicates("iamfloc").set_index("iamfloc")
        stale_keys = set(sc.index[(sc["is_stale"] == 1) & sc["pci23"].notna()])
    else:
        print(f"note: {sc_f.name} not found -> no stale-set table")
    is_stale_fr = is_cand & fr["seg_key"].isin(stale_keys) & (fr["n"] >= 0) & (fr["n"] < fr["nb_frames"])
    fr.loc[is_cand & ~fr["seg_key"].isin(cl.index), "why"] = "not_clean"
    fr.loc[(fr["why"] == "candidate") & ((fr["n"] < 0) | (fr["n"] >= fr["nb_frames"])), "why"] = "outside_video"
    fr.loc[fr["why"] == "candidate", "why"] = "ok"
    print(f"\nframe positions every {spacing:g} m: {len(fr):,} | reasons:")
    print(fr["why"].value_counts().to_string())

    keep = fr[fr["why"] == "ok"].drop_duplicates(["clip", "n"]).copy()
    keep = keep.join(cl[[c for c in ["rd20full", "pci23", "pci23_d", "cpname", "is_primary"]
                         if c in cl.columns]], on="seg_key")
    loc = keep["local"].dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
    # expected burned-in camera clock (synced) = true local time + the camera's whole-hour error
    keep["cam_clock"] = (loc - pd.to_timedelta(keep["clock_error_s"] - clock_shift, unit="s")).dt.floor("s").dt.strftime("%H:%M:%S")
    keep["local"] = loc.dt.round("100ms").dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-5]   # true local time, 0.1 s
    keep["drive"] = session
    keep["spacing_m"] = spacing
    keep["file"] = [f"data/frames/{session}/{c}_n{n:05d}.jpg" for c, n in zip(keep["clip"], keep["n"])]
    keep["video_t"] = keep["video_t"].round(3)
    keep["latitude"], keep["longitude"] = keep["latitude"].round(6), keep["longitude"].round(6)
    keep["speed_mph"] = keep["speed_mph"].round(1)
    cols = ["drive", "clip", "n", "video_t", "sync_correction_s", "cam_clock", "local", "latitude", "longitude", "speed_mph",
            "track_deg", "seg_key", "rd20full", "pci23", "pci23_d", "cpname", "is_primary", "dist_m",
            "spacing_m", "file"]
    keep = keep[[c for c in cols if c in keep.columns]].sort_values(["clip", "n"]).reset_index(drop=True)
    keep.to_csv(PROC / f"frames_{session}.csv", index_label="frame_id")

    if stale_keys:
        st = fr[is_stale_fr].drop_duplicates(["clip", "n"]).copy()
        st = st[~st["seg_key"].isin(cl.index)]                       # never overlaps the clean table
        if len(st):
            meta = sc.loc[st["seg_key"].astype(str)]
            for c in ("rd20full", "pci23", "pci23_d", "cpname"):
                st[c] = meta[c].to_numpy()
            # for the stale table, is_primary means "would be primary if it were not stale": residential, non-alley
            st["is_primary"] = ((meta["is_residential"] == 1) & (meta["is_alley"] == 0)).astype(int).to_numpy()
            sloc = st["local"].dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
            st["cam_clock"] = (sloc - pd.to_timedelta(st["clock_error_s"] - clock_shift, unit="s")).dt.floor("s").dt.strftime("%H:%M:%S")
            st["local"] = sloc.dt.round("100ms").dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-5]
            st["drive"] = f"{session}__stale"
            st["spacing_m"] = spacing
            st["file"] = [f"data/frames/{session}__stale/{c}_n{n:05d}.jpg" for c, n in zip(st["clip"], st["n"])]
            st["video_t"] = st["video_t"].round(3)
            st["latitude"], st["longitude"] = st["latitude"].round(6), st["longitude"].round(6)
            st["speed_mph"] = st["speed_mph"].round(1)
            st = st[[c for c in cols if c in st.columns]].sort_values(["clip", "n"]).reset_index(drop=True)
            st.to_csv(PROC / f"frames_{session}__stale.csv", index_label="frame_id")
            sp = st.groupby("seg_key").size()
            print(f"\nSTALE SET (V7/H3, never trained on): {len(st):,} frames on {sp.size} stale segments "
                  f"({int(st.drop_duplicates('seg_key')['is_primary'].sum())} residential non-alley) "
                  f"-> frames_{session}__stale.csv | extract with: extract {session}__stale")
        else:
            print("\nSTALE SET: no frames on stale segments in this drive")

    per = keep.groupby("seg_key").size()
    n_clean = int(cl["drive"].eq(session).sum()) if "drive" in cl.columns else None
    print(f"\nKEPT: {len(keep):,} frames on {per.size} clean segments"
          + (f" (of {n_clean} clean usable segments credited to this drive in Gate 0)" if n_clean else ""))
    if "is_primary" in keep.columns:
        pk = keep[keep["is_primary"] == 1]
        print(f"  primary: {len(pk):,} frames on {pk['seg_key'].nunique()} segments")
    print(f"  frames per segment: min {per.min()}, median {per.median():.0f}, max {per.max()} "
          f"| segments with < 3 frames: {int((per < 3).sum())}")
    print(f"  disk estimate: ~{len(keep) * 2.0 / 1024:.1f} GB at ~2 MB per 4K JPEG (estimate)")
    print(f"saved: clip_offsets_{session}.csv, frames_{session}.csv")


# ------------------------------------------------------------------------------------------- extract
def run_onepass(clip_path, ns, out_dir, hw):
    """one decoding pass over the clip, writing only frames whose number is in ns (sorted)"""
    expr = "+".join(f"eq(n,{n})" for n in ns)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    cmd += ["-hwaccel", "videotoolbox"] if hw else []
    cmd += ["-i", str(clip_path), "-vf", f"select='{expr}'", "-fps_mode", "passthrough",
            "-q:v", str(JPEG_Q), str(out_dir / "tmp_%06d.jpg")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    got = sorted(out_dir.glob("tmp_*.jpg"))
    ok = res.returncode == 0 and len(got) == len(ns)
    if ok:
        for src, n in zip(got, ns):                     # outputs arrive in frame order
            src.rename(out_dir / f"f_{n:05d}.jpg")
    return ok, res.stderr.strip()


def run_perframe(clip_path, ns, fps, out_dir, hw):
    """one short ffmpeg call per frame: seek to just before frame n, write the first frame at/after it"""
    for n in ns:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
        cmd += ["-hwaccel", "videotoolbox"] if hw else []
        # target a quarter frame early: frame n-1 falls before it, frame n is the first one kept
        cmd += ["-ss", f"{(n - 0.25) / fps:.6f}", "-i", str(clip_path), "-frames:v", "1",
                "-q:v", str(JPEG_Q), str(out_dir / f"f_{n:05d}.jpg")]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not (out_dir / f"f_{n:05d}.jpg").exists():
            return False, res.stderr.strip()
    return True, ""


def extract_clip(session, clip, ns, fps, out_dir, method, hw):
    """extract one clip into a temp folder, then move the frames into place; returns (ok, hw, seconds, err)"""
    clip_path = FOOTAGE / session.split("__")[0] / f"{clip}.MP4"     # <SESSION>__stale reads the same footage
    run = (lambda d, h: run_onepass(clip_path, ns, d, h)) if method == "onepass" else \
          (lambda d, h: run_perframe(clip_path, ns, fps, d, h))
    tmp = Path(tempfile.mkdtemp(dir=out_dir, prefix=".tmp_"))
    t0 = time.time()
    ok, err = run(tmp, hw)
    if not ok and hw:                                   # retry in software; keep hardware off only if that works
        shutil.rmtree(tmp); tmp = Path(tempfile.mkdtemp(dir=out_dir, prefix=".tmp_"))
        ok_sw, err_sw = run(tmp, False)
        if ok_sw:
            print(f"  {clip}: hardware decoding failed, software worked; software for the rest of this run\n"
                  f"    ...{err[-200:]}")
            ok, err, hw = True, "", False
        else:
            err = err_sw
    secs = time.time() - t0
    if ok:
        for n in ns:
            (tmp / f"f_{n:05d}.jpg").rename(out_dir / f"{clip}_n{n:05d}.jpg")
    shutil.rmtree(tmp)
    return ok, hw, secs, err


def extract(session, method, hwaccel, prune=False):
    fr_f = PROC / f"frames_{session}.csv"
    if not fr_f.exists():
        stop(f"missing {fr_f.name}: run the plan step first")
    fr = pd.read_csv(fr_f)
    fps_of = pd.read_csv(PROC / f"clip_offsets_{session.split('__')[0]}.csv").set_index("clip")["fps"]
    out_dir = ROOT / f"data/frames/{session}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = PROC / f"extract_log_{session}.csv"         # extraction time is reported separately from V4 (M2)
    t_all, n_written, hw = time.time(), 0, hwaccel

    for clip, g in fr.groupby("clip", sort=True):
        if all((ROOT / f).exists() for f in g["file"]):
            print(f"  {clip}: {len(g)} frames already extracted, skipped")
            continue
        ns = sorted(g["n"].tolist())
        ok, hw, secs, err = extract_clip(session, clip, ns, float(fps_of[clip]), out_dir, method, hw)
        if not ok:
            stop(f"{clip}: extraction failed ({method}); frames for this clip not written.\n  ffmpeg said: ...{err[-400:]}")
        n_written += len(ns)
        print(f"  {clip}: {len(ns)} frames in {secs:.1f} s ({secs / len(ns):.2f} s/frame, {method}, "
              f"{'hardware' if hw else 'software'} decoding)")
        pd.DataFrame([dict(clip=clip, method=method, hardware=hw, n_frames=len(ns), seconds=round(secs, 1))]).to_csv(
            log_f, mode="a", header=not log_f.exists(), index=False)

    print(f"\nwrote {n_written:,} frames in {time.time() - t_all:.0f} s -> {out_dir.relative_to(ROOT)}")
    missing = [f for f in fr["file"] if not (ROOT / f).exists()]
    print(f"check: {len(fr) - len(missing):,} of {len(fr):,} planned frames are on disk"
          + ("" if not missing else f"  <-- {len(missing)} MISSING"))
    planned = {Path(f).name for f in fr["file"]}
    extra = [p for p in out_dir.glob("*.jpg") if p.name not in planned]
    if extra and prune:
        for p in extra:
            p.unlink()
        print(f"pruned: deleted {len(extra)} old frames that are no longer in the plan")
    elif extra:
        print(f"note: {len(extra)} JPEGs in this folder are not in the plan (re-run with --prune to delete them)")


def timetest(session):
    """M2: re-extract 10 already-extracted frames one by one (hardware and software) and compare"""
    from PIL import Image
    fr = pd.read_csv(PROC / f"frames_{session}.csv")
    fps_of = pd.read_csv(PROC / f"clip_offsets_{session}.csv").set_index("clip")["fps"]
    done = [c for c, g in fr.groupby("clip", sort=True) if all((ROOT / f).exists() for f in g["file"])]
    if not done:
        stop("no fully extracted clip to compare against")
    clip = done[0]
    ns = sorted(fr.loc[fr["clip"] == clip, "n"].tolist())
    sample = ns[:: max(1, len(ns) // 10)][:10]
    out_dir = ROOT / f"data/frames/{session}"
    print(f"TIME TEST (M2) on {clip}: {len(sample)} frames, one ffmpeg call per frame")
    for hw in (True, False):
        tmp = Path(tempfile.mkdtemp(dir=out_dir, prefix=".tmp_"))
        t0 = time.time()
        ok, err = run_perframe(FOOTAGE / session / f"{clip}.MP4", sample, float(fps_of[clip]), tmp, hw)
        secs = time.time() - t0
        if not ok:
            print(f"  {'hardware' if hw else 'software'}: FAILED ...{err[-200:]}")
            shutil.rmtree(tmp)
            continue
        d = [np.abs(np.asarray(Image.open(tmp / f"f_{n:05d}.jpg"), dtype=np.int16)
                    - np.asarray(Image.open(out_dir / f"{clip}_n{n:05d}.jpg"), dtype=np.int16)).mean()
             for n in sample]
        shutil.rmtree(tmp)
        print(f"  {'hardware' if hw else 'software'} decoding: {secs / len(sample):.2f} s/frame | "
              f"difference from the one-pass frame (0-255, near 0 = same frame): {', '.join(f'{x:.1f}' for x in d)}")


# ----------------------------------------------------------------------------------------- spotcheck
def label(im, text):
    """small black box with the frame number, top-left (works on old and new Pillow)"""
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 80, 44), fill="black")
    try:
        d.text((10, 8), text, fill="yellow", font_size=28)
    except TypeError:                                   # Pillow < 10.1 has no font_size
        d.text((10, 8), text, fill="yellow")


def spotcheck(seed, drive=None):
    from PIL import Image
    tables = [t for t in sorted(PROC.glob("frames_*.csv")) if "__stale" not in t.stem]
    if drive:
        tables = [t for t in tables if t.stem == f"frames_{drive}"]
    if not tables:
        stop("no frames_<SESSION>.csv found: run plan + extract first")
    fr = pd.concat([pd.read_csv(f) for f in tables], ignore_index=True)
    fr = fr[[(ROOT / f).exists() for f in fr["file"]]]
    if len(fr) < 10:
        stop(f"only {len(fr)} extracted frames on disk")
    pick = fr.sample(10, random_state=seed).reset_index(drop=True)
    pick.index = [str(i) for i in range(1, 11)]
    if "sync_correction_s" in fr.columns:                # targeted extras: frames from clock-corrected clips
        corr = fr[(fr["sync_correction_s"].fillna(0) != 0) & ~fr["file"].isin(pick["file"])]
        if len(corr):
            t = corr.sample(min(3, len(corr)), random_state=seed).reset_index(drop=True)
            t.index = [f"T{i}" for i in range(1, len(t) + 1)]
            pick = pd.concat([pick, t])
    out = ROOT / ("data/frames/_v2_spotcheck" if seed == SEED else f"data/frames/_v2_spotcheck_seed{seed}")
    if drive:
        out = ROOT / f"data/frames/_v2_spotcheck_{drive}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    strips, thumbs = [], []
    for i, r in pick.iterrows():
        im = Image.open(ROOT / r["file"]).convert("RGB")
        w, h = im.size
        s = im.crop((0, int(h * 0.88), w, h)).resize((1920, int(1920 * h * 0.12 / w)))
        label(s, f"#{i}")
        strips.append(s)
        t = im.resize((960, int(960 * h / w)))
        label(t, f"#{i}")
        thumbs.append(t)

    sheet = Image.new("RGB", (1920, sum(s.height for s in strips)))
    y = 0
    for s in strips:
        sheet.paste(s, (0, y)); y += s.height
    sheet.save(out / "strips.jpg", quality=90)
    th = thumbs[0].height
    grid = Image.new("RGB", (1920, th * math.ceil(len(thumbs) / 2)))
    for k, t in enumerate(thumbs):
        grid.paste(t, ((k % 2) * 960, (k // 2) * th))
    grid.save(out / "frames.jpg", quality=85)

    pick["exp_N"] = pick["latitude"].round(4)
    pick["exp_W"] = (-pick["longitude"]).round(4)
    show = [c for c in ["drive", "clip", "n", "sync_correction_s", "cam_clock", "exp_N", "exp_W", "rd20full"]
            if c in pick.columns]
    pick[show + ["file"]].to_csv(out / "spotcheck.csv", index_label="num")
    print(f"V2 SPOT CHECK (seed {seed}): compare each strip's burned-in clock and N/W with the expected values.")
    print("PASS rule (pre-registered 2026-11-13): for the 10 random frames, clock within +-1 s and N/W within")
    print("0.0002, and the full frame shows a street (not a driveway, lot, or intersection box).")
    print("T-frames (targeted, from clock-corrected clips) are reported separately; their clock may also read")
    print("+correction s (camera not yet synced early in the clip), so N/W within 0.0002 decides them.\n")
    print(pick[show].to_string())
    print(f"\nsaved: {out.relative_to(ROOT)}/strips.jpg, frames.jpg, spotcheck.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="V2 distance-based frame sampler")
    ap.add_argument("step", choices=["plan", "extract", "timetest", "spotcheck"])
    ap.add_argument("session", nargs="?")
    ap.add_argument("--spacing", type=float, default=SPACING_M)
    ap.add_argument("--method", choices=["onepass", "perframe"], default="onepass")
    ap.add_argument("--no-hwaccel", action="store_true")
    ap.add_argument("--prune", action="store_true", help="extract: delete JPEGs no longer in the plan")
    ap.add_argument("--seed", type=int, default=SEED, help="spotcheck: random draw (590 = original V2)")
    ap.add_argument("--drive", help="spotcheck: draw only from this SESSION (new drives, Day 4C on)")
    ap.add_argument("--min-lag", type=float, default=None,
                    help="plan only: accept start lags down to this value (-1 to 0 s; default 0) for this run")
    ap.add_argument("--clock-shift", type=float, default=0,
                    help="plan only: whole-drive camera-clock error in whole seconds (-3..3), measured in V2")
    args = ap.parse_args()
    if args.step != "spotcheck" and not args.session:
        ap.error("plan, extract and timetest need a SESSION (the footage folder name)")
    if (args.min_lag is not None or args.clock_shift) and args.step != "plan":
        ap.error("--min-lag and --clock-shift are for the plan step only")
    if args.step == "plan":
        plan(args.session, args.spacing, args.min_lag, args.clock_shift)
    elif args.step == "extract":
        extract(args.session, args.method, not args.no_hwaccel, args.prune)
    elif args.step == "timetest":
        timetest(args.session)
    else:
        spotcheck(args.seed, args.drive)
