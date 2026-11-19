# 01_data_collection/extract_gps.py
# Pull the VIOFO's embedded 1 Hz GPS stream out of every FRONT-camera clip of one drive session.
#
# Usage (from the project root, with the pavement env active):
#   python 01_data_collection/extract_gps.py ~/dashcam_footage/10-25-2026-Mira-Mesa
#
# Writes, under data/gps_logs/<session-folder-name>/:
#   <clipstem>_gps.csv   one per clip
#   _track.csv           every clip combined, sorted, de-duplicated  <-- Gate 0 reads this one
# Columns:
#   clip        clip file stem (e.g. 2026_1025_123017_003179F)
#   video_t_s   seconds from the START OF THE VIDEO (filename timestamp) — frames join on this
#   t_s         seconds since this clip's first GPS fix (reference)
#   utc / local timestamps (local = America/Los_Angeles, daylight saving handled automatically)
#   latitude, longitude   signed decimal degrees
#   speed_mph   ground speed
#   track_deg   heading, 0 = north, 90 = east
#
# Why video_t_s: the VIOFO names a file when recording starts, but the first GPS fix lands a second
# or two later. Anchoring time to the filename removes that shift (~70 ft at 25 mph).
#
# Changes vs. the Project-4 version: session folder is an argument (no hard-coded paths); outputs go
# inside the pavement_index project; the UTC offset comes from the date itself (DST ended Nov 1, 2026,
# so a fixed -7 would silently break later drives); a combined track file; distance driven per clip.

import subprocess, csv, sys, math
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

if len(sys.argv) != 2:
    sys.exit("usage: python 01_data_collection/extract_gps.py <session footage folder>")
SESSION_DIR = Path(sys.argv[1]).expanduser()
ROOT        = Path(__file__).resolve().parent.parent          # pavement_index/
GPS_DIR     = ROOT / "data/gps_logs" / SESSION_DIR.name
LOCAL_TZ    = ZoneInfo("America/Los_Angeles")
SPEED_IS_KMH = True        # VERIFIED 2026-10-25: this VIOFO writes no GPSSpeedRef; speed from positions
                           # (12.0 m/s = 43.3 km/h) matched the reported 43.3 -> km/h. None = trust the tag.

GPS_DIR.mkdir(parents=True, exist_ok=True)
SEP = "|"
FMT = SEP.join(f"${t}" for t in
               ["GPSDateTime", "GPSLatitude", "GPSLatitudeRef",
                "GPSLongitude", "GPSLongitudeRef", "GPSSpeed", "GPSSpeedRef", "GPSTrack"])

def parse_utc(s):
    s = s.strip().rstrip("Z")
    for f in ("%Y:%m:%d %H:%M:%S.%f", "%Y:%m:%d %H:%M:%S"):
        try: return datetime.strptime(s, f).replace(tzinfo=timezone.utc)
        except ValueError: pass
    return None

def video_start(stem):
    # 2026_1025_123017_003179F -> 2026-10-25 12:30:17, camera clock = local time
    try:
        y, md, hms = stem.split("_")[:3]
        return datetime.strptime(f"{y}{md} {hms}", "%Y%m%d %H%M%S").replace(tzinfo=LOCAL_TZ)
    except Exception:
        return None

def signed(val, ref, neg_letter):
    v = float(val)                       # -n usually gives signed decimals already
    if ref and ref.strip().upper().startswith(neg_letter) and v > 0:
        v = -v
    return v

def miles(a, b):                         # great-circle distance between two fixes
    la1, lo1, la2, lo2 = map(math.radians, (a["latitude"], a["longitude"], b["latitude"], b["longitude"]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 3958.8 * math.asin(math.sqrt(h))

clips = sorted(p for p in SESSION_DIR.iterdir()
               if p.suffix.upper() == ".MP4" and p.stem.upper().endswith("F"))
if not clips:
    sys.exit(f"No front-cam (*F.MP4) clips found in {SESSION_DIR}")
print(f"{len(clips)} front-cam clips in {SESSION_DIR.name}\n")

all_rows, total_mi = [], 0.0
for clip in clips:
    vstart = video_start(clip.stem)
    out = subprocess.run(["exiftool", "-ee", "-n", "-q", "-f", "-p", FMT, str(clip)],
                         capture_output=True, text=True).stdout
    rows, t0 = [], None
    for line in out.splitlines():
        p = line.split(SEP)
        if len(p) != 8:
            continue
        dt_s, lat_s, lat_r, lon_s, lon_r, spd_s, spd_r, trk_s = [x.strip() for x in p]
        dt = parse_utc(dt_s)
        if dt is None or "-" in (lat_s, lon_s) or not lat_s or not lon_s:
            continue                                   # dropped fix: skip, timestamps keep us aligned
        try:
            lat, lon = signed(lat_s, lat_r, "S"), signed(lon_s, lon_r, "W")
        except ValueError:
            continue
        if lat == 0 and lon == 0:
            continue                                   # no-fix placeholder
        t0 = t0 or dt
        local_dt = dt.astimezone(LOCAL_TZ)
        vt = (dt - vstart).total_seconds() if vstart else (dt - t0).total_seconds()
        try:    spd = float(spd_s)
        except ValueError: spd = None
        kmh = SPEED_IS_KMH if SPEED_IS_KMH is not None else not spd_r.upper().startswith("M")
        mph = round(spd * (0.621371 if kmh else 2.23694), 1) if spd is not None else ""
        try:    trk = round(float(trk_s), 1)
        except ValueError: trk = ""
        rows.append(dict(clip=clip.stem, video_t_s=round(vt, 1), t_s=round((dt - t0).total_seconds(), 1),
                         utc=dt.strftime("%Y-%m-%d %H:%M:%S"), local=local_dt.strftime("%Y-%m-%d %H:%M:%S"),
                         latitude=round(lat, 6), longitude=round(lon, 6), speed_mph=mph, track_deg=trk))

    if not rows:
        print(f"  {clip.stem}  NO GPS FIXES  <-- investigate")
        continue
    with open(GPS_DIR / f"{clip.stem}_gps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    mi   = sum(miles(a, b) for a, b in zip(rows, rows[1:]))
    lag  = rows[0]["video_t_s"]
    gaps = [round(b["t_s"] - a["t_s"], 1) for a, b in zip(rows, rows[1:]) if b["t_s"] - a["t_s"] > 2.0]
    spds = [r["speed_mph"] for r in rows if r["speed_mph"] != ""]
    flag = "  <-- ODD LAG (camera clock?)" if not (0 <= lag <= 5) else ""
    print(f"  {clip.stem}  {len(rows):4d} fixes / {rows[-1]['t_s']:4.0f}s  {mi:4.2f} mi  "
          f"first fix {lag:+.0f}s  {min(spds):.0f}-{max(spds):.0f} mph"
          + (f"  GAPS: {gaps}" if gaps else "") + flag)
    total_mi += mi
    all_rows += rows

# combined track: sort by time, drop fixes repeated at clip boundaries
all_rows.sort(key=lambda r: r["utc"])
seen, track = set(), []
for r in all_rows:
    if r["utc"] not in seen:
        seen.add(r["utc"]); track.append(r)
with open(GPS_DIR / "_track.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(track[0])); w.writeheader(); w.writerows(track)

print(f"\n{len(track):,} unique fixes | {track[0]['local']} -> {track[-1]['local']} | ~{total_mi:.1f} miles")
print(f"Wrote per-clip CSVs + _track.csv to {GPS_DIR}")
