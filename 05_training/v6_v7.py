# 05_training/v6_v7.py
# Viability checks V6 (signal) and V7 (stale-set sanity), exactly as frozen in pipeline_rules.json.
#
# V6: per primary segment, pool its kept frame vectors (mean and std, concatenated) -> ridge regression,
#     compared with the mean predictor, cross-validated with all segments of a street kept together
#     (outer GroupKFold by rd20full; ridge alpha chosen in an inner GroupKFold on the training fold only).
#     PASS if  MAE(ridge) <= 0.90 x MAE(mean predictor)  AND  Spearman(ridge predictions, PCI) >= 0.30.
#     Shown alongside for context (not part of the V6 rule): rung 3 hand-crafted image features and
#     rung 2 metadata (planning area, ZIP, width, length, lanes, pavement age), same folds.
# V7: ridge fitted on ALL primary segments (alpha by grouped CV) predicts the STALE set (segments repaved on/after
#     2023-02-01, residential, non-alley). PASS if the mean prediction > the mean of their 2023 labels.
#
# Refuses to run on real data until the frozen gate is met (>= 150 primary segments across >= 2 planning
# areas with >= 20 each; decided 2026-11-14: no preliminary V6).
# Usage (project root, pavement env active):
#   python 05_training/v6_v7.py --selftest     # synthetic data: must PASS with signal and FAIL without
#   python 05_training/v6_v7.py                # real data, only when the gate is met
# Writes: data/modeling/v6_results.csv, v6_oof_predictions.csv, v7_stale_predictions.csv

import argparse, json, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / ".git").exists())
ALPHAS = np.logspace(-2, 5, 15)
OUTER, INNER = 5, 5
BANDS = ["Good", "Satisfactory", "Fair", "Poor", "Very Poor", "Serious", "Failed"]
NUM_META = ["pwidth", "pav_length", "lane_no", "has_prior_repair", "years_since_repair"]
CAT_META = ["cpname", "zip"]
AGE_CAP = 30.0                     # no repair record -> years_since_repair capped at 30 for ridge (plan section 7)


def stop(msg):
    sys.exit(f"STOP: {msg}")


# ----------------------------------------------------------------------------------------- features
def pooled(frames, vec, keys):
    """segment x (2 * D): mean and std of each segment's frame vectors"""
    rows = frames.groupby("seg_key").indices
    return np.stack([np.concatenate([vec[rows[k]].mean(0), vec[rows[k]].std(0)]) for k in keys])


def handcrafted(frames, keys):
    """segment x 6: mean and std of the frame brightness mean / brightness std / log Laplacian variance"""
    f = frames.assign(log_lap=np.log1p(frames["lap_var"]))
    g = f.groupby("seg_key")[["gray_mean", "gray_std", "log_lap"]]
    return pd.concat([g.mean(), g.std(ddof=0).add_suffix("_sd")], axis=1).loc[keys].to_numpy()


def vectors(npz, kind):
    return np.concatenate([npz["cls"], npz["patch_mean"]], 1) if kind == "concat_cls_patchmean" else npz[kind]


# ------------------------------------------------------------------------------------------- models
def ridge_pipe(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def meta_pipe(alpha):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM_META),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_META)])
    return make_pipeline(pre, Ridge(alpha=alpha))


def pick_alpha(make, X, y, groups):
    """inner grouped CV on the training data only -> alpha with the lowest MAE"""
    k = min(INNER, len(np.unique(groups)))
    best, best_mae = None, np.inf
    for a in ALPHAS:
        err = []
        for tr, va in GroupKFold(n_splits=k).split(X, y, groups):
            m = make(a).fit(sub(X, tr), y[tr])
            err.append(np.abs(np.clip(m.predict(sub(X, va)), 0, 100) - y[va]))
        mae = np.concatenate(err).mean()
        if mae < best_mae:
            best, best_mae = a, mae
    return best


def sub(X, idx):
    return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]


def grouped_cv(models, y, groups):
    """outer GroupKFold; returns out-of-fold predictions per model and the chosen alphas"""
    oof = {name: np.full(len(y), np.nan) for name in ["mean predictor"] + list(models)}
    alphas = {name: [] for name in models}
    for tr, te in GroupKFold(n_splits=OUTER).split(np.zeros(len(y)), y, groups):
        oof["mean predictor"][te] = y[tr].mean()
        for name, (make, X) in models.items():
            a = pick_alpha(make, sub(X, tr), y[tr], groups[tr])
            alphas[name].append(a)
            oof[name][te] = np.clip(make(a).fit(sub(X, tr), y[tr]).predict(sub(X, te)), 0, 100)
    return oof, alphas


def metrics(y, p, bands):
    rho = spearmanr(y, p).correlation if np.std(p) > 0 else 0.0
    bad = np.isin(bands, ["Poor", "Very Poor", "Serious", "Failed"])
    return dict(MAE=np.abs(p - y).mean(), RMSE=np.sqrt(((p - y) ** 2).mean()), Spearman=rho,
                MAE_Poor_to_Failed=np.abs(p - y)[bad].mean() if bad.any() else np.nan)


# --------------------------------------------------------------------------------------------- core
def run(seg, frames, vec, meta, stale_seg, stale_frames, stale_vec, out_dir, kind):
    keys = seg["seg_key"].tolist()
    y = seg["pci23"].to_numpy(float)
    groups = seg["rd20full"].to_numpy()
    bands = seg["pci23_d"].to_numpy()
    Xd = pooled(frames, vec, keys)
    Xh = handcrafted(frames, keys)
    Xm = meta.set_index("iamfloc").loc[keys, NUM_META + CAT_META].reset_index(drop=True)
    Xm["years_since_repair"] = Xm["years_since_repair"].fillna(AGE_CAP).clip(upper=AGE_CAP)
    Xm[CAT_META] = Xm[CAT_META].astype(str)
    print(f"V6 data: {len(y)} primary segments, {len(np.unique(groups))} streets, "
          f"{seg['cpname'].nunique()} planning areas | image features {Xd.shape[1]} ({kind}, mean + std)")
    models = {"ridge DINOv2 (V6)": (ridge_pipe, Xd), "ridge hand-crafted (rung 3, context)": (ridge_pipe, Xh),
              "ridge metadata (rung 2, context)": (meta_pipe, Xm)}
    oof, alphas = grouped_cv(models, y, groups)
    res = pd.DataFrame({n: metrics(y, p, bands) for n, p in oof.items()}).T
    res["alphas (outer folds)"] = ["-"] + [", ".join(f"{a:g}" for a in alphas[n]) for n in models]
    print("\nOUT-OF-FOLD RESULTS (5 folds, streets kept together):")
    print(res.round(3).to_string())
    m0, m1 = res.loc["mean predictor", "MAE"], res.loc["ridge DINOv2 (V6)", "MAE"]
    rho = res.loc["ridge DINOv2 (V6)", "Spearman"]
    v6 = (m1 <= 0.90 * m0) and (rho >= 0.30)
    print(f"\nV6: MAE {m1:.2f} vs mean predictor {m0:.2f} ({(1 - m1 / m0):+.1%}; need >= +10.0%) | Spearman {rho:.2f} "
          f"(need >= 0.30) -> {'PASS' if v6 else 'FAIL'}")
    per = pd.DataFrame({"band": bands, "y": y, **{n: p for n, p in oof.items()}})
    tab = per.groupby("band").apply(lambda g: pd.Series({"n": len(g), **{n: np.abs(g[n] - g["y"]).mean() for n in oof}}),
                                    include_groups=False)
    tab["n"] = tab["n"].astype(int)
    print("\nMAE by band (PCI points):")
    print(tab.reindex([b for b in BANDS if b in tab.index]).round(1).to_string())

    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "v6_results.csv")
    per.insert(0, "seg_key", keys)
    per.to_csv(out_dir / "v6_oof_predictions.csv", index=False)

    # V7
    v7 = None
    if stale_seg is not None and len(stale_seg):
        sk = stale_seg["seg_key"].tolist()
        a = pick_alpha(ridge_pipe, Xd, y, groups)
        model = ridge_pipe(a).fit(Xd, y)
        pred = np.clip(model.predict(pooled(stale_frames, stale_vec, sk)), 0, 100)
        lab = stale_seg["pci23"].to_numpy(float)
        v7 = pred.mean() > lab.mean()
        after = stale_seg["repaired_after_survey_end"].astype(bool).to_numpy()
        print(f"\nV7: {len(sk)} stale segments | mean prediction {pred.mean():.1f} vs mean 2023 label {lab.mean():.1f} "
              f"-> {'PASS' if v7 else 'FAIL'} (pre-registered: prediction > label)")
        base = y.mean()                  # what a model with NO signal (the mean predictor) would predict for every segment
        print(f"    GUARD (context, added 2026-11-14 after the selftest): the mean predictor would predict {base:.1f}"
              f"{' -> it would ALSO pass, so V7 alone is not evidence of signal' if base > lab.mean() else ''}. "
              f"Model above the mean predictor: {pred.mean() - base:+.1f} points")
        if after.any():
            print(f"    subset repaved after the survey ended (Aug 2023): {int(after.sum())} | prediction "
                  f"{pred[after].mean():.1f} vs label {lab[after].mean():.1f} (context)")
        pd.DataFrame(dict(seg_key=sk, rd20full=stale_seg["rd20full"], pci23=lab, predicted=pred,
                          repaired_after_survey_end=after)).to_csv(out_dir / "v7_stale_predictions.csv", index=False)
    else:
        print("\nV7: no stale set found (data/modeling/trim<T>/stale_segments.csv) -> not run")
    print(f"\nsaved: {out_dir}/v6_results.csv, v6_oof_predictions.csv" + (", v7_stale_predictions.csv" if v7 is not None else ""))
    return v6, v7


def real():
    rules = json.loads((ROOT / "04_pipeline_design/pipeline_rules.json").read_text())
    d = ROOT / f"data/modeling/trim{rules['end_trim_ahead_m']:g}"
    segs = pd.read_csv(d / "segments.csv")
    seg = segs[segs["set_primary"]].reset_index(drop=True)
    areas = seg["cpname"].value_counts()
    if not (len(seg) >= 150 and (areas >= 20).sum() >= 2):
        stop(f"gate not met: {len(seg)} primary segments, {int((areas >= 20).sum())} areas with >= 20 "
             f"(need >= 150 across >= 2). No preliminary V6 (decided 2026-11-14).")
    frames = pd.read_csv(d / "frames.csv", index_col="row")
    kind = rules["embedding"]["primary_vector"]
    vec = vectors(np.load(d / "frame_vectors.npz"), kind)
    meta = pd.read_csv(ROOT / "data/processed/segments_clean.csv", low_memory=False)
    meta["iamfloc"] = meta["iamfloc"].astype(str).str.strip().str.upper()
    meta = meta.drop_duplicates("iamfloc")
    st_f = d / "stale_segments.csv"
    if st_f.exists():
        ss = pd.read_csv(st_f)
        ss = ss[ss["set_stale"]].reset_index(drop=True)
        sf = pd.read_csv(d / "stale_frames.csv", index_col="row")
        sv = vectors(np.load(d / "stale_frame_vectors.npz"), kind)
    else:
        ss = sf = sv = None
    run(seg, frames, vec, meta, ss, sf, sv, ROOT / "data/modeling", kind)


def synthetic(signal, seed=0):
    """fake primary set + stale set with the real file layout; signal=False shuffles the labels"""
    rng = np.random.default_rng(seed)
    D = 384
    n_seg = 160
    pci = np.clip(rng.normal(66, 26, n_seg), 0, 100)
    streets = [f"ST {i % 45}" for i in range(n_seg)]
    areas = rng.choice(["A", "B", "C"], n_seg)
    fr_rows, vec = [], []
    w = rng.normal(size=D)
    def frames_for(key, p, k):
        for _ in range(k):
            fr_rows.append(dict(seg_key=key, gray_mean=100 + 0.2 * p + rng.normal(0, 15), gray_std=10 + rng.normal(0, 2),
                                lap_var=float(np.exp(3 + rng.normal(0, 0.5)))))
            vec.append((0.02 * (p - 66) * w / np.linalg.norm(w)) * 8 + rng.normal(0, 1, D))
    keys = [f"SS-{i:06d}-PV1" for i in range(n_seg)]
    lab = pci if signal else rng.permutation(pci)
    for kk, p in zip(keys, pci):
        frames_for(kk, p, int(rng.integers(3, 20)))
    seg = pd.DataFrame(dict(seg_key=keys, rd20full=streets, cpname=areas, pci23=lab,
                            pci23_d=pd.cut(lab, [-1, 10, 25, 40, 55, 70, 85, 101], labels=BANDS[::-1]).astype(str),
                            set_primary=True))
    n_fr = len(fr_rows)
    sk = [f"SS-9{i:05d}-PV1" for i in range(20)]
    for kk in sk:
        frames_for(kk, 90.0, 6)                                   # repaved: looks like ~90
    frames = pd.DataFrame(fr_rows)
    vec = np.array(vec, np.float32)
    meta = pd.DataFrame(dict(iamfloc=keys, pwidth=rng.normal(36, 5, n_seg), pav_length=rng.normal(400, 100, n_seg),
                             lane_no=2, has_prior_repair=rng.integers(0, 2, n_seg),
                             years_since_repair=np.where(rng.random(n_seg) < 0.3, np.nan, rng.uniform(0, 15, n_seg)),
                             cpname=areas, zip=rng.choice([92109, 92126], n_seg)))
    stale_seg = pd.DataFrame(dict(seg_key=sk, rd20full=[f"SX {i}" for i in range(20)], pci23=rng.uniform(20, 50, 20),
                                  repaired_after_survey_end=rng.random(20) < 0.8, set_stale=True))
    return (seg, frames.iloc[:n_fr].reset_index(drop=True), vec[:n_fr], meta, stale_seg,
            frames.iloc[n_fr:].reset_index(drop=True), vec[n_fr:])


def selftest():
    out = Path(tempfile.mkdtemp())
    print("=== SELFTEST 1: synthetic data WITH signal (V6 and V7 must PASS)")
    v6a, v7a = run(*synthetic(True), out, "synthetic")
    print("\n=== SELFTEST 2: same data, labels SHUFFLED (V6 must FAIL)")
    v6b, _ = run(*synthetic(False), out, "synthetic")
    ok = v6a and v7a and not v6b
    print(f"\nSELFTEST {'OK' if ok else 'FAILED'}: signal -> V6 {v6a}, V7 {v7a} | shuffled -> V6 {v6b} "
          f"(want True, True | False)")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="V6 (signal) and V7 (stale-set sanity), frozen rules")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else real()
