# 05_training/ladder.py
# Model ladder (Day 5, M5): rungs 1-5 on the same segments and the same folds, nested grouped CV, two schemes:
#   leave-street-out  outer GroupKFold(5) by street (rd20full), inner GroupKFold(5) by street   (= the V6 folds)
#   leave-area-out    outer = one planning-area group held out at a time; inner = one training area held out at a time
# Rungs: 1 mean predictor | 2a metadata ridge, 2b metadata gradient boosting | 3 hand-crafted ridge |
#        4 ridge on pooled DINOv2 (mean || std) | 5 gated attention-MIL (mil.py) | combined 2a+4 (exploratory)
# Rungs 1-4 reuse v6_v7.py's feature and model code, so on the same data leave-street-out rungs 1, 2a, 3, 4 ARE V6
# (--check-v6 proves it on the Day 4C tables).
#
# PRE-REGISTERED 2026-11-15 (Day 5), approved by Wael before any ladder result. This header and the constants below
# are the record; the file's SHA-256 goes into the Day 5 handoff.
#  A  Dataset: frozen pipeline tables (trim 30, primary set) including Uptown (2026-11-15), the LAST drive: no more
#     drives whatever the ladder shows. Uptown was driven after V6 and the Gate 2 preview were seen (disclosed).
#  B  Leave-area-out: one fold per planning area (cpname); an area with < 10 primary segments joins the main area of
#     its drive (the drive's most common cpname) -> Balboa Park joins Greater Golden Hill. Inner CV grouped the same
#     way. Metadata one-hots ignore an unseen area.
#  C  Gate 2, per scheme: rung 4 vs the stronger rung 2 (lower MAE). "images BEAT metadata" = the 95% CI of the paired
#     MAE difference (2,000 bootstrap resamples of streets, seed 0) lies below 0; CI above 0 = "metadata better";
#     otherwise "comparable".
#  D  Rung 2 = ridge (as in V6) and HistGradientBoosting (HGB_GRID), both reported.
#  E  Rung 5: dropout {0.25, 0.5} x weight decay {1e-4, 1e-3, 1e-2} x epochs {50, 100, 200} (epochs read at checkpoints
#     of one 200-epoch run); lr 1e-3, batch 16, seed 0 set before the model is built; CPU (repeatable, mps is not).
#     Fallback fixed in advance: if one 200-epoch run takes > 15 s (--bench), leave-street-out MIL uses 3 inner folds.
#  F  H3: rung-4 (images) and rung-2a (metadata) predictions for every stale segment, each from models trained without
#     that segment's street, next to the mean predictor (guard).
#  G  Ablations (rungs 1-4 only): --trim 15/37, --vector cls/patch_mean, --pooling mean. If time: MIL on tiles,
#     repave filter vs none. Cut: ImageNet-mean fill, DINOv2-base.
#  Routine: metrics MAE (primary), RMSE, Spearman, within-one-band accuracy, MAE Poor->Failed, per-band MAE; predictions
#     clipped to 0-100; all tuning inside the training fold; "combined" = fixed 50/50 average of rungs 2a and 4
#     (added after V6 showed complementary errors; EXPLORATORY).
#  ADDENDUM 2026-11-15 (after Uptown's pipeline checks, BEFORE any ladder result): Uptown FAILED V3-local (13/20, seed 607;
#     SegFormer-B0 leaves light concrete roadway untinted; on that drive tile rejection tracked condition). Per the
#     pre-registered V3-local rule (Day 4C, D1) and A, Uptown is OUT of the analysis. The dataset is the Day 4C freeze:
#     data/modeling_4C (5 drives, 173 primary, 52 stale; identical to the rebuilt tables minus Uptown; the V6 data).
#     Run everything with --data data/modeling_4C.
#
# Usage (project root, pavement env active):
#   python 05_training/ladder.py --selftest                  # synthetic data (+ a small real-torch MIL test)
#   python 05_training/ladder.py --check-v6 data/modeling_4C # must reproduce V6's out-of-fold predictions exactly
#   python 05_training/ladder.py --bench                     # time one MIL run on CPU -> data/modeling/ladder/bench.json
#   python 05_training/ladder.py --data data/modeling_4C --mil --h3     # MAIN RUN (frozen settings), both schemes, Gate 2, H3
#   python 05_training/ladder.py --data data/modeling_4C --trim 15      # ablations: --trim 15|37, --vector cls|patch_mean,
#                                                                       #            --pooling mean
# Writes <data>/ladder/<tag>/ : results.csv, per_band_mae.csv, gate2.csv, oof_predictions.csv
#   (+ mil_choices.csv, h3_stale_predictions.csv)

import argparse, hashlib, json, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v6_v7 as v6                     # rungs 1-4: features, pipelines, alpha grid, metrics exactly as in V6
import mil                             # rung 5 (torch is imported inside its functions)

ROOT = v6.ROOT
RULES = ROOT / "04_pipeline_design/pipeline_rules.json"
OUT = ROOT / "data/modeling/ladder"
SEED = 0
OUTER_STREET = INNER_STREET = 5
MIN_AREA = 10                          # B
HGB_GRID = [dict(max_depth=d, learning_rate=lr, max_iter=it) for d in (2, 3) for lr in (0.03, 0.1) for it in (100, 300)]
MIL_GRID = [(dr, wd) for dr in (0.25, 0.5) for wd in (1e-4, 1e-3, 1e-2)]
MIL_EPOCHS = (50, 100, 200)
MIL_SLOW_S, MIL_INNER_FALLBACK = 15.0, 3
BOOT_B = 2000
EDGES = [10, 25, 40, 55, 70, 85]       # City bands, left-inclusive (checked in segments_clean.csv)
BAND_UP = ["Failed", "Serious", "Very Poor", "Poor", "Fair", "Satisfactory", "Good"]
R2A, R2B, R3, R4, R5 = "2a metadata ridge", "2b metadata boosting", "3 hand-crafted ridge", "4 DINOv2 ridge", "5 attention-MIL"
COMB = "combined 2a+4 (exploratory)"


def stop(msg):
    sys.exit(f"STOP: {msg}")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------- data
def meta_frames(meta, keys):
    """metadata rows for these segments: ridge version (no record -> 30, cap 30, as V6) and boosting version (raw)"""
    X = meta.loc[keys, v6.NUM_META + v6.CAT_META].reset_index(drop=True)
    X[v6.CAT_META] = X[v6.CAT_META].astype(str)
    Xr = X.copy()
    Xr["years_since_repair"] = Xr["years_since_repair"].fillna(v6.AGE_CAP).clip(upper=v6.AGE_CAP)
    return Xr, X


def area_groups(seg):
    """B: planning area; an area with < MIN_AREA primary segments joins its drive's most common area"""
    area = seg["cpname"].astype(str)
    drive = (seg["drives"].astype(str).str.split(r"[;,|\s]+").str[0] if "drives" in seg
             else pd.Series("all", index=seg.index))
    main = area.groupby(drive).agg(lambda s: s.value_counts().idxmax())
    small = area.map(area.value_counts()) < MIN_AREA
    merged = {a: main[dr] for a, dr in zip(area[small], drive[small])}
    return area.where(~small, drive.map(main)).to_numpy(), merged


def assemble(seg, frames, vec, meta, stale_seg=None, stale_frames=None, stale_vec=None, pooling="mean_std"):
    keys = seg["seg_key"].tolist()
    D = vec.shape[1]
    Xd = v6.pooled(frames, vec, keys)
    if pooling == "mean":
        Xd = Xd[:, :D]
    Xr, Xg = meta_frames(meta, keys)
    rows = frames.groupby("seg_key").indices
    areas, merged = area_groups(seg)
    d = dict(seg=seg, keys=keys, y=seg["pci23"].to_numpy(float), streets=seg["rd20full"].to_numpy(),
             bands=seg["pci23_d"].to_numpy(), areas=areas, merged=merged, Xd=Xd, Xh=v6.handcrafted(frames, keys),
             Xr=Xr, Xg=Xg, meta=meta, pooling=pooling, D=D,
             bags=[(k, vec[rows[k]].astype(np.float32), float(p) / 100) for k, p in zip(keys, seg["pci23"])])
    if stale_seg is not None and len(stale_seg):
        sk = stale_seg["seg_key"].tolist()
        Xs = v6.pooled(stale_frames, stale_vec, sk)
        d.update(stale=stale_seg.reset_index(drop=True), Xd_stale=Xs[:, :D] if pooling == "mean" else Xs,
                 Xr_stale=meta_frames(meta, sk)[0])
    return d


def load(trim_dir, vector="concat", pooling="mean_std"):
    if not (trim_dir / "segments.csv").exists():
        stop(f"missing {trim_dir}/segments.csv")
    segs = pd.read_csv(trim_dir / "segments.csv")
    seg = segs[segs["set_primary"]].reset_index(drop=True)
    frames = pd.read_csv(trim_dir / "frames.csv", index_col="row")
    kind = "concat_cls_patchmean" if vector == "concat" else vector
    vec = v6.vectors(np.load(trim_dir / "frame_vectors.npz"), kind)
    if len(vec) != len(frames):
        stop("frames.csv and frame_vectors.npz have different lengths")
    meta = pd.read_csv(ROOT / "data/processed/segments_clean.csv", low_memory=False)
    meta["iamfloc"] = meta["iamfloc"].astype(str).str.strip().str.upper()
    meta = meta.drop_duplicates("iamfloc").set_index("iamfloc")
    ss = sf = sv = None
    if (trim_dir / "stale_segments.csv").exists():
        ss = pd.read_csv(trim_dir / "stale_segments.csv")
        ss = ss[ss["set_stale"]].reset_index(drop=True)
        sf = pd.read_csv(trim_dir / "stale_frames.csv", index_col="row")
        sv = v6.vectors(np.load(trim_dir / "stale_frame_vectors.npz"), kind)
    return assemble(seg, frames, vec, meta, ss, sf, sv, pooling)


# ------------------------------------------------------------------------------------- rungs 1-4
def hgb_pipe(p):
    pre = ColumnTransformer([("num", "passthrough", v6.NUM_META),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), v6.CAT_META)])
    return make_pipeline(pre, HistGradientBoostingRegressor(min_samples_leaf=10, random_state=SEED, **p))


def rung_models(d):
    """name -> (pipeline maker, grid, features); None = the mean predictor"""
    return {"1 mean predictor": None, R2A: (v6.meta_pipe, v6.ALPHAS, d["Xr"]), R2B: (hgb_pipe, HGB_GRID, d["Xg"]),
            R3: (v6.ridge_pipe, v6.ALPHAS, d["Xh"]), R4: (v6.ridge_pipe, v6.ALPHAS, d["Xd"])}


def tune(make, grid, X, y, groups, k_max):
    """inner grouped CV on the training data only -> the grid value with the lowest MAE (first one on ties, as V6)"""
    k = min(k_max, len(np.unique(groups)))
    best, best_mae = None, np.inf
    for p in grid:
        err = []
        for tr, va in GroupKFold(n_splits=k).split(X, y, groups):
            m = make(p).fit(v6.sub(X, tr), y[tr])
            err.append(np.abs(np.clip(m.predict(v6.sub(X, va)), 0, 100) - y[va]))
        mae = np.concatenate(err).mean()
        if mae < best_mae:
            best, best_mae = p, mae
    return best


def folds(d, scheme):
    """groups, outer splits, inner fold count for a scheme"""
    if scheme == "street":
        g, k_out, k_in = d["streets"], OUTER_STREET, INNER_STREET
    else:
        g = d["areas"]
        k_out = k_in = len(np.unique(g))          # one area group per fold, inside and out
    return g, list(GroupKFold(n_splits=k_out).split(np.zeros(len(g)), d["y"], g)), k_in


def cv_rungs(d, groups, outer, k_in, names):
    y, ms = d["y"], rung_models(d)
    oof = {n: np.full(len(y), np.nan) for n in names}
    chosen = {n: [] for n in names}
    fold = np.full(len(y), -1)
    for f, (tr, te) in enumerate(outer):
        fold[te] = f
        for n in names:
            if ms[n] is None:
                oof[n][te] = y[tr].mean()
                continue
            make, grid, X = ms[n]
            p = tune(make, grid, v6.sub(X, tr), y[tr], groups[tr], k_in)
            chosen[n].append(p)
            oof[n][te] = np.clip(make(p).fit(v6.sub(X, tr), y[tr]).predict(v6.sub(X, te)), 0, 100)
    return oof, chosen, fold


# ----------------------------------------------------------------------------------------- rung 5
def fit_mil(train_bags, eval_bags, dropout, wd, ckpts, dev="cpu"):
    """one MIL training run; returns {epoch: predictions (PCI 0-100) for eval_bags} at the checkpoint epochs"""
    import torch
    torch.use_deterministic_algorithms(dev == "cpu")          # repeatable on CPU (mps training is not bit-for-bit)
    mu, sd = mil.standardizer(train_bags)
    torch.manual_seed(SEED)                                   # before the model is built: the weights are drawn here
    model = mil.make_model(train_bags[0][1].shape[1], dropout=dropout)
    out = {}

    def hook(ep, m):
        if ep in ckpts:
            out[ep] = 100 * mil.predict(m, eval_bags, mu, sd, dev)[0]
    mil.train(model, train_bags, mu, sd, dev, epochs=max(ckpts), weight_decay=wd, seed=SEED, on_epoch=hook)
    return out


def cv_mil(d, groups, outer, k_in, grid=MIL_GRID, ckpts=MIL_EPOCHS, log=True):
    y, bags = d["y"], d["bags"]
    oof, rec, t0 = np.full(len(y), np.nan), [], time.time()
    for f, (tr, te) in enumerate(outer):
        k = min(k_in, len(np.unique(groups[tr])))
        err = {(dr, wd, e): [] for dr, wd in grid for e in ckpts}
        for itr, iva in GroupKFold(n_splits=k).split(np.zeros(len(tr)), groups=groups[tr]):
            tb, vb = [bags[tr[i]] for i in itr], [bags[tr[i]] for i in iva]
            for dr, wd in grid:
                pr = fit_mil(tb, vb, dr, wd, ckpts)
                for e in ckpts:
                    err[(dr, wd, e)].append(np.abs(pr[e] - y[tr[iva]]))
        mae = {key: float(np.concatenate(v).mean()) for key, v in err.items()}
        dr, wd, e = best = min(mae, key=mae.get)                # first minimum in grid order
        oof[te] = fit_mil([bags[i] for i in tr], [bags[i] for i in te], dr, wd, (e,))[e]
        rec.append(dict(fold=f, dropout=dr, weight_decay=wd, epochs=e, inner_MAE=round(mae[best], 3),
                        inner_folds=k, test_segments=len(te)))
        if log:
            print(f"   MIL fold {f + 1}/{len(outer)}: dropout {dr}, weight decay {wd:g}, {e} epochs "
                  f"(inner MAE {mae[best]:.2f}, {k} inner folds) | {time.time() - t0:.0f} s", flush=True)
    return oof, rec


# ------------------------------------------------------------------------------------------ report
def metrics(y, p, bands):
    m = v6.metrics(y, p, bands)
    bt = np.array([BAND_UP.index(b) for b in bands])
    m["within_one_band"] = float(np.mean(np.abs(np.searchsorted(EDGES, p, side="right") - bt) <= 1))
    return {k: m[k] for k in ["MAE", "RMSE", "Spearman", "within_one_band", "MAE_Poor_to_Failed"]}


def boot_diff(err_a, err_b, streets, B=BOOT_B):
    """MAE(a) - MAE(b) and its 95% CI, resampling whole streets (paired)"""
    dif = err_a - err_b
    _, inv = np.unique(streets, return_inverse=True)
    s, c = np.bincount(inv, weights=dif), np.bincount(inv)
    idx = np.random.default_rng(SEED).integers(0, len(s), size=(B, len(s)))
    lo, hi = np.percentile(s[idx].sum(1) / c[idx].sum(1), [2.5, 97.5])
    return float(dif.mean()), float(lo), float(hi)


def verdict(lo, hi):
    return "images BEAT metadata" if hi < 0 else ("metadata better" if lo > 0 else "comparable")


def fmt(ps):
    if not ps:
        return "-"
    if isinstance(ps[0], dict):
        return ", ".join(f"d{p['max_depth']}/lr{p['learning_rate']:g}/{p['max_iter']}" for p in ps)
    return ", ".join(f"{p:g}" for p in ps)


def run_scheme(d, scheme, use_mil=False, mil_k_street=INNER_STREET, names=None, quiet=False):
    groups, outer, k_in = folds(d, scheme)
    y, bands = d["y"], d["bands"]
    names = names or list(rung_models(d))
    t = time.time()
    oof, chosen, fold = cv_rungs(d, groups, outer, k_in, names)
    mil_rec = None
    if use_mil:
        if not quiet:
            print(f"   rungs 1-4 done in {time.time() - t:.0f} s; MIL: {len(MIL_GRID)} settings x "
                  f"{len(MIL_EPOCHS)} checkpoints per inner fold", flush=True)
        oof[R5], mil_rec = cv_mil(d, groups, outer, mil_k_street if scheme == "street" else k_in, log=not quiet)
    if R2A in oof and R4 in oof:
        oof[COMB] = 0.5 * (oof[R2A] + oof[R4])
    res = pd.DataFrame({n: metrics(y, p, bands) for n, p in oof.items()}).T
    res["chosen (outer folds)"] = [fmt(chosen.get(n)) for n in oof]
    if mil_rec:
        res.loc[R5, "chosen (outer folds)"] = ", ".join(f"do{m['dropout']}/wd{m['weight_decay']:g}/{m['epochs']}ep"
                                                         for m in mil_rec)
    per = pd.DataFrame({"band": bands, "y": y, **oof})
    tab = pd.DataFrame({n: (per[n] - per["y"]).abs().groupby(per["band"]).mean() for n in oof})
    tab.insert(0, "n", per.groupby("band").size())
    tab = tab.reindex([b for b in v6.BANDS if b in tab.index])
    gate = []
    r2 = [n for n in (R2A, R2B) if n in oof]
    if r2 and R4 in oof:
        best2 = min(r2, key=lambda n: res.loc[n, "MAE"])
        for img in [R4] + ([R5] if R5 in oof else []):
            dif, lo, hi = boot_diff(np.abs(oof[img] - y), np.abs(oof[best2] - y), d["streets"])
            gate.append(dict(scheme=scheme, images=img, metadata=best2, MAE_images=res.loc[img, "MAE"],
                             MAE_metadata=res.loc[best2, "MAE"], difference=dif, ci_low=lo, ci_high=hi,
                             verdict=verdict(lo, hi), role="GATE 2" if img == R4 else "context"))
    return dict(scheme=scheme, oof=oof, fold=fold, groups=groups, res=res, per_band=tab, gate=pd.DataFrame(gate),
                mil=mil_rec, n_outer=len(outer), k_in=k_in, secs=time.time() - t)


def show(r, d):
    s = r["scheme"]
    head = (f"LEAVE-STREET-OUT (outer {r['n_outer']} folds by street, inner {r['k_in']} by street)" if s == "street" else
            f"LEAVE-AREA-OUT (outer {r['n_outer']} area groups, inner: one training area held out at a time)")
    print(f"\n=== {head} | {len(d['y'])} segments, {len(np.unique(d['streets']))} streets | {r['secs']:.0f} s")
    print(r["res"].round(3).to_string())
    print("\nMAE by band (PCI points):")
    print(r["per_band"].round(1).to_string())
    for g in r["gate"].itertuples(index=False):
        print(f"{g.role}: {g.images} {g.MAE_images:.2f} vs {g.metadata} {g.MAE_metadata:.2f} -> difference "
              f"{g.difference:+.2f} (95% CI {g.ci_low:+.2f} to {g.ci_high:+.2f}, bootstrap over streets) -> {g.verdict}")


def save(results, d, out, extra=None):
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([r["res"].assign(scheme=r["scheme"]) for r in results]).rename_axis("model").to_csv(out / "results.csv")
    pd.concat([r["per_band"].assign(scheme=r["scheme"]) for r in results]).to_csv(out / "per_band_mae.csv")
    pd.concat([r["gate"] for r in results]).to_csv(out / "gate2.csv", index=False)
    o = d["seg"][["seg_key", "rd20full", "cpname", "pci23", "pci23_d"]].copy()
    o["area_group"] = d["areas"]
    for r in results:
        o[f"fold_{r['scheme']}"] = r["fold"]
        for n, p in r["oof"].items():
            o[f"{r['scheme']} | {n}"] = p
    o.to_csv(out / "oof_predictions.csv", index=False)
    mil_rows = [dict(scheme=r["scheme"], **m) for r in results if r["mil"] for m in r["mil"]]
    if mil_rows:
        pd.DataFrame(mil_rows).to_csv(out / "mil_choices.csv", index=False)
    if extra is not None:
        extra.to_csv(out / "h3_stale_predictions.csv", index=False)
    print(f"\nsaved: {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}/ (results, per_band_mae, gate2, "
          f"oof_predictions{', mil_choices' if mil_rows else ''}{', h3_stale_predictions' if extra is not None else ''})")


# --------------------------------------------------------------------------------------------- H3
def h3(d, quiet=False):
    """F: each stale segment predicted by rung 4 and rung 2a trained WITHOUT its street; mean predictor as the guard"""
    ss = d["stale"]
    y, st = d["y"], d["streets"]
    lab, s_st = ss["pci23"].to_numpy(float), ss["rd20full"].to_numpy()
    img, met, base = (np.full(len(ss), np.nan) for _ in range(3))
    for s in np.unique(s_st):
        tr, te = np.where(st != s)[0], np.where(s_st == s)[0]
        a = tune(v6.ridge_pipe, v6.ALPHAS, d["Xd"][tr], y[tr], st[tr], INNER_STREET)
        img[te] = np.clip(v6.ridge_pipe(a).fit(d["Xd"][tr], y[tr]).predict(d["Xd_stale"][te]), 0, 100)
        a = tune(v6.meta_pipe, v6.ALPHAS, d["Xr"].iloc[tr], y[tr], st[tr], INNER_STREET)
        met[te] = np.clip(v6.meta_pipe(a).fit(d["Xr"].iloc[tr], y[tr]).predict(d["Xr_stale"].iloc[te]), 0, 100)
        base[te] = y[tr].mean()
    after = ss["repaired_after_survey_end"].astype(bool).to_numpy()
    dif, lo, hi = boot_diff(img, met, s_st)            # mean(images - metadata), 95% CI resampling streets
    if not quiet:
        print(f"\n=== H3: {len(ss)} stale segments on {len(np.unique(s_st))} streets "
              f"(each predicted by models that never saw its street)")
        print(f"  mean 2023 label {lab.mean():.1f} | mean predictor (guard) {base.mean():.1f} | metadata ridge (2a) "
              f"{met.mean():.1f} | DINOv2 ridge (4) {img.mean():.1f}")
        print(f"  images - metadata: {dif:+.1f} points (95% CI {lo:+.1f} to {hi:+.1f}, bootstrap over streets); "
              f"images higher on {int((img > met).sum())}/{len(ss)} segments")
        if after.any():
            print(f"  repaved after the survey ended (n={int(after.sum())}): label {lab[after].mean():.1f} | "
                  f"metadata {met[after].mean():.1f} | images {img[after].mean():.1f}")
    return pd.DataFrame(dict(seg_key=ss["seg_key"], rd20full=s_st, cpname=ss["cpname"], pci23=lab,
                             repaired_after_survey_end=after, mean_predictor=base, metadata_2a=met, images_4=img))


# ------------------------------------------------------------------------------------- run modes
def main_run(trim, vector, pooling, use_mil, use_h3, data="data/modeling"):
    data_dir = Path(data) if Path(data).is_absolute() else ROOT / data
    rules = json.loads(RULES.read_text())
    frozen = (trim is None or trim == rules["end_trim_ahead_m"]) and vector == "concat" and pooling == "mean_std"
    if (use_mil or use_h3) and not frozen:
        stop("--mil / --h3 run on the frozen settings only (ablations are rungs 1-4, pre-registered G)")
    T = trim if trim is not None else rules["end_trim_ahead_m"]
    d = load(data_dir / f"trim{T:g}", vector, pooling)
    mil_k = INNER_STREET
    if use_mil:
        bj = OUT / "bench.json"
        if not bj.exists():
            stop("run --bench first (it applies the pre-registered fallback rule E)")
        mil_k = json.loads(bj.read_text())["street_inner_folds_mil"]
    parts = ([f"trim{T:g}"] if T != rules["end_trim_ahead_m"] else []) + ([f"vec_{vector}"] if vector != "concat" else []) \
        + (["pool_mean"] if pooling == "mean" else [])
    tag = "_".join(parts) or "frozen" + ("_mil" if use_mil else "")
    n_area = pd.Series(d["areas"]).value_counts()
    print(f"MODEL LADDER | ladder.py {sha(__file__)[:12]} | rules {sha(RULES)[:12]} | {data_dir.name}/trim{T:g} | "
          f"vector {vector} | pooling {pooling} | tag {tag}")
    print(f"{len(d['y'])} primary segments, {len(np.unique(d['streets']))} streets | drives: "
          f"{', '.join(sorted(d['seg']['drives'].astype(str).unique()))}")
    print("area groups: " + ", ".join(f"{a} {n}" for a, n in n_area.items())
          + (f" | merged (< {MIN_AREA}): " + ", ".join(f"{a} -> {b}" for a, b in d["merged"].items()) if d["merged"] else ""))
    if use_mil:
        print(f"MIL on CPU; leave-street-out inner folds {mil_k} (from bench.json)")
    results = []
    for scheme in ("street", "area"):
        r = run_scheme(d, scheme, use_mil, mil_k)
        show(r, d)
        results.append(r)
    stale = h3(d) if use_h3 and "stale" in d else None
    if use_h3 and "stale" not in d:
        print("\nH3: no stale set found -> not run")
    save(results, d, data_dir / "ladder" / tag, stale)


def check_v6(dir_):
    """rungs 1, 2a, 3, 4 (leave-street-out) on the Day 4C tables must equal V6's saved out-of-fold predictions"""
    dir_ = Path(dir_) if Path(dir_).is_absolute() else ROOT / dir_
    ref_f = dir_ / "v6_oof_predictions.csv"
    if not ref_f.exists():
        stop(f"missing {ref_f}")
    d = load(dir_ / "trim30")
    names = ["1 mean predictor", R2A, R3, R4]
    r = run_scheme(d, "street", names=names, quiet=True)
    ref = pd.read_csv(ref_f).set_index("seg_key").loc[d["keys"]]
    pairs = {"1 mean predictor": "mean predictor", R2A: "ridge metadata (rung 2, context)",
             R3: "ridge hand-crafted (rung 3, context)", R4: "ridge DINOv2 (V6)"}
    worst = 0.0
    print(f"CHECK V6 on {dir_}: {len(d['y'])} segments, {len(np.unique(d['streets']))} streets")
    for n, col in pairs.items():
        diff = float(np.abs(r["oof"][n] - ref[col].to_numpy()).max())
        worst = max(worst, diff)
        print(f"  {n:24s} MAE {r['res'].loc[n, 'MAE']:.3f} | max |difference| from V6's saved predictions {diff:.1e}")
    ok = worst < 1e-6
    print(f"CHECK V6 {'PASS' if ok else 'FAIL'}: the ladder {'reproduces' if ok else 'does NOT reproduce'} V6 "
          f"(want every difference < 1e-6)")
    if not ok:
        sys.exit(1)


def bench():
    """E: time one 200-epoch MIL run on CPU (~80% of the frozen primary bags) and apply the pre-registered fallback"""
    import torch
    rules = json.loads(RULES.read_text())
    d = load(ROOT / f"data/modeling/trim{rules['end_trim_ahead_m']:g}")
    n = len(d["bags"])
    idx = np.random.default_rng(SEED).permutation(n)[: int(0.8 * n)]
    tb = [d["bags"][i] for i in idx]
    t = time.time()
    fit_mil(tb, tb[:5], 0.25, 1e-4, (200,), "cpu")
    cpu = time.time() - t
    mps = None
    if torch.backends.mps.is_available():
        t = time.time()
        fit_mil(tb, tb[:5], 0.25, 1e-4, (200,), "mps")
        mps = time.time() - t
    k_st = INNER_STREET if cpu <= MIL_SLOW_S else MIL_INNER_FALLBACK
    g = len(np.unique(d["areas"]))
    runs = OUTER_STREET * (k_st * len(MIL_GRID) + 1) + g * ((g - 1) * len(MIL_GRID) + 1)
    print(f"BENCH: one 200-epoch MIL run on {len(tb)} bags: CPU {cpu:.1f} s" + (f" | mps {mps:.1f} s (info)" if mps else ""))
    print(f"rule E (pre-registered): > {MIL_SLOW_S:g} s -> 3 inner folds for leave-street-out MIL | "
          f"decision: {k_st} inner folds")
    print(f"MIL main run: {runs} training runs (street {OUTER_STREET} x ({k_st} x {len(MIL_GRID)} + 1), area {g} x "
          f"({g - 1} x {len(MIL_GRID)} + 1)) -> about {runs * cpu / 60:.0f} min on CPU (estimate)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bench.json").write_text(json.dumps(dict(date=time.strftime("%Y-%m-%d %H:%M"), bags=len(tb), cpu_s=round(cpu, 2),
                                                    mps_s=round(mps, 2) if mps else None, rule_s=MIL_SLOW_S,
                                                    street_inner_folds_mil=k_st, runs=runs), indent=1))
    print(f"saved: {(OUT / 'bench.json').relative_to(ROOT)}")


# -------------------------------------------------------------------------------------- selftest
def synthetic(signal):
    """v6_v7's synthetic set + drives, a tiny area to merge, and metadata for the stale segments"""
    seg, frames, vec, meta, ss, sf, sv = v6.synthetic(signal)
    seg = seg.copy()
    seg["drives"] = "drive_" + seg["cpname"]
    seg.loc[seg.index[:4], "cpname"] = "TINY"                    # 4 segments, all on drive_A's main area's drive
    seg.loc[seg.index[:4], "drives"] = "drive_A"
    rng = np.random.default_rng(5)
    sm = pd.DataFrame(dict(iamfloc=ss["seg_key"], pwidth=36.0, pav_length=400.0, lane_no=2, has_prior_repair=1,
                           years_since_repair=rng.uniform(5, 15, len(ss)), cpname="A", zip=92109))
    ss = ss.assign(cpname="A")
    meta = pd.concat([meta, sm], ignore_index=True).set_index("iamfloc")
    return assemble(seg, frames, vec, meta, ss, sf, sv)


def selftest():
    ok = {}
    print("=== 1) leave-street-out, synthetic WITH signal: rung 4 must beat the mean (V6 rule)")
    d = synthetic(True)
    r = run_scheme(d, "street", quiet=True)
    show(r, d)
    m0, m4, rho = r["res"].loc["1 mean predictor", "MAE"], r["res"].loc[R4, "MAE"], r["res"].loc[R4, "Spearman"]
    ok["signal"] = m4 <= 0.9 * m0 and rho >= 0.3
    print("\n=== 2) same, labels SHUFFLED: rung 4 must NOT beat the mean")
    ds = synthetic(False)
    rs = run_scheme(ds, "street", names=["1 mean predictor", R4], quiet=True)
    ok["shuffled"] = not (rs["res"].loc[R4, "MAE"] <= 0.9 * rs["res"].loc["1 mean predictor", "MAE"])
    print(rs["res"].round(3).to_string())
    print("\n=== 3) leave-area-out: TINY (4 segments) must merge into drive_A's main area; one fold per area group")
    ra = run_scheme(d, "area", quiet=True)
    show(ra, d)
    ok["areas"] = d["merged"] == {"TINY": "A"} and ra["n_outer"] == 3 and all(np.isfinite(p).all() for p in ra["oof"].values())
    print(f"merged: {d['merged']} | outer folds {ra['n_outer']}")
    print("\n=== 4) bootstrap: identical errors -> 0 (comparable); errors 5 lower -> images BEAT")
    e = np.abs(np.random.default_rng(1).normal(10, 5, len(d["y"])))
    z, b = boot_diff(e, e, d["streets"]), boot_diff(e - 5, e, d["streets"])
    ok["bootstrap"] = z == (0.0, 0.0, 0.0) and verdict(*z[1:]) == "comparable" and verdict(*b[1:]) == "images BEAT metadata"
    print(f"identical: {z} -> {verdict(*z[1:])} | 5 lower: {tuple(round(x, 2) for x in b)} -> {verdict(*b[1:])}")
    print("\n=== 5) H3 on synthetic stale segments that 'look repaved': images must predict above metadata")
    h = h3(d)
    ok["h3"] = h["images_4"].mean() > h["metadata_2a"].mean() and h[["images_4", "metadata_2a"]].notna().all().all()
    try:
        import torch  # noqa: F401
        have_torch = True
    except ImportError:
        have_torch = False
    if have_torch:
        print("\n=== 6) MIL (real torch, CPU): checkpoint = fresh run, repeatable, learns (small grid)")
        tb, vb = d["bags"][:100], d["bags"][100:130]
        a = fit_mil(tb, vb, 0.25, 1e-4, (10, 20))
        b10 = fit_mil(tb, vb, 0.25, 1e-4, (10,))[10]
        a2 = fit_mil(tb, vb, 0.25, 1e-4, (10, 20))
        ck, rep = float(np.abs(a[10] - b10).max()), float(np.abs(a[20] - a2[20]).max())
        g, outer, _ = folds(d, "street")
        oof, rec = cv_mil(d, g, outer, 2, grid=[(0.25, 1e-4), (0.5, 1e-2)], ckpts=(25, 50), log=False)
        m_mil = float(np.abs(oof - d["y"]).mean())
        ok["mil"] = ck < 1e-4 and rep < 1e-6 and m_mil < 0.9 * m0 and np.isfinite(oof).all()
        print(f"checkpoint epoch 10 vs fresh 10-epoch run: max diff {ck:.1e} (want < 1e-4) | repeat run diff {rep:.1e} "
              f"(want < 1e-6) | nested MIL MAE {m_mil:.2f} vs mean {m0:.2f} (want < 90%) | choices "
              f"{[(x['dropout'], x['weight_decay'], x['epochs']) for x in rec]}")
    else:
        print("\n=== 6) MIL skipped: torch not installed here (runs on the Mac)")
    good = all(ok.values())
    print(f"\nSELFTEST {'OK' if good else 'FAILED'}: " + ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in ok.items()))
    if not good:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="model ladder: rungs 1-5, nested grouped CV, leave-street-out + leave-area-out")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-v6", metavar="DIR", help="folder holding the Day 4C trim30/ and v6_oof_predictions.csv")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--mil", action="store_true", help="add rung 5 (frozen settings only)")
    ap.add_argument("--h3", action="store_true", help="add the H3 stale-set comparison (frozen settings only)")
    ap.add_argument("--trim", type=int, choices=[15, 30, 37])
    ap.add_argument("--vector", choices=["concat", "cls", "patch_mean"], default="concat")
    ap.add_argument("--pooling", choices=["mean_std", "mean"], default="mean_std")
    ap.add_argument("--data", default="data/modeling", help="folder holding trim<T>/ (the analysis: data/modeling_4C)")
    a = ap.parse_args()
    import warnings
    from scipy.linalg import LinAlgWarning
    warnings.filterwarnings("ignore", category=LinAlgWarning)   # ridge's dual solve at small alphas: harmless (4C note)
    if a.selftest:
        selftest()
    elif a.check_v6:
        check_v6(a.check_v6)
    elif a.bench:
        bench()
    else:
        main_run(a.trim, a.vector, a.pooling, a.mil, a.h3, a.data)
