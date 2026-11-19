# 05_training/mil.py
# Rung 5 / Viability check V8: gated attention multiple-instance learning (Ilse, Tomczak & Welling, 2018),
# trained from scratch in PyTorch on frozen DINOv2 frame vectors.
#   bag      = one segment            instance = one kept frame (its vector: CLS + patch-mean, 768 numbers)
#   label    = the segment's 2023 PCI / 100
#   model    : frame vector -> Linear(768, 128) + ReLU + Dropout -> gated attention (tanh * sigmoid, 64)
#              -> softmax weights over the bag's frames -> weighted sum -> Linear(128, 1) = predicted PCI / 100
# Bags of different sizes are padded into mini-batches; padded slots get attention weight exactly 0.
#
# This file is also the rung-5 model module for the grouped/nested CV in 05_training/ladder.py
# (imports make_model, train, predict; train's on_epoch hook gives checkpoint predictions).
#
# Usage (project root, pavement env active):
#   python 05_training/mil.py --selftest   # ~30 s: synthetic bags; checks the math, padding, learning
#   python 05_training/mil.py --v8         # V8: train on the real primary bags; PASS = loss decreases, no errors
# V8 prints the training LOSS only. It is not a performance result (no held-out data) and none is reported.

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / ".git").exists())
RULES = ROOT / "04_pipeline_design/pipeline_rules.json"
HP = dict(d_hidden=128, d_attn=64, dropout=0.25, lr=1e-3, weight_decay=1e-4, epochs=200, batch=16, seed=0)


def stop(msg):
    sys.exit(f"STOP: {msg}")


# ------------------------------------------------------------------------------------------- data
def instance_matrix(vec, kind):
    """frame vectors -> instance features: 'concat' (CLS + patch-mean, 768), 'cls' or 'patch_mean' (384)"""
    if kind == "concat":
        return np.concatenate([vec["cls"], vec["patch_mean"]], axis=1)
    return vec[kind]


def load_bags(trim_dir, kind, which="set_primary"):
    """list of (seg_key, instance array K x D, label in 0-1) for the chosen analysis set"""
    fr = pd.read_csv(trim_dir / "frames.csv", index_col="row")
    vec = np.load(trim_dir / "frame_vectors.npz")
    X = instance_matrix(vec, kind)
    if len(X) != len(fr):
        stop("frames.csv and frame_vectors.npz have different lengths")
    seg = pd.read_csv(trim_dir / "segments.csv")
    keys = seg.loc[seg[which], "seg_key"]
    rows = fr.groupby("seg_key").indices
    bags = []
    for k in keys:
        r = rows.get(k)
        if r is None:
            stop(f"segment {k} is in the set but has no kept frames")
        bags.append((k, X[r].astype(np.float32), float(fr["pci23"].iloc[r[0]]) / 100.0))
    return bags


def standardizer(bags):
    """mean / std of all instances in these (training) bags"""
    allx = np.concatenate([b[1] for b in bags])
    mu, sd = allx.mean(0), allx.std(0)
    return mu, np.where(sd < 1e-6, 1.0, sd)


def pad(bags, mu, sd):
    """bags -> x (B, Kmax, D) standardized and zero-padded, mask (B, Kmax) True = real frame, y (B,)"""
    kmax = max(len(b[1]) for b in bags)
    d = bags[0][1].shape[1]
    x = np.zeros((len(bags), kmax, d), np.float32)
    m = np.zeros((len(bags), kmax), bool)
    for i, b in enumerate(bags):
        x[i, :len(b[1])] = (b[1] - mu) / sd
        m[i, :len(b[1])] = True
    return x, m, np.array([b[2] for b in bags], np.float32)


def numpy_forward(params, x, m):
    """reference forward pass in plain numpy (eval mode) to check the PyTorch model"""
    P = {k: v for k, v in params.items()}
    h = np.maximum(x @ P["feat.0.weight"].T + P["feat.0.bias"], 0)
    v = np.tanh(h @ P["V.weight"].T + P["V.bias"])
    u = 1 / (1 + np.exp(-(h @ P["U.weight"].T + P["U.bias"])))
    s = ((v * u) @ P["w.weight"].T + P["w.bias"])[..., 0]
    s = np.where(m, s, -np.inf)
    a = np.exp(s - s.max(1, keepdims=True)); a = a / a.sum(1, keepdims=True)
    z = (a[..., None] * h).sum(1)
    return (z @ P["head.weight"].T + P["head.bias"])[:, 0], a


# ------------------------------------------------------------------------------------------ model
def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def make_model(d_in, d_hidden=HP["d_hidden"], d_attn=HP["d_attn"], dropout=HP["dropout"]):
    import torch
    import torch.nn as nn

    class GatedAttentionMIL(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout))
            self.V = nn.Linear(d_hidden, d_attn)
            self.U = nn.Linear(d_hidden, d_attn)
            self.w = nn.Linear(d_attn, 1)
            self.head = nn.Linear(d_hidden, 1)

        def forward(self, x, mask):
            h = self.feat(x)                                                       # B, K, H
            s = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)  # B, K
            s = s.masked_fill(~mask, float("-inf"))                                # padded slots -> weight 0
            a = torch.softmax(s, dim=1)
            z = torch.bmm(a.unsqueeze(1), h).squeeze(1)                            # B, H
            return self.head(z).squeeze(-1), a

    return GatedAttentionMIL()


def train(model, bags, mu, sd, dev, epochs=HP["epochs"], batch=HP["batch"], lr=HP["lr"],
          weight_decay=HP["weight_decay"], seed=HP["seed"], log_every=0, on_epoch=None):
    """mini-batch training on padded bags, MSE on PCI/100; returns the per-epoch mean training loss.
    on_epoch(epoch, model), if given, runs after each epoch (the model ladder uses it to predict at epoch checkpoints,
    so one 200-epoch run also gives the 50- and 100-epoch models); training mode is restored after it.
    Without on_epoch, training is exactly as before (added 2026-11-15, Day 5)."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    for ep in range(epochs):
        order = rng.permutation(len(bags))
        tot = 0.0
        for i in range(0, len(order), batch):
            x, m, y = pad([bags[j] for j in order[i:i + batch]], mu, sd)
            x, m, y = torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev), torch.from_numpy(y).to(dev)
            pred, _ = model(x, m)
            loss = torch.mean((pred - y) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y)
        losses.append(tot / len(bags))
        if log_every and (ep == 0 or (ep + 1) % log_every == 0):
            print(f"  epoch {ep + 1:4d}: training loss {losses[-1]:.5f}")
        if on_epoch is not None:
            on_epoch(ep + 1, model)
            model.train()
    return losses


def predict(model, bags, mu, sd, dev, batch=64):
    """eval mode: predictions (PCI/100, clipped to 0-1) and attention weights per bag"""
    import torch
    model.to(dev).eval()
    preds, atts = [], []
    with torch.no_grad():
        for i in range(0, len(bags), batch):
            part = bags[i:i + batch]
            x, m, _ = pad(part, mu, sd)
            p, a = model(torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev))
            p, a = p.cpu().numpy(), a.cpu().numpy()
            preds.extend(np.clip(p, 0, 1)); atts.extend(a[j, :len(b[1])] for j, b in enumerate(part))
    return np.array(preds), atts


# --------------------------------------------------------------------------------------- selftest
def selftest():
    import torch
    dev = device()
    rng = np.random.default_rng(1)
    D = 32
    bags, n_keys = [], []
    for i in range(80):                    # a bag is "bad" (0.3) if ANY of its frames shows "distress", else 0.9
        k = int(rng.integers(3, 20)); x = rng.normal(size=(k, D)).astype(np.float32)
        n_key = int(rng.integers(0, 4)); x[:n_key, 0] += 4.0
        bags.append((f"S{i}", x, 0.3 if n_key else 0.9)); n_keys.append(n_key)
    mu, sd = standardizer(bags)

    # 1) PyTorch forward == numpy reference (eval mode), on the device
    model = make_model(D).to(dev).eval()
    x, m, _ = pad(bags[:8], mu, sd)
    with torch.no_grad():
        pt, pa = model(torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev))
    params = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    npred, na = numpy_forward(params, x, m)
    d1 = float(np.abs(pt.cpu().numpy() - npred).max())
    # 2) padding: a bag alone vs inside a batch with a longer bag -> same output; padded weights exactly 0
    xa, ma, _ = pad([bags[0]], mu, sd)
    with torch.no_grad():
        alone, _ = model(torch.from_numpy(xa).to(dev), torch.from_numpy(ma).to(dev))
    d2 = float(abs(alone.cpu().numpy()[0] - pt.cpu().numpy()[0]))
    pad_w = float(pa.cpu().numpy()[~m].max()) if (~m).any() else 0.0
    # 3) it learns: loss falls far below the mean predictor's, and attention goes to the distress frames
    y = np.array([b[2] for b in bags]); base = float(((y - y.mean()) ** 2).mean())
    model = make_model(D)
    losses = train(model, bags, mu, sd, dev, epochs=150)
    _, att = predict(model, bags, mu, sd, dev)
    key_w = float(np.mean([a[:n].mean() * len(a) for a, n in zip(att, n_keys) if n]))
    # PASS/FAIL = the three exact checks + learning. The attention ratio is INFORMATIONAL only (changed 2026-11-14):
    # in this toy task the distress frames are detectable without attention concentrating, and mps training is not
    # bit-for-bit repeatable (1.5x and 1.4x on two identical runs), so it cannot be a pass/fail test.
    ok = d1 < 1e-4 and d2 < 1e-5 and pad_w == 0.0 and losses[-1] < 0.25 * base
    print(f"SELFTEST {'OK' if ok else 'FAILED'} | device {dev}")
    print(f"  torch vs numpy forward: max diff {d1:.1e} (want < 1e-4)")
    print(f"  bag alone vs padded batch: diff {d2:.1e} (want < 1e-5) | max weight on padded slots {pad_w} (want 0)")
    print(f"  learning: loss {losses[0]:.4f} -> {losses[-1]:.4f} vs mean-predictor {base:.4f} (want < 25% of it)")
    print(f"  attention on distress frames: {key_w:.2f}x the uniform weight (informational, not a pass/fail check)")
    if not ok:
        sys.exit(1)


# --------------------------------------------------------------------------------------------- V8
def v8(kind):
    import torch
    rules = json.loads(RULES.read_text())
    trim_dir = ROOT / f"data/modeling/trim{rules['end_trim_ahead_m']:g}"
    if not (trim_dir / "segments.csv").exists():
        stop(f"missing {trim_dir.relative_to(ROOT)}: run build_segment_table.py first")
    kind = kind or {"concat_cls_patchmean": "concat"}.get(rules["embedding"]["primary_vector"], "concat")
    bags = load_bags(trim_dir, kind)
    dev = device()
    sizes = [len(b[1]) for b in bags]
    print(f"V8: attention-MIL on {len(bags)} primary bags ({trim_dir.relative_to(ROOT)}), instance = frame, "
          f"vector = {kind} ({bags[0][1].shape[1]} dims), frames per bag {min(sizes)}-{max(sizes)}, device {dev}")
    print(f"hyperparameters (fixed, untuned): {HP}")
    mu, sd = standardizer(bags)
    model = make_model(bags[0][1].shape[1])
    t = time.time()
    losses = train(model, bags, mu, sd, dev, log_every=25)
    secs = time.time() - t
    first5, last5 = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
    finite = bool(np.isfinite(losses).all())
    passed = finite and last5 < first5
    print(f"\nV8: training loss (mean of first 5 epochs) {first5:.5f} -> (last 5) {last5:.5f} | "
          f"{len(losses)} epochs in {secs:.1f} s on {dev} -> {'PASS' if passed else 'FAIL'} "
          f"(pre-registered: loss decreases, no errors)")
    print("note: a training loss only shows the model can fit; it says nothing about accuracy on unseen streets.")
    out = ROOT / "data/modeling/v8_loss.csv"
    pd.DataFrame(dict(epoch=np.arange(1, len(losses) + 1), train_loss=losses)).to_csv(out, index=False)
    print(f"saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="gated attention-MIL (rung 5) + V8 check")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--v8", action="store_true")
    ap.add_argument("--vector", choices=["concat", "cls", "patch_mean"], help="default: the frozen primary")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.v8:
        v8(a.vector)
    else:
        ap.error("choose --selftest or --v8")
