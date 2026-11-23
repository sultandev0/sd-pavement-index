# 06_optimization: model optimization

Course section: **Model Optimization**. The hyperparameter search runs inside the nested cross-validation in
`05_training/ladder.py`, so it lives with the training code instead of in a separate script. For every outer fold,
each setting is scored by an inner cross-validation on that fold's training data only, grouped the same way as the
outer split (by street, or by planning area). The setting with the lowest pooled inner mean absolute error is refit
on the whole training fold and then predicts the held-out fold.

| Model | Searched in the inner cross-validation | Fixed | In the code |
|---|---|---|---|
| Ridge: metadata (2a), hand-crafted (3), DINOv2 (4) | alpha: 15 log-spaced values from 10^-2 to 10^5 | standardization; median imputation (metadata) | `v6_v7.ALPHAS`, `ladder.tune` |
| Gradient boosting: metadata (2b) | max depth {2, 3} × learning rate {0.03, 0.1} × iterations {100, 300} | min samples per leaf 10, squared-error loss | `ladder.HGB_GRID` |
| Attention-MIL (5) | dropout {0.25, 0.5} × weight decay {1e-4, 1e-3, 1e-2} × epochs {50, 100, 200} | hidden 128, attention 64, Adam lr 1e-3, batch 16, MSE loss, seed 0, CPU | `ladder.MIL_GRID`, `ladder.MIL_EPOCHS`, `ladder.cv_mil` |

The MIL epochs are read at checkpoints of one 200-epoch run per setting and inner fold, so the 50- and 100-epoch
models cost nothing extra (`mil.train`'s `on_epoch` hook).

**Ablations** (robustness checks of the representation, not used to choose the model): end trim 15 / 30 / 37 m;
CLS, patch-mean, or concatenated frame vector; mean-only vs mean + standard deviation pooling
(`ladder.py --trim`, `--vector`, `--pooling`). Attention pooling is the MIL rung itself.

**Where the choices are recorded:** `05_training/logs/mil_choices.csv` (MIL, every outer fold) and the
"chosen (outer folds)" column in `05_training/logs/ladder_run_frozen_mil.txt` (ridge alphas and boosting settings).
