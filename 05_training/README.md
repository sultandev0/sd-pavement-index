# 05_training: model training

Course section: **Model Training**. Hyperparameter optimization runs inside the same code (nested cross-validation);
see `06_optimization/README.md`.

| File | What it is |
|---|---|
| `mil.py` | **The deep-learning model, built from scratch in PyTorch:** a gated attention multiple-instance learning network (Ilse et al., 2018). One bag = one street segment; instances = the frozen DINOv2 vectors of its dashcam frames (768 numbers each); output = PCI / 100. Layers: Linear 768→128 + ReLU + dropout → gated attention (tanh ⊙ sigmoid, 64 units) → softmax over the bag's frames (padded slots get weight 0) → weighted sum → Linear 128→1. About 115,000 trainable parameters. Loss: mean squared error; optimizer: Adam. `--selftest` checks the forward pass against a numpy reference, the padding, and that the network learns. |
| `ladder.py` | Trains and evaluates the whole model ladder on the same segments and folds: (1) mean predictor, (2a) metadata ridge, (2b) metadata gradient boosting, (3) hand-crafted brightness/texture ridge, (4) ridge on pooled DINOv2 features, (5) attention-MIL, plus an exploratory 50/50 combination of 2a and 4. Nested grouped cross-validation in two schemes (leave-street-out and leave-area-out); all hyperparameters tuned in the inner folds only; paired bootstrap over streets for the pre-registered comparison; H3 predictions for segments repaired after the survey. The pre-registered settings are recorded in the file header. |
| `v6_v7.py` | Viability checks V6 (signal) and V7 (repaired-segment sanity). Rungs 1-4 in `ladder.py` reuse its feature and model code, so `ladder.py --check-v6` can prove the two agree exactly. |
| `mil_training_demo.ipynb` | Training demo: builds the network, runs its self-test, trains it on one leave-street-out fold while recording training and held-out error every epoch, and checks that it reproduces `ladder.py`'s predictions for that fold. |
| `figures/mil_training_curves.png` | The learning curves from the demo. |
| `logs/ladder_run_frozen_mil.txt` | Console output of the main run (rungs 1-5, both schemes, the Gate 2 comparison, H3). |
| `logs/ladder_run_ablations.txt` | Console output of the ablations (end trim 15/37 m, CLS / patch-mean vectors, mean-only pooling). |
| `logs/mil_choices.csv` | Attention-MIL hyperparameters chosen by the inner cross-validation in every outer fold. |
| `logs/bench.json` | CPU timing of one 200-epoch MIL run, used by the pre-registered rule that fixed the number of inner folds. |

## How to run (from the project root, `pavement` environment)

```
python 05_training/mil.py --selftest
python 05_training/ladder.py --selftest
python 05_training/ladder.py --check-v6 data/modeling_4C          # must reproduce V6 exactly
python 05_training/ladder.py --bench                              # CPU timing -> inner folds for MIL
python 05_training/ladder.py --data data/modeling_4C --mil --h3   # main run, about 23 minutes on CPU
python 05_training/ladder.py --data data/modeling_4C --trim 15    # ablations: --trim 15|37, --vector cls|patch_mean, --pooling mean
```

The inputs (segment tables and frame vectors under `data/`) are built by `04_pipeline_design/` and are not in the
repository, because the source frames show license plates and the camera's GPS overlay.
