# Dashcam vs. Laser Survey: Estimating San Diego's Pavement Condition Index from Dashcam Video
 
**AAI-590 Capstone, MS in Applied Artificial Intelligence, University of San Diego**
Author: Wael Alhathal
 
## Overview
 
The City of San Diego scores every paved street segment with a Pavement Condition Index (PCI), from 0 (failed) to 100 (perfect). This project asks how closely a model that sees only dashcam video can estimate the City's 2023 PCI for the same street segment, and where the two disagree.
 
**Where to find the model:** the deep-learning model is a gated attention multiple-instance learning network built from scratch in PyTorch, in [`05_training/mil.py`](05_training/mil.py); a training demo with learning curves is in [`05_training/mil_training_demo.ipynb`](05_training/mil_training_demo.ipynb). All models are trained and compared in [`05_training/ladder.py`](05_training/ladder.py); hyperparameter optimization is described in [`06_optimization/README.md`](06_optimization/README.md).
 
## Data
 
| Dataset (City of San Diego, public domain) | Used for |
|---|---|
| [Streets Repair Line Segments](https://data.sandiego.gov/datasets/gis-streets-repair-segs/) | Segment geometry, 2023 PCI (label), width, length, lanes |
| [Streets Repair Projects](https://data.sandiego.gov/datasets/streets-repair-projects/) | Repair history (pavement age, post-survey repairs) |
| [Community Planning Districts](https://data.sandiego.gov/datasets/community-planning-district-boundaries/) | Planning area per segment |
| The author's dashcam footage (4K, 30 fps, 1 Hz GPS), five analysis drives | Model input |
 
Data version: downloaded 2026-10-24. Raw data files, dashcam footage, and extracted frames are not included in this repository (size and privacy: the frames show license plates and the camera's GPS overlay).
 
## Pipeline
 
```
dashcam video + GPS -> a frame every 7 m on matched street segments -> SegFormer-B0 road mask
  -> 7 near-field road tiles (252 px, native 4K) -> frozen DINOv2-small vectors -> one vector per frame
  -> per segment: ridge on pooled vectors (rung 4)  |  attention-MIL over the frames (rung 5)
  -> nested grouped cross-validation (leave-street-out, leave-area-out) -> out-of-fold predictions
```
 
## Contents
 
| Folder / file | Description |
|---|---|
| `01_data_collection/extract_gps.py` | Extracts the dashcam's 1 Hz GPS track from every front-camera clip of a drive |
| `01_data_collection/sample_frames.py` | Plans a frame every 7 m on map-matched segments (per-clip clock correction) and extracts them with ffmpeg; spot-check tool for frame alignment |
| `02_data_cleaning/clean_segments.ipynb` | Audits and cleans the City data and writes `data/processed/segments_clean.csv` |
| `02_data_cleaning/alley_flag.ipynb` | Defines and checks the alley rule (alleys are excluded from every analysis set) |
| `02_data_cleaning/gate0_drive.ipynb` | Matches one drive's GPS fixes to City segments (Gate 0) and counts usable segments |
| `03_eda/eda_data_summary.ipynb` | Exploratory data analysis of the cleaned table (Figures 1–6) |
| `04_pipeline_design/road_masks.py` | SegFormer-B0 road masks for every frame |
| `04_pipeline_design/embed_tiles.py` | Cuts the near-field road strip into 7 tiles and embeds each with frozen DINOv2-small; hand-crafted brightness/texture features |
| `04_pipeline_design/build_segment_table.py` | Applies the frozen tile, end-trim, and minimum-frame rules and builds the segment tables and bags |
| `04_pipeline_design/pipeline_rules.json` | The frozen pipeline rules (dated change notes inside) |
| `04_pipeline_design/v3_review.py` | Blind mask-review sheets, strip survey, and survivorship audit |
| `04_pipeline_design/smoke_v1_models.py`, `v1_mask_compare.py`, `v0_blind_test.py`, `v0_check_misses.py` | Pre-registered viability checks (V1, V0) |
| `05_training/` | **Model training:** `mil.py` (the deep-learning model), `ladder.py` (all models, nested cross-validation), `v6_v7.py` (viability checks), the training demo notebook, learning curves, and run logs. See [`05_training/README.md`](05_training/README.md) |
| `06_optimization/README.md` | **Model optimization:** the hyperparameter grids and how they are tuned inside the nested cross-validation |
| `environment.yml`, `setup/fix_libomp.sh` | Conda environment and the OpenMP fix for pip PyTorch alongside conda NumPy |
| `.gitignore` | Keeps data, footage, and course documents out of the repository |
 
## Running the code
 
Requires the `pavement` conda environment (`environment.yml`): Python 3.12, pandas, GeoPandas, scikit-learn, Matplotlib, Jupyter, and (via pip) PyTorch, torchvision, and Hugging Face Transformers. After any pip upgrade of torch, run `bash setup/fix_libomp.sh`.
 
1. Download the City datasets above into `data/raw/`; run `02_data_cleaning/clean_segments.ipynb`, then `03_eda/eda_data_summary.ipynb`.
2. Per drive: `extract_gps.py`, `gate0_drive.ipynb`, `sample_frames.py plan` / `extract`.
3. `road_masks.py` → `embed_tiles.py` → `build_segment_table.py`.
4. Model training and optimization: see [`05_training/README.md`](05_training/README.md).
 
