# Dashcam vs. Laser Survey: Estimating San Diego's Pavement Condition Index from Dashcam Video

**AAI-590 Capstone, MS in Applied Artificial Intelligence, University of San Diego**
Author: Wael Alhathal

## Overview

The City of San Diego scores every paved street segment with a Pavement Condition Index (PCI), from 0 (failed) to 100 (perfect). This project asks how closely a model that sees only dashcam video can estimate the City's 2023 PCI for the same street segment, and where the two disagree.

## Data

| Dataset (City of San Diego, public domain) | Used for |
|---|---|
| [Streets Repair Line Segments](https://data.sandiego.gov/datasets/gis-streets-repair-segs/) | Segment geometry, 2023 PCI (label), width, length, lanes |
| [Streets Repair Projects](https://data.sandiego.gov/datasets/streets-repair-projects/) | Repair history (pavement age, post-survey repairs) |
| [Community Planning Districts](https://data.sandiego.gov/datasets/community-planning-district-boundaries/) | Planning area per segment |

Data version: downloaded 2026-10-24. Raw data files and dashcam footage are not included in this repository (size and privacy).

## Key findings so far

- 31,460 of 32,745 segments have a 2023 PCI.
- 26.8% were repaired after the survey began, so their scores may predate the repair; these are excluded from training, leaving 23,028 segments.
- Repair history is the strongest metadata predictor of PCI; street width and lane count are nearly unrelated to condition.

## Contents

| File | Description |
|---|---|
| `02_data_cleaning/clean_segments.ipynb` | Audits and cleans the City data and writes `data/processed/segments_clean.csv` |
| `03_eda/eda_data_summary.ipynb` | Exploratory data analysis of the cleaned table (Figures 1–6) |
| `.gitignore` | Keeps data, footage, and course documents out of the repository |

## Running the notebooks

Requires Python 3.12 with pandas, GeoPandas, Matplotlib, NumPy, and Jupyter.

1. Download the City datasets above into `data/raw/`.
2. Run `02_data_cleaning/clean_segments.ipynb`, then `03_eda/eda_data_summary.ipynb`.

*This README will be updated as the project progresses.*
