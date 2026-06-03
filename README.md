# pm25-72hr-forecast

DeepONet-based **72-hour PM2.5 forecasting** for Taiwan air-quality monitoring
stations (71 stations, pure observational data — no CMAQ simulation).

This repo contains the **data**, the **FPCA imputation method**, and the
**methodologically-corrected baseline model** for reproducing the results.

---

## 1. Task

| | |
|---|---|
| **Input** | Past **24 hours** × 5 variables (PM2.5, WIND_U, WIND_V, RH, AMB_TEMP) + station lat/lon |
| **Output** | Next **72 hours** of PM2.5 concentration |
| **Train** | 2018-01-01 ~ 2024-12-31 (176,750 windows) |
| **Test** | 2025-01-01 ~ 2025-11-30 (23,189 windows) |
| **Window** | sliding, step = 24h (non-overlapping) |
| **Stations** | 71, jointly trained |
| **Metric** | **NCU-style RMSE** = per-hour pooled RMSE averaged over 72h (comparable to SOTA 6.88) |

---

## 2. Data (`data/`)

CSVs are committed **gzipped** (`*.csv.gz`) to stay well under GitHub's file-size
limits. Decompress before use, e.g. `gunzip -k data/raw/*.csv.gz`.

```
data/
├── raw/                  # original station observations (one CSV per variable)
│   ├── PM2.5.csv.gz
│   ├── WIND_U.csv.gz  WIND_V.csv.gz
│   ├── RH.csv.gz      AMB_TEMP.csv.gz
├── fpca_processed/       # FPCA-reconstructed series (used to fill missing cells)
│   └── *_FPCA_2025.csv.gz
└── station_info/
    └── station .csv      # station coordinates (note: filename has a space)
```

### Missing-value handling: **raw-first, FPCA-fill**
Inputs and target both use **raw observations as primary**; a cell is filled by
the FPCA reconstruction **only when the raw value is missing** (~1.3% of target
cells). We do **not** train on FPCA-smoothed values as the target — that would
mismatch the evaluation (which is on raw observations).

---

## 3. FPCA imputation method (`code/fpca/`)

`pm25_fpca_2018~2024_2025.R` (the same script is used for all 5 variables — only
the input CSV filename changes). Uses R `fdapace::FPCA`.

- **One curve = one day** (`SubjectID = factor(date)`) → models the diurnal cycle.
- **Basis fit on training years only** (`< 2025`); the test year is projected onto
  the train basis via `predict()` → **no train/test leakage**.
- **Window = one day** (`INPUT_HOURS = STEP_SIZE = 24`, from 00:00) → each day is
  reconstructed from its own 24 hours only → **no look-ahead leakage**.
- `FVEthreshold = 0.999` → retains 99.9% of variance → effectively **imputes
  missing points, barely smooths**.

---

## 4. Baseline model (`code/models/exp_raw_input.py`)

FEDONet (Fourier-enhanced DeepONet) with a **station embedding** (the key
ingredient) + temporal features.

- Branch: 122-d input + station-embed(32) + temporal(4) → 512→256→128→p
- Trunk: Fourier features (`m=64`) → 128 → p
- Temporal: `month`, `weekday` as sin/cos (4 dims). The `hour` feature is omitted
  on purpose — with step=24 every window starts at 00:00, so it is constant.
- Train: full-batch, AdamW (lr=1e-3, wd=1e-2), CosineAnnealingLR, 2000 epochs.
- **Single model** (no ensemble).

### Result

| Model | File | NCU-style RMSE | MAE |
|---|---|---|---|
| **Our model** (multi-var + station embed + temporal) | `exp_raw_input.py` | **7.3250** | **5.2707** |
| SOTA reference — CNN-BASE (**uses WRF-CMAQ** forecasts) | Lee et al. 2024 | 6.88 | — |

> ⚠️ The SOTA 6.88 is **not** an apples-to-apples target: that CNN ingests
> WRF-CMAQ physical forecasts as input (the CMAQ system alone is RMSE 10.48), and
> is evaluated on a different period/stations. Our model uses **pure observations**.

---

## 4b. Baselines & comparison (`code/models/`, `NEW_EXPERIMENTS_LOG.md`)

All baselines run on the **identical** 71-station / 23,189-window test set with the
same NCU-style metric, so they are directly comparable.

| Family | Method | Script | NCU-RMSE | MAE |
|---|---|---|---|---|
| trivial | Diurnal persistence | `exp_simple_baselines.py` | 9.93 | 7.06 |
| trivial | Persistence | `exp_simple_baselines.py` | 9.37 | 6.65 |
| statistical | Climatology (station×month×hour) | `exp_simple_baselines.py` | 8.39 | 6.43 |
| GP / FDA | FoFGPR (FPCA scores + SE-kernel GP) | `exp_fda_baselines.py` | 8.18 | 5.99 |
| linear / FDA | Ridge = **FLM** (function-on-function linear) | `exp_simple_baselines.py` / `exp_fda_baselines.py` | 7.85 | 5.82 |
| FDA | FAM (functional additive) | `exp_fda_baselines.py` | 7.88 | 5.82 |
| operator | DeepONet — per-station (71 models) | `exp_deeponet_baselines_v2.py` | 8.05 | 5.82 |
| operator | DeepONet — joint (Lu 2021) | `exp_deeponet_baselines_v2.py` | 7.83 | 5.75 |
| operator | FEDONet — joint (Sojitra 2025, frozen Fourier trunk) | `exp_deeponet_baselines_v2.py` | 7.81 | 5.74 |
| **ours** | **conditioned multi-input FEDONet** | `exp_raw_input.py` | **7.3250** | **5.27** |

**Takeaway.** Every *single-PM2.5-input* method — linear (Ridge/FLM), nonlinear FDA
(FAM), Gaussian process (FoFGPR), and the vanilla/Fourier DeepONets — plateaus at
**7.8–8.2**. The gain to 7.33 comes from **multi-variable input + station
conditioning**, not from the operator architecture itself (FLM ≡ Ridge ≡ DeepONet
≈ 7.85 on single PM2.5 input).

Full details, ablations, residual analysis, and the methodology/leakage audit:
**`NEW_EXPERIMENTS_LOG.md`** (and `EXPERIMENT_SUMMARY.md`).

---

## 5. Reproduce

```bash
gunzip -k data/raw/*.csv.gz data/fpca_processed/*.csv.gz
python code/models/exp_raw_input.py
```

Requires PyTorch (CUDA optional). FPCA preprocessing requires R + `fdapace`.
