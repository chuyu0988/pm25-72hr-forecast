"""
PER-STATION (independent) DeepONet baselines
============================================================
User's intended "independent station" setup: train ONE separate DeepONet PER
STATION (71 independent models), NOT a single joint model.

Same single-function design as exp_deeponet_baselines.py:
  - branch input = PM2.5 x 24h (24-dim single function), no coords.
  - trunk: (A) plain coordinate MLP = Original DeepONet (Lu 2021)
           (B) frozen random Fourier-feature trunk, sigma=10 = FEDONet (Sojitra 2025)
  - output G(u)(t) = sum_k b_k(u) T_k(t) + b0.
Architecture / optimizer / epochs identical to the joint baseline so the ONLY
difference is per-station vs joint training.
  - Adam(1e-3)+Cosine, MSE, full-batch, 2000 epochs, seed=42 (re-seeded per model).

Each station's test predictions are pooled, then the SAME NCU-style RMSE
(per-hour pooled across all stations, averaged over 72h) + MAE are computed,
so numbers are directly comparable to:
  joint Original DeepONet 8.1327 | joint FEDONet(sig10) 7.9593 | my model 7.3250
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import time
import sys
from functools import reduce

sys.stdout.reconfigure(line_buffering=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

base_path = r"C:\Users\user\Desktop\HW-NCHU\meeting\ccproject_deeponet\data"
VARS = ["PM25", "WIND_U", "WIND_V", "RH", "TEMP"]
fpca_files = {
    "PM25":   f"{base_path}\\fpca_processed\\PM2.5_FPCA_2025.csv",
    "WIND_U": f"{base_path}\\fpca_processed\\WIND_U_FPCA_2025.csv",
    "WIND_V": f"{base_path}\\fpca_processed\\WIND_V_FPCA_2025.csv",
    "RH":     f"{base_path}\\fpca_processed\\RH_FPCA_2025.csv",
    "TEMP":   f"{base_path}\\fpca_processed\\AMB_TEMP_FPCA_2025.csv",
}
raw_files = {
    "PM25":   f"{base_path}\\raw\\PM2.5.csv",
    "WIND_U": f"{base_path}\\raw\\WIND_U.csv",
    "WIND_V": f"{base_path}\\raw\\WIND_V.csv",
    "RH":     f"{base_path}\\raw\\RH.csv",
    "TEMP":   f"{base_path}\\raw\\AMB_TEMP.csv",
}
META = ['date', 'Time', 'year', 'SubjectID']

def load_melt(path, value_name):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c not in META]
    df = df[cols]
    df['PublishTime'] = pd.to_datetime(df['PublishTime'])
    return df.melt(id_vars=['PublishTime'], var_name='Station', value_name=value_name)

print("Loading FPCA features...")
fpca_melts = [load_melt(fpca_files[v], v) for v in VARS]
df_fpca = reduce(lambda l, r: pd.merge(l, r, on=['PublishTime', 'Station'], how='inner'), fpca_melts)
print("Loading RAW features and merging (left)...")
df_all = df_fpca
for v in VARS:
    raw_m = load_melt(raw_files[v], v + "_RAW")
    df_all = pd.merge(df_all, raw_m, on=['PublishTime', 'Station'], how='left')
for v in VARS:
    df_all[v + "_H"] = df_all[v + "_RAW"].fillna(df_all[v])
df_all = df_all.sort_values(by=['Station', 'PublishTime'])

station_info = pd.read_csv(f"{base_path}\\station_info\\station .csv", encoding='utf-8-sig')
station_coords = {row['SITE_NAME']: (row['lat'], row['lon']) for _, row in station_info.iterrows()}

INPUT_HOURS, OUTPUT_HOURS = 24, 72
TOTAL_WINDOW = INPUT_HOURS + OUTPUT_HOURS
STEP_SIZE = 24
SPLIT_DATE = pd.Timestamp('2025-01-01')
TRAIN_START = pd.Timestamp('2018-01-01')
TEST_END = pd.Timestamp('2025-11-30')
MATRIX_COLS = [v + "_H" for v in VARS] + ["PM25_RAW"]
IDX_PM25_H, IDX_PM25_RAW = 0, 5

print("\nBuilding windows...")
station_datasets = {}
for station in df_all['Station'].unique():
    df_s = df_all[df_all['Station'] == station].set_index('PublishTime').sort_index()
    df_s = df_s[MATRIX_COLS].asfreq('h')
    data_values = df_s.values
    times = df_s.index
    train_X, train_Y, test_X, test_Y = [], [], [], []
    num_samples = len(data_values) - TOTAL_WINDOW + 1
    if num_samples <= 0:
        continue
    for i in range(0, num_samples, STEP_SIZE):
        window = data_values[i: i + TOTAL_WINDOW]
        current_time = times[i]
        x_window = window[:INPUT_HOURS, 0:5]
        if np.isnan(x_window).any():
            continue
        if TRAIN_START <= current_time < SPLIT_DATE:
            y_window = window[INPUT_HOURS:, IDX_PM25_H]
            if np.isnan(y_window).any():
                continue
            train_X.append(x_window[:, 0]); train_Y.append(y_window)     # PM2.5 only -> 24-dim
        elif SPLIT_DATE <= current_time <= (TEST_END - pd.Timedelta(hours=TOTAL_WINDOW)):
            y_window = window[INPUT_HOURS:, IDX_PM25_RAW]
            if np.isnan(y_window).all():
                continue
            test_X.append(x_window[:, 0]); test_Y.append(y_window)
    if len(train_X) > 0 and len(test_X) > 0:
        station_datasets[station] = {
            'train_x': np.array(train_X, dtype=np.float32),
            'train_y': np.array(train_Y, dtype=np.float32),
            'test_x': np.array(test_X, dtype=np.float32),
            'test_y_raw': np.array(test_Y),
        }

valid_stations = sorted([s for s in station_datasets if s in station_coords])
print(f"Stations: {len(valid_stations)}")

P = 128

class DeepONet_Original(nn.Module):
    def __init__(self, branch_in=24, p=P):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(branch_in, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, p),
        )
        self.trunk = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, p), nn.ReLU(),
        )
        self.b0 = nn.Parameter(torch.zeros(1))
    def _coords(self):
        t = torch.arange(1, 73, dtype=torch.float32, device=self.b0.device).unsqueeze(-1)
        return t / 72.0
    def forward(self, x):
        return torch.matmul(self.branch(x), self.trunk(self._coords()).T) + self.b0

class FEDONet_Paper(nn.Module):
    def __init__(self, branch_in=24, p=P, m_fourier=128, sigma=10.0):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(branch_in, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, p),
        )
        B = torch.randn(m_fourier, 1) * sigma
        self.register_buffer('B', B)
        self.trunk = nn.Sequential(
            nn.Linear(2 * m_fourier, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, p), nn.ReLU(),
        )
        self.b0 = nn.Parameter(torch.zeros(1))
    def _phi(self):
        t = torch.arange(1, 73, dtype=torch.float32, device=self.b0.device).unsqueeze(-1) / 72.0
        proj = 2 * math.pi * t @ self.B.T
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
    def forward(self, x):
        return torch.matmul(self.branch(x), self.trunk(self._phi()).T) + self.b0

def train_one(model_ctor, xtr, ytr, xte, epochs=2000):
    torch.manual_seed(42)
    model = model_ctor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(xtr), ytr)
        loss.backward(); opt.step(); sch.step()
    model.eval()
    with torch.no_grad():
        return model(xte).cpu().numpy()

def evaluate(tag, pred_all, true_all):
    mask = ~np.isnan(true_all)
    ncu_h = np.array([np.sqrt(np.mean((true_all[:, h][mask[:, h]] - pred_all[:, h][mask[:, h]])**2))
                      for h in range(72) if mask[:, h].sum() > 0])
    rmse = float(ncu_h.mean())
    res = pred_all[mask] - true_all[mask]
    mae = float(np.mean(np.abs(res)))
    seg = {"1-12": (0, 12), "13-24": (12, 24), "25-48": (24, 48), "49-72": (48, 72)}
    segs = " ".join(f"{k}:{ncu_h[a:b].mean():.2f}" for k, (a, b) in seg.items())
    print(f"\n[{tag}] POOLED NCU-RMSE={rmse:.4f} | MAE={mae:.4f} | segs {segs}")
    return rmse, mae

# ============================================================
# Train per-station, pool predictions
# ============================================================
for MODEL_NAME, ctor in [("Original DeepONet (per-station)", DeepONet_Original),
                         ("FEDONet sig=10 (per-station)", FEDONet_Paper)]:
    print(f"\n{'='*64}\n{MODEL_NAME}  — training {len(valid_stations)} independent models\n{'='*64}")
    t0 = time.time()
    pred_list, true_list, per_station_rmse = [], [], []
    for k, sn in enumerate(valid_stations):
        ds = station_datasets[sn]
        xtr = torch.tensor(ds['train_x']).to(device)
        ytr = torch.tensor(ds['train_y']).to(device)
        xte = torch.tensor(ds['test_x']).to(device)
        pred = train_one(ctor, xtr, ytr, xte)
        true = ds['test_y_raw']
        pred_list.append(pred); true_list.append(true)
        m = ~np.isnan(true)
        if m.sum() > 0:
            per_station_rmse.append(np.sqrt(np.mean((pred[m] - true[m])**2)))
        if (k + 1) % 20 == 0:
            print(f"  trained {k+1}/{len(valid_stations)} | {time.time()-t0:.0f}s")
    pred_all = np.concatenate(pred_list)
    true_all = np.concatenate(true_list)
    rmse, mae = evaluate(MODEL_NAME, pred_all, true_all)
    psr = np.array(per_station_rmse)
    print(f"  per-station RMSE: mean={psr.mean():.3f} median={np.median(psr):.3f} "
          f"min={psr.min():.3f} max={psr.max():.3f} | total time {time.time()-t0:.0f}s")

print(f"\n{'='*64}\nCOMPARISON\n{'='*64}")
print("  per-station (above)  vs  joint Original 8.1327 / joint FEDONet 7.9593 / my model 7.3250")
print("Done!")
