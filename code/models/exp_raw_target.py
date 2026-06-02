"""
Raw-target experiment (honest baseline)
=======================================
Based on the best config from exp_next_round.py (M5 station embedding + Temporal),
but with three corrections requested by the user:

  1) TRAINING TARGET = raw PM2.5 where observed, FPCA only fills missing (空值).
     (Previously the whole target was the FPCA-smoothed signal -> too smooth.)
  2) NO ensemble. Single model (seed 42).
  3) Temporal feature: dropped the broken constant `hour` pair (always 0 because
     every window starts at 00:00 with STEP=24). Keep month + weekday = 4 dims.

Inputs (branch X) stay FPCA-processed, identical for train and test.
Evaluation is unchanged: per-station avg RMSE/MAE on VALID raw values only.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import time
import sys
import gc
from functools import reduce
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.stdout.reconfigure(line_buffering=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# DATA LOADING
# ============================================================
base_path = r"C:\Users\user\Desktop\HW-NCHU\meeting\ccproject_deeponet\data"
file_paths = {
    "PM25":   f"{base_path}\\fpca_processed\\PM2.5_FPCA_2025.csv",
    "WIND_U": f"{base_path}\\fpca_processed\\WIND_U_FPCA_2025.csv",
    "WIND_V": f"{base_path}\\fpca_processed\\WIND_V_FPCA_2025.csv",
    "RH":     f"{base_path}\\fpca_processed\\RH_FPCA_2025.csv",
    "TEMP":   f"{base_path}\\fpca_processed\\AMB_TEMP_FPCA_2025.csv",
}

print("Loading data...")
dfs_features = []
for feature_name, path in file_paths.items():
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c not in ['date', 'Time', 'year', 'SubjectID']]
    df = df[cols]
    df['PublishTime'] = pd.to_datetime(df['PublishTime'])
    df_melt = df.melt(id_vars=['PublishTime'], var_name='Station', value_name=feature_name)
    dfs_features.append(df_melt)

df_merged_features = reduce(
    lambda left, right: pd.merge(left, right, on=['PublishTime', 'Station'], how='inner'),
    dfs_features
)

raw_df = pd.read_csv(f"{base_path}\\raw\\PM2.5.csv")
raw_df['PublishTime'] = pd.to_datetime(raw_df['PublishTime'])
raw_melted = raw_df.melt(id_vars=['PublishTime'], var_name='Station', value_name='PM25_Raw')

df_all = pd.merge(df_merged_features, raw_melted, on=['PublishTime', 'Station'], how='left')
cols_order = ['PublishTime', 'Station', 'PM25', 'WIND_U', 'WIND_V', 'RH', 'TEMP', 'PM25_Raw']
df_all = df_all[cols_order]
df_all = df_all.sort_values(by=['Station', 'PublishTime'])

station_info = pd.read_csv(f"{base_path}\\station_info\\station .csv", encoding='utf-8-sig')
station_coords = {}
for _, row in station_info.iterrows():
    station_coords[row['SITE_NAME']] = (row['lat'], row['lon'])

INPUT_HOURS = 24
OUTPUT_HOURS = 72
TOTAL_WINDOW = INPUT_HOURS + OUTPUT_HOURS
STEP_SIZE = 24
SPLIT_DATE = pd.Timestamp('2025-01-01')
TRAIN_START = pd.Timestamp('2018-01-01')
TEST_END = pd.Timestamp('2025-11-30')

# Column index map inside the per-station value matrix
# ['PM25'(FPCA)=0, WIND_U=1, WIND_V=2, RH=3, TEMP=4, PM25_Raw=5]
IDX_FPCA = 0
IDX_RAW = 5

print("Building windows...")
station_datasets = {}
n_target_raw = 0      # count of target cells taken from raw observation
n_target_fpca = 0     # count of target cells filled by FPCA (missing raw)

for station in df_all['Station'].unique():
    df_s = df_all[df_all['Station'] == station].copy()
    df_s = df_s.set_index('PublishTime').sort_index()
    df_s = df_s[['PM25', 'WIND_U', 'WIND_V', 'RH', 'TEMP', 'PM25_Raw']].asfreq('h')
    data_values = df_s.values
    times = df_s.index

    train_X, train_Y, train_T = [], [], []
    test_X, test_Y, test_T = [], [], []
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
            # --- CORRECTION 1: raw where observed, FPCA fills missing ---
            y_raw = window[INPUT_HOURS:, IDX_RAW]
            y_fpca = window[INPUT_HOURS:, IDX_FPCA]
            mask_missing = np.isnan(y_raw)
            y_window = np.where(mask_missing, y_fpca, y_raw)
            if np.isnan(y_window).any():
                # FPCA should have no gaps; skip defensively if it ever does
                continue
            n_target_raw += int((~mask_missing).sum())
            n_target_fpca += int(mask_missing.sum())
            train_X.append(x_window); train_Y.append(y_window); train_T.append(current_time)
        elif SPLIT_DATE <= current_time <= (TEST_END - pd.Timedelta(hours=TOTAL_WINDOW)):
            y_window = window[INPUT_HOURS:, IDX_RAW]   # test target = raw (unchanged)
            if np.isnan(y_window).all():
                continue
            test_X.append(x_window); test_Y.append(y_window); test_T.append(current_time)

    if len(train_X) > 0 and len(test_X) > 0:
        station_datasets[station] = {
            'train_x': np.array(train_X, dtype=np.float32),
            'train_y': np.array(train_Y, dtype=np.float32),
            'train_t': train_T,
            'test_x': np.array(test_X, dtype=np.float32),
            'test_y_raw': np.array(test_Y),
            'test_t': test_T,
        }

frac_fpca = n_target_fpca / max(1, n_target_raw + n_target_fpca)
print(f"Target cells: raw={n_target_raw}, fpca-filled={n_target_fpca} "
      f"({frac_fpca*100:.2f}% filled by FPCA)")

# ============================================================
# Temporal (CORRECTION 3: month + weekday only, NO hour)
# ============================================================
def extract_temporal(timestamps):
    feats = []
    for ts in timestamps:
        m, wd = ts.month, ts.dayofweek
        feats.append([
            np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12),
            np.sin(2*np.pi*wd/7), np.cos(2*np.pi*wd/7),
        ])
    return np.array(feats, dtype=np.float32)

TEMPORAL_DIM = 4

valid_stations = sorted([s for s in station_datasets.keys() if s in station_coords])
station_to_id = {s: i for i, s in enumerate(valid_stations)}
N_STATIONS = len(valid_stations)

NUM_VARS = 5
all_train_x, all_train_y, all_train_sid, all_train_temp = [], [], [], []
all_test_x, all_test_y_raw, all_test_station, all_test_sid, all_test_temp = [], [], [], [], []

for sname in valid_stations:
    ds = station_datasets[sname]
    lat, lon = station_coords[sname]
    sid = station_to_id[sname]
    tx = ds['train_x'][:, :, :NUM_VARS]
    tex = ds['test_x'][:, :, :NUM_VARS]
    nt, ne = tx.shape[0], tex.shape[0]
    all_train_x.append(np.concatenate([tx.reshape(nt, -1), np.full((nt, 2), [lat, lon])], axis=1))
    all_train_y.append(ds['train_y'])
    all_train_sid.append(np.full(nt, sid, dtype=np.int64))
    all_train_temp.append(extract_temporal(ds['train_t']))
    all_test_x.append(np.concatenate([tex.reshape(ne, -1), np.full((ne, 2), [lat, lon])], axis=1))
    all_test_y_raw.append(ds['test_y_raw'])
    all_test_station.extend([sname] * ne)
    all_test_sid.append(np.full(ne, sid, dtype=np.int64))
    all_test_temp.append(extract_temporal(ds['test_t']))

all_train_x = np.concatenate(all_train_x).astype(np.float32)
all_train_y = np.concatenate(all_train_y).astype(np.float32)
all_train_sid = np.concatenate(all_train_sid)
all_train_temp = np.concatenate(all_train_temp).astype(np.float32)
all_test_x = np.concatenate(all_test_x).astype(np.float32)
all_test_y_raw = np.concatenate(all_test_y_raw)
all_test_station = np.array(all_test_station)
all_test_sid = np.concatenate(all_test_sid)
all_test_temp = np.concatenate(all_test_temp).astype(np.float32)

train_x_t = torch.tensor(all_train_x).to(device)
train_y_t = torch.tensor(all_train_y).to(device)
train_sid_t = torch.tensor(all_train_sid).to(device)
train_temp_t = torch.tensor(all_train_temp).to(device)
test_x_t = torch.tensor(all_test_x).to(device)
test_sid_t = torch.tensor(all_test_sid).to(device)
test_temp_t = torch.tensor(all_test_temp).to(device)

print(f"Train: {train_x_t.shape[0]} | Stations: {N_STATIONS}")
print(f"Shapes: x={train_x_t.shape[1]} temp={train_temp_t.shape[1]}")

# ============================================================
# MODEL (M5 station embedding + 4-dim temporal)
# ============================================================
class FEDONetM5(nn.Module):
    def __init__(self, n_stations, embed_dim=32, init_std=1.0, p=128, m=64, temporal_dim=4):
        super().__init__()
        self.station_embed = nn.Embedding(n_stations, embed_dim)
        nn.init.normal_(self.station_embed.weight, mean=0.0, std=init_std)
        input_dim = 122 + embed_dim + temporal_dim

        self.freqs = nn.Parameter(torch.empty(m).uniform_(0.0, 2.0))
        trunk_in_dim = 2 * m + 1
        self.branch = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, p),
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in_dim, 128), nn.ReLU(),
            nn.Linear(128, p),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def _encode_time(self):
        t = torch.arange(1, 73, dtype=torch.float32, device=self.freqs.device)
        t_norm = (t / 72).unsqueeze(-1)
        angles = 2 * math.pi * t.unsqueeze(-1) * self.freqs
        return torch.cat([t_norm, torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, x, sid, temporal):
        embed = self.station_embed(sid)
        b = self.branch(torch.cat([x, embed, temporal], dim=-1))
        t = self.trunk(self._encode_time())
        return torch.matmul(b, t.T) + self.bias

# ============================================================
# TRAIN single model (CORRECTION 2: no ensemble)
# ============================================================
SEED = 42
M = 64
torch.manual_seed(SEED)
model = FEDONetM5(N_STATIONS, embed_dim=32, init_std=1.0, p=128, m=M, temporal_dim=TEMPORAL_DIM).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-5)
criterion = nn.MSELoss()

print(f"\n{'='*60}\nTraining single model (seed={SEED}, m={M}, raw target)\n{'='*60}")
t0 = time.time()
model.train()
for epoch in range(2000):
    optimizer.zero_grad()
    preds = model(train_x_t, train_sid_t, train_temp_t)
    loss = criterion(preds, train_y_t)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if (epoch + 1) % 200 == 0:
        print(f"  Epoch {epoch+1}/2000 | Loss: {loss.item():.4f} | Time: {time.time()-t0:.1f}s")

model.eval()
with torch.no_grad():
    pred = model(test_x_t, test_sid_t, test_temp_t).cpu().numpy()

# ============================================================
# EVALUATION (unchanged: per-station avg on valid raw cells)
# ============================================================
results = []
for sn in sorted(set(all_test_station)):
    ms = (all_test_station == sn)
    yt, yp = all_test_y_raw[ms], pred[ms]
    mv = ~np.isnan(yt)
    if mv.sum() > 0:
        mae = mean_absolute_error(yt[mv], yp[mv])
        rmse = np.sqrt(mean_squared_error(yt[mv], yp[mv]))
        results.append({'Station': sn, 'MAE': mae, 'RMSE': rmse})
rdf = pd.DataFrame(results)

print(f"\n{'='*60}\nFINAL RESULT (single model, raw target)\n{'='*60}")
print(f"Avg MAE : {rdf['MAE'].mean():.4f}")
print(f"Avg RMSE: {rdf['RMSE'].mean():.4f}")
print(f"(Reference: FPCA-target single best ~7.25-7.30; Ens x5 best 7.2044)")

# Hourly RMSE by block
mask_all = ~np.isnan(all_test_y_raw)
print(f"\nHourly RMSE by block:")
for name, s, e in [("1-12hr", 0, 12), ("13-24hr", 12, 24), ("25-48hr", 24, 48), ("49-72hr", 48, 72)]:
    hrs = []
    for h in range(s, e):
        m = mask_all[:, h]
        if m.sum() > 0:
            hrs.append(np.sqrt(np.mean((all_test_y_raw[:, h][m] - pred[:, h][m]) ** 2)))
    print(f"  {name}: RMSE={np.mean(hrs):.4f}")

# NCU-style hourly-pooled RMSE (for SOTA comparison)
ncu_hourly = []
for h in range(72):
    m = mask_all[:, h]
    if m.sum() > 0:
        ncu_hourly.append(np.sqrt(np.mean((all_test_y_raw[:, h][m] - pred[:, h][m]) ** 2)))
print(f"\nNCU-style RMSE (hourly-pooled, avg over 72h): {np.mean(ncu_hourly):.4f}  (SOTA target 6.88)")

# High-pollution bias diagnostic (does removing smoothing help peaks?)
yt_flat = all_test_y_raw[mask_all]
yp_flat = pred[mask_all]
print(f"\nHigh-pollution bias (pred - true), by true PM2.5 bin:")
for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 50), (50, 75), (75, 1e9)]:
    b = (yt_flat >= lo) & (yt_flat < hi)
    if b.sum() > 0:
        bias = (yp_flat[b] - yt_flat[b]).mean()
        print(f"  [{lo:>3.0f}, {hi if hi < 1e9 else 'inf':>4}): bias={bias:+7.2f}  n={int(b.sum())}")

print("\nDone!")
