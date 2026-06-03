"""
Experiment 3: Shift lat/lon origin to Taiwan's bottom-left corner (0,0)
=======================================================================
ONLY change vs exp_raw_input.py: instead of feeding raw coordinates
(~lat 22-25, ~lon 120-122), shift so the SW corner of the station bounding
box becomes (0,0):
    lat' = lat - lat_min        lon' = lon - lon_min
(min taken over all stations = train min, no leakage; coords are constant.)
This removes the large constant offset that otherwise dominates the first
linear layer, WITHOUT rescaling (degrees preserved). No min-max (dropped).

Everything else identical: raw-primary 5-var input (FPCA fills gaps),
raw-primary train target, pure-raw test target, M5 embed + 4d temporal,
single model seed=42, full-batch, 2000 epochs.
Reports BOTH MAE and NCU-style RMSE (+ segment RMSE).
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

INPUT_HOURS = 24
OUTPUT_HOURS = 72
TOTAL_WINDOW = INPUT_HOURS + OUTPUT_HOURS
STEP_SIZE = 24
SPLIT_DATE = pd.Timestamp('2025-01-01')
TRAIN_START = pd.Timestamp('2018-01-01')
TEST_END = pd.Timestamp('2025-11-30')

MATRIX_COLS = [v + "_H" for v in VARS] + ["PM25_RAW"]
IDX_PM25_H = 0
IDX_PM25_RAW = 5

print("\nBuilding windows...")
station_datasets = {}
for station in df_all['Station'].unique():
    df_s = df_all[df_all['Station'] == station].set_index('PublishTime').sort_index()
    df_s = df_s[MATRIX_COLS].asfreq('h')
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
            y_window = window[INPUT_HOURS:, IDX_PM25_H]
            if np.isnan(y_window).any():
                continue
            train_X.append(x_window); train_Y.append(y_window); train_T.append(current_time)
        elif SPLIT_DATE <= current_time <= (TEST_END - pd.Timedelta(hours=TOTAL_WINDOW)):
            y_window = window[INPUT_HOURS:, IDX_PM25_RAW]
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
valid_stations = sorted([s for s in station_datasets if s in station_coords])
station_to_id = {s: i for i, s in enumerate(valid_stations)}
N_STATIONS = len(valid_stations)

# *** THE ONLY CHANGE: shift coordinate origin to SW corner (0,0) ***
LAT_MIN = min(station_coords[s][0] for s in valid_stations)
LON_MIN = min(station_coords[s][1] for s in valid_stations)
print(f"\n[coord origin shift] SW corner -> (0,0):  LAT_MIN={LAT_MIN:.4f}, LON_MIN={LON_MIN:.4f}")
print(f"  shifted lat range -> [0, {max(station_coords[s][0] for s in valid_stations)-LAT_MIN:.3f}]")
print(f"  shifted lon range -> [0, {max(station_coords[s][1] for s in valid_stations)-LON_MIN:.3f}]")

NUM_VARS = 5
all_train_x, all_train_y, all_train_sid, all_train_temp = [], [], [], []
all_test_x, all_test_y_raw, all_test_station, all_test_sid, all_test_temp = [], [], [], [], []
for sname in valid_stations:
    ds = station_datasets[sname]
    lat, lon = station_coords[sname]
    lat, lon = lat - LAT_MIN, lon - LON_MIN          # <-- shifted coords
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

print(f"\nTrain windows: {train_x_t.shape[0]} | Test windows: {test_x_t.shape[0]} | Stations: {N_STATIONS}")

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

SEED, M = 42, 64
torch.manual_seed(SEED)
model = FEDONetM5(N_STATIONS, embed_dim=32, init_std=1.0, p=128, m=M, temporal_dim=TEMPORAL_DIM).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-5)
criterion = nn.MSELoss()

print(f"\n{'='*60}\nTraining (seed={SEED}, m={M}, coord-origin-shift, raw input + raw target)\n{'='*60}")
t0 = time.time()
model.train()
for epoch in range(2000):
    optimizer.zero_grad()
    loss = criterion(model(train_x_t, train_sid_t, train_temp_t), train_y_t)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if (epoch + 1) % 400 == 0:
        print(f"  Epoch {epoch+1}/2000 | Loss: {loss.item():.4f} | Time: {time.time()-t0:.1f}s")

model.eval()
with torch.no_grad():
    pred = model(test_x_t, test_sid_t, test_temp_t).cpu().numpy()

# EVALUATION — MAE + NCU-style RMSE (+ segment RMSE)
mask_all = ~np.isnan(all_test_y_raw)
ncu_hourly = []
for h in range(72):
    m = mask_all[:, h]
    if m.sum() > 0:
        ncu_hourly.append(np.sqrt(np.mean((all_test_y_raw[:, h][m] - pred[:, h][m]) ** 2)))
ncu_hourly = np.array(ncu_hourly)
ncu_rmse = float(ncu_hourly.mean())
res = pred[mask_all] - all_test_y_raw[mask_all]
mae = float(np.mean(np.abs(res)))
seg = {"1-12hr": (0, 12), "13-24hr": (12, 24), "25-48hr": (24, 48), "49-72hr": (48, 72)}

print(f"\n{'='*60}\nRESULT — Experiment 3 (coord origin shift to SW corner)\n{'='*60}")
print(f"NCU-style RMSE (hourly-pooled, avg over 72h): {ncu_rmse:.4f}")
print(f"Overall MAE:                                  {mae:.4f}")
print(f"  baseline (exp_raw_input.py): RMSE 7.3250")
print(f"  delta vs baseline (RMSE):    {ncu_rmse-7.3250:+.4f}")
print(f"  SOTA target (uses CMAQ):     6.88")
print("Segment RMSE (avg of per-hour pooled RMSE):")
for name, (a, b) in seg.items():
    print(f"  {name:9s}: {ncu_hourly[a:b].mean():.4f}")
print("\nDone!")
