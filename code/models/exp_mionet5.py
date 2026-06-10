"""
MIONet-5: one branch net per variable, combined by Hadamard product
====================================================================
MIONet (Jin, Meng, Lu 2022): multiple branch nets, one per input function,
fused via element-wise (Hadamard) product with the trunk, then summed.

  G(v1..v5)(y) = sum_p [ b_pm25(v1) ⊙ b_windu(v2) ⊙ b_windv(v3)
                          ⊙ b_rh(v4) ⊙ b_temp(v5) ⊙ b_ctx(c) ⊙ f(y) ]

Each variable branch takes its 24-h series (24-dim).
A context branch takes [lat, lon, station_embed(32), month sin/cos(2)] = 36-dim.
Trunk = Fourier time encoding (same as exp_best), m=64.
No bias (consistent with current convention). seed=42, full-batch, 2000 epochs.

Reference: exp_best (concat single branch) = NCU-RMSE 7.2931 / MAE 5.2463.
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
fpca_files = {v: f"{base_path}\\fpca_processed\\{n}_FPCA_2025.csv"
              for v, n in [("PM25","PM2.5"), ("WIND_U","WIND_U"), ("WIND_V","WIND_V"),
                           ("RH","RH"), ("TEMP","AMB_TEMP")]}
raw_files = {v: f"{base_path}\\raw\\{n}.csv"
             for v, n in [("PM25","PM2.5"), ("WIND_U","WIND_U"), ("WIND_V","WIND_V"),
                          ("RH","RH"), ("TEMP","AMB_TEMP")]}
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

print("Loading RAW features and merging...")
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
        m = ts.month
        feats.append([np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)])
    return np.array(feats, dtype=np.float32)

valid_stations = sorted([s for s in station_datasets if s in station_coords])
station_to_id = {s: i for i, s in enumerate(valid_stations)}
N_STATIONS = len(valid_stations)
NUM_VARS = 5

# Build per-variable arrays (N, 24) for each var, plus coords/temporal/sid
all_var = [[] for _ in range(NUM_VARS)]      # train, per var
all_y, all_sid, all_coord, all_temp = [], [], [], []
all_var_te = [[] for _ in range(NUM_VARS)]
all_y_te, all_sid_te, all_coord_te, all_temp_te, all_station_te = [], [], [], [], []

for sname in valid_stations:
    ds = station_datasets[sname]
    lat, lon = station_coords[sname]
    sid = station_to_id[sname]
    tx, tex = ds['train_x'][:, :, :NUM_VARS], ds['test_x'][:, :, :NUM_VARS]  # (N,24,5)
    nt, ne = tx.shape[0], tex.shape[0]
    for vi in range(NUM_VARS):
        all_var[vi].append(tx[:, :, vi])       # (nt, 24)
        all_var_te[vi].append(tex[:, :, vi])
    all_y.append(ds['train_y']); all_y_te.append(ds['test_y_raw'])
    all_sid.append(np.full(nt, sid, dtype=np.int64)); all_sid_te.append(np.full(ne, sid, dtype=np.int64))
    all_coord.append(np.full((nt, 2), [lat, lon], dtype=np.float32))
    all_coord_te.append(np.full((ne, 2), [lat, lon], dtype=np.float32))
    all_temp.append(extract_temporal(ds['train_t'])); all_temp_te.append(extract_temporal(ds['test_t']))
    all_station_te.extend([sname] * ne)

def cat_t(arrs, dtype=torch.float32):
    return torch.tensor(np.concatenate(arrs).astype(np.float32 if dtype==torch.float32 else np.int64)).to(device)

var_t = [cat_t(all_var[vi]) for vi in range(NUM_VARS)]            # list of (N,24)
y_t = cat_t(all_y)
sid_t = torch.tensor(np.concatenate(all_sid)).to(device)
coord_t = cat_t(all_coord)
temp_t = cat_t(all_temp)
var_te_t = [cat_t(all_var_te[vi]) for vi in range(NUM_VARS)]
y_te_raw = np.concatenate(all_y_te)
sid_te_t = torch.tensor(np.concatenate(all_sid_te)).to(device)
coord_te_t = cat_t(all_coord_te)
temp_te_t = cat_t(all_temp_te)
station_te = np.array(all_station_te)

print(f"Train: {var_t[0].shape[0]} | Test: {var_te_t[0].shape[0]} | Stations: {N_STATIONS}")

# ============================================================
# MIONet-5 model
# ============================================================
class MIONet5(nn.Module):
    def __init__(self, n_stations, embed_dim=32, p=128, m=64):
        super().__init__()
        self.station_embed = nn.Embedding(n_stations, embed_dim)
        nn.init.normal_(self.station_embed.weight, mean=0.0, std=1.0)

        # one branch per variable: 24 -> p
        def make_branch(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, p),
            )
        self.branches = nn.ModuleList([make_branch(24) for _ in range(5)])

        # context branch: lat/lon(2) + embed(32) + month(2) = 36 -> p
        self.ctx_branch = nn.Sequential(
            nn.Linear(2 + embed_dim + 2, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, p),
        )

        self.freqs = nn.Parameter(torch.empty(m).uniform_(0.0, 2.0))
        trunk_in_dim = 2 * m + 1
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in_dim, 128), nn.ReLU(),
            nn.Linear(128, p),
        )
        # no bias

    def _encode_time(self):
        t = torch.arange(1, 73, dtype=torch.float32, device=self.freqs.device)
        t_norm = (t / 72).unsqueeze(-1)
        angles = 2 * math.pi * t.unsqueeze(-1) * self.freqs
        return torch.cat([t_norm, torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, var_list, sid, coord, temporal):
        # Hadamard product of all branch outputs -> (N, p)
        h = self.branches[0](var_list[0])
        for vi in range(1, 5):
            h = h * self.branches[vi](var_list[vi])
        ctx_in = torch.cat([coord, self.station_embed(sid), temporal], dim=-1)
        h = h * self.ctx_branch(ctx_in)                      # (N, p)

        t = self.trunk(self._encode_time())                  # (72, p)
        return torch.matmul(h, t.T)                          # (N, 72)

SEED = 42
torch.manual_seed(SEED)
model = MIONet5(N_STATIONS, embed_dim=32, p=128, m=64).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-5)
criterion = nn.MSELoss()

print(f"\n{'='*60}\nTraining MIONet-5 (seed={SEED}, m=64, Hadamard fusion)\n{'='*60}")
t0 = time.time()
model.train()
for epoch in range(2000):
    optimizer.zero_grad()
    preds = model(var_t, sid_t, coord_t, temp_t)
    loss = criterion(preds, y_t)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if (epoch + 1) % 400 == 0:
        print(f"  Epoch {epoch+1}/2000 | Loss: {loss.item():.4f} | Time: {time.time()-t0:.1f}s")

model.eval()
with torch.no_grad():
    pred = model(var_te_t, sid_te_t, coord_te_t, temp_te_t).cpu().numpy()

mask_all = ~np.isnan(y_te_raw)
ncu_hourly = []
for h in range(72):
    mh = mask_all[:, h]
    if mh.sum() > 0:
        ncu_hourly.append(np.sqrt(np.mean((y_te_raw[:, h][mh] - pred[:, h][mh]) ** 2)))
ncu_hourly = np.array(ncu_hourly)
ncu_rmse = float(ncu_hourly.mean())
res = pred[mask_all] - y_te_raw[mask_all]
mae = float(np.mean(np.abs(res)))

seg = {"1-12hr": (0, 12), "13-24hr": (12, 24), "25-48hr": (24, 48), "49-72hr": (48, 72)}
print(f"\n{'='*60}\nRESULT (MIONet-5, Hadamard fusion)\n{'='*60}")
print(f"NCU-RMSE: {ncu_rmse:.4f}")
print(f"MAE:      {mae:.4f}")
print("Segment RMSE:")
for name, (a, b) in seg.items():
    print(f"  {name:9s}: {ncu_hourly[a:b].mean():.4f}")
print(f"\nReference exp_best (concat single branch): NCU-RMSE 7.2931, MAE 5.2463")
print(f"vs exp_best: {ncu_rmse - 7.2931:+.4f}")
print("\nDone!")
