"""
Per-Station Per-Day Residual Analysis  (most-correct model)
===========================================================
Uses the methodologically-correct best model:
  raw-primary inputs (5 vars, FPCA only fills gaps) + raw-primary train target
  + pure-raw test target, M5 station embedding + 4-dim temporal, single model.
  (Identical data/model to exp_raw_input.py -> NCU-style RMSE ~7.325.)

Each test window starts at 00:00 of a day (STEP_SIZE=24), forecasting the next
72 hours. We treat the window-start date as the "forecast issue day", and
aggregate that window's 72-hour residuals into per-(station, day) metrics.

Outputs (folder: daily_station_residuals/):
  station_day_metrics.csv            per (station, issue-day): n, rmse, bias, mae, mean_true
  00_station_day_rmse_heatmap.png    station x day RMSE heatmap (overview)
  01_station_day_bias_heatmap.png    station x day bias  heatmap (diverging)
  02_overall_station_rmse.png        each station's overall test RMSE (this model)
  ts_pageNN.png                      per-station daily residual time-series (small multiples)
  And a printed text analysis with improvement directions.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import time
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from functools import reduce

sys.stdout.reconfigure(line_buffering=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

for fname in ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Noto Sans CJK TC', 'DejaVu Sans']:
    try:
        plt.rcParams['font.sans-serif'] = [fname]
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

OUT_DIR = r"C:\Users\user\Desktop\HW-NCHU\meeting\ccproject_deeponet\code\models\daily_station_residuals"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# DATA LOADING  (FPCA + RAW for all 5 variables) -- same as exp_raw_input.py
# ============================================================
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
    df_all[v + "_H"] = df_all[v + "_RAW"].fillna(df_all[v])  # raw where observed, FPCA fills gaps

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

NUM_VARS = 5
all_train_x, all_train_y, all_train_sid, all_train_temp = [], [], [], []
all_test_x, all_test_y_raw, all_test_station, all_test_sid, all_test_temp, all_test_time = [], [], [], [], [], []

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
    all_test_time.extend(ds['test_t'])

all_train_x = np.concatenate(all_train_x).astype(np.float32)
all_train_y = np.concatenate(all_train_y).astype(np.float32)
all_train_sid = np.concatenate(all_train_sid)
all_train_temp = np.concatenate(all_train_temp).astype(np.float32)
all_test_x = np.concatenate(all_test_x).astype(np.float32)
all_test_y_raw = np.concatenate(all_test_y_raw)
all_test_station = np.array(all_test_station)
all_test_sid = np.concatenate(all_test_sid)
all_test_temp = np.concatenate(all_test_temp).astype(np.float32)
all_test_time = np.array(all_test_time)

train_x_t = torch.tensor(all_train_x).to(device)
train_y_t = torch.tensor(all_train_y).to(device)
train_sid_t = torch.tensor(all_train_sid).to(device)
train_temp_t = torch.tensor(all_train_temp).to(device)
test_x_t = torch.tensor(all_test_x).to(device)
test_sid_t = torch.tensor(all_test_sid).to(device)
test_temp_t = torch.tensor(all_test_temp).to(device)

print(f"Train windows: {train_x_t.shape[0]} | Test windows: {test_x_t.shape[0]} | Stations: {N_STATIONS}")

# ============================================================
# MODEL (M5 station embedding + 4-dim temporal) -- same as exp_raw_input.py
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

SEED, M = 42, 64
torch.manual_seed(SEED)
model = FEDONetM5(N_STATIONS, embed_dim=32, init_std=1.0, p=128, m=M, temporal_dim=TEMPORAL_DIM).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-5)
criterion = nn.MSELoss()

print(f"\n{'='*60}\nTraining single model (seed={SEED}, m={M}, raw input + raw target)\n{'='*60}")
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

# NCU-style RMSE (sanity check vs 7.325)
mask_all = ~np.isnan(all_test_y_raw)
ncu_hourly = [np.sqrt(np.mean((all_test_y_raw[:, h][mask_all[:, h]] - pred[:, h][mask_all[:, h]])**2))
              for h in range(72) if mask_all[:, h].sum() > 0]
ncu_rmse = float(np.mean(ncu_hourly))
print(f"\nNCU-style RMSE (sanity): {ncu_rmse:.4f}")

# ============================================================
# PER-(STATION, ISSUE-DAY) RESIDUAL AGGREGATION
# ============================================================
print("\nAggregating residuals per (station, issue-day)...")
N, H = pred.shape
issue_date = pd.to_datetime(all_test_time).normalize()  # window-start day = forecast issue day

rows = []
for i in range(N):
    yt = all_test_y_raw[i]          # (72,)
    yp = pred[i]                    # (72,)
    m = ~np.isnan(yt)
    if m.sum() == 0:
        continue
    res = yp[m] - yt[m]
    rows.append({
        'station': all_test_station[i],
        'date': issue_date[i],
        'n': int(m.sum()),
        'rmse': float(np.sqrt(np.mean(res**2))),
        'mae': float(np.mean(np.abs(res))),
        'bias': float(np.mean(res)),
        'mean_true': float(np.mean(yt[m])),
        'max_true': float(np.max(yt[m])),
    })
daily = pd.DataFrame(rows)
daily.to_csv(f"{OUT_DIR}/station_day_metrics.csv", index=False, encoding='utf-8-sig')
print(f"  Saved station_day_metrics.csv ({len(daily):,} station-day rows)")

# Per-station overall test RMSE (pool all hours across all days)
per_station = []
for sn in valid_stations:
    idx = np.where(all_test_station == sn)[0]
    yt = all_test_y_raw[idx].ravel()
    yp = pred[idx].ravel()
    m = ~np.isnan(yt)
    if m.sum() == 0:
        continue
    res = yp[m] - yt[m]
    per_station.append({
        'station': sn,
        'rmse': float(np.sqrt(np.mean(res**2))),
        'bias': float(np.mean(res)),
        'mae': float(np.mean(np.abs(res))),
        'n': int(m.sum()),
        'mean_true': float(np.mean(yt[m])),
    })
ps = pd.DataFrame(per_station).sort_values('rmse').reset_index(drop=True)
ps.to_csv(f"{OUT_DIR}/per_station_overall.csv", index=False, encoding='utf-8-sig')

# ============================================================
# PIVOT: station x day matrices
# ============================================================
order = ps.sort_values('rmse', ascending=False)['station'].tolist()  # worst at top
rmse_mat = daily.pivot_table(index='station', columns='date', values='rmse').reindex(order)
bias_mat = daily.pivot_table(index='station', columns='date', values='bias').reindex(order)
dates = rmse_mat.columns
date_idx = pd.to_datetime(dates)
# monthly tick positions
month_ticks = [i for i, d in enumerate(date_idx) if d.day == 1]
month_labels = [date_idx[i].strftime('%Y-%m') for i in month_ticks]

# --- 00: station x day RMSE heatmap ---
fig, ax = plt.subplots(figsize=(20, 16))
im = ax.imshow(rmse_mat.values, aspect='auto', cmap='hot_r', vmin=0, vmax=np.nanpercentile(rmse_mat.values, 98))
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=7)
ax.set_xticks(month_ticks)
ax.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=9)
ax.set_xlabel('Forecast issue day (2025)')
ax.set_ylabel('Station (worst RMSE at top)')
ax.set_title(f'Per-Station Per-Day Forecast RMSE  (raw-input model, NCU-RMSE={ncu_rmse:.3f})')
cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
cbar.set_label('Daily RMSE (72h window)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/00_station_day_rmse_heatmap.png", dpi=110, bbox_inches='tight')
plt.close()
print("  [00] station_day_rmse_heatmap.png")

# --- 01: station x day BIAS heatmap (diverging) ---
fig, ax = plt.subplots(figsize=(20, 16))
vlim = np.nanpercentile(np.abs(bias_mat.values), 98)
norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
im = ax.imshow(bias_mat.values, aspect='auto', cmap='RdBu_r', norm=norm)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=7)
ax.set_xticks(month_ticks)
ax.set_xticklabels(month_labels, rotation=45, ha='right', fontsize=9)
ax.set_xlabel('Forecast issue day (2025)')
ax.set_ylabel('Station')
ax.set_title('Per-Station Per-Day Bias (Pred - True);  red = over-predict, blue = under-predict')
cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
cbar.set_label('Daily mean residual')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_station_day_bias_heatmap.png", dpi=110, bbox_inches='tight')
plt.close()
print("  [01] station_day_bias_heatmap.png")

# --- 02: overall per-station RMSE bar ---
fig, ax = plt.subplots(figsize=(18, 8))
colors = ['green' if r < 6 else ('orange' if r < 8 else 'red') for r in ps['rmse']]
ax.bar(range(len(ps)), ps['rmse'], color=colors, edgecolor='black')
ax.axhline(ps['rmse'].mean(), color='blue', linestyle='--', label=f"mean = {ps['rmse'].mean():.2f}")
ax.set_xticks(range(len(ps)))
ax.set_xticklabels(ps['station'], rotation=90, fontsize=7)
ax.set_ylabel('Overall test RMSE')
ax.set_title('Per-Station Overall Test RMSE (sorted; green<6, orange 6-8, red>8)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_overall_station_rmse.png", dpi=110, bbox_inches='tight')
plt.close()
print("  [02] overall_station_rmse.png")

# --- ts pages: per-station daily residual time series (small multiples) ---
PER_PAGE = 16  # 4x4
stations_by_rmse = ps.sort_values('rmse', ascending=False)['station'].tolist()
n_pages = math.ceil(len(stations_by_rmse) / PER_PAGE)
for pg in range(n_pages):
    chunk = stations_by_rmse[pg*PER_PAGE:(pg+1)*PER_PAGE]
    fig, axes = plt.subplots(4, 4, figsize=(22, 14), sharex=True)
    axes = axes.ravel()
    for k, sn in enumerate(chunk):
        ax = axes[k]
        d = daily[daily['station'] == sn].sort_values('date')
        ax.plot(d['date'], d['rmse'], color='crimson', lw=1.0, label='daily RMSE')
        ax.plot(d['date'], d['bias'], color='steelblue', lw=0.9, label='daily bias')
        ax.axhline(0, color='gray', lw=0.6)
        srow = ps[ps['station'] == sn].iloc[0]
        ax.set_title(f"{sn}  (RMSE={srow['rmse']:.2f}, bias={srow['bias']:+.2f})", fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
    for k in range(len(chunk), len(axes)):
        axes[k].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=10)
    fig.suptitle(f'Per-Station Daily Residuals (page {pg+1}/{n_pages}; worst-first)', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f"{OUT_DIR}/ts_page{pg+1:02d}.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  [ts] ts_page{pg+1:02d}.png ({len(chunk)} stations)")

# ============================================================
# TEXT ANALYSIS
# ============================================================
print(f"\n{'='*60}\nANALYSIS\n{'='*60}")
print(f"Stations: {N_STATIONS} | station-day records: {len(daily):,}")
print(f"Overall NCU-style RMSE: {ncu_rmse:.4f}")

print("\n[Worst 10 stations by overall RMSE]")
for _, r in ps.tail(10).iloc[::-1].iterrows():
    print(f"  {r['station']:<10} RMSE={r['rmse']:.2f}  bias={r['bias']:+.2f}  mean_true={r['mean_true']:.1f}  n={int(r['n']):,}")

print("\n[Best 5 stations]")
for _, r in ps.head(5).iterrows():
    print(f"  {r['station']:<10} RMSE={r['rmse']:.2f}  bias={r['bias']:+.2f}  mean_true={r['mean_true']:.1f}")

# Worst individual station-days
print("\n[Worst 15 station-days by daily RMSE]")
worst_days = daily.sort_values('rmse', ascending=False).head(15)
for _, r in worst_days.iterrows():
    print(f"  {r['date'].date()} {r['station']:<10} RMSE={r['rmse']:.1f}  bias={r['bias']:+.1f}  mean_true={r['mean_true']:.1f}  max_true={r['max_true']:.0f}")

# Monthly aggregate of daily RMSE
daily['month'] = daily['date'].dt.month
mo = daily.groupby('month').agg(rmse=('rmse', 'mean'), bias=('bias', 'mean'),
                                mean_true=('mean_true', 'mean'), n=('rmse', 'count')).reset_index()
print("\n[Daily-RMSE averaged by month]")
for _, r in mo.iterrows():
    print(f"  M{int(r['month']):>2}: meanDailyRMSE={r['rmse']:.2f}  bias={r['bias']:+.2f}  mean_true={r['mean_true']:.1f}  station-days={int(r['n'])}")

# Bias vs pollution level (does the model under-predict high days?)
daily['true_bin'] = pd.cut(daily['mean_true'], [0, 10, 20, 30, 50, 300],
                           labels=['0-10', '10-20', '20-30', '30-50', '50+'])
lvl = daily.groupby('true_bin', observed=True).agg(rmse=('rmse', 'mean'), bias=('bias', 'mean'),
                                                   n=('rmse', 'count')).reset_index()
print("\n[Daily RMSE/bias by that day's mean PM2.5 level]")
for _, r in lvl.iterrows():
    print(f"  {str(r['true_bin']):<7} meanDailyRMSE={r['rmse']:.2f}  bias={r['bias']:+.2f}  station-days={int(r['n'])}")

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Done!")
