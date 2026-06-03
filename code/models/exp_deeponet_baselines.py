"""
DeepONet BASELINES (faithful to original papers)  —  for comparison vs my model
================================================================================
Two published architectures, trained on the SAME data/task/eval as my model so
they serve as clean DeepONet baselines:

  (A) Original DeepONet         — Lu et al. 2021 (Nat. Mach. Intell.; lululxvi/deeponet)
      unstacked: branch FNN ⊗ trunk FNN, plain coordinate trunk, dot product + b0.
  (B) FEDONet                   — Sojitra, Dhingra, San 2025 (arXiv:2509.12344)
      same skeleton, but trunk input is a FROZEN random Fourier-feature embedding
      phi(zeta) = [sin(2*pi*B*zeta), cos(2*pi*B*zeta)],  B_ij ~ N(0, sigma^2), fixed.

Shared design (paper-faithful + comparable):
  - Branch input = ONE input function = PM2.5 sampled at 24 hourly sensor points
    = 24-dim  [u(t_1),...,u(t_24)].  This is the textbook single-function branch of
    Lu 2021 / FEDONet.  (NO other variables, NO lat/lon, NO station embedding, NO
    temporal — multi-variable + conditioning are MY model's additions, not the baseline.)
  - Station-INDEPENDENT operator: trunk coordinate = forecast hour t in {1..72}
    (shared (72,p) trunk -> memory-light; paper's spatial zeta dropped, per user OK).
  - Output G(u)(t) = sum_k b_k(u) * T_k(t) + b0.
  - Data: raw-primary 5-var input (FPCA fills gaps), raw-primary train target,
    pure-raw test target. Same windows/split as exp_raw_input.py.
  - Training: Adam(lr=1e-3) + CosineAnnealing, MSE, full-batch, 2000 epochs, seed=42.
  - Eval: NCU-style RMSE (= paper metric) + MAE + segment RMSE.

Hyperparameters NOT specified by the papers (documented choices):
  p=128; branch [120,512,256,128,p] ReLU; trunk hidden [128,128] ReLU (out ReLU);
  FEDONet M=128 Fourier feats; sigma swept over {1,5,10} (report best, since the
  paper leaves sigma unspecified and a bad sigma would unfairly cripple the baseline).
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
            train_X.append(x_window); train_Y.append(y_window)
        elif SPLIT_DATE <= current_time <= (TEST_END - pd.Timedelta(hours=TOTAL_WINDOW)):
            y_window = window[INPUT_HOURS:, IDX_PM25_RAW]
            if np.isnan(y_window).all():
                continue
            test_X.append(x_window); test_Y.append(y_window)
    if len(train_X) > 0 and len(test_X) > 0:
        station_datasets[station] = {
            'train_x': np.array(train_X, dtype=np.float32),
            'train_y': np.array(train_Y, dtype=np.float32),
            'test_x': np.array(test_X, dtype=np.float32),
            'test_y_raw': np.array(test_Y),
        }

valid_stations = sorted([s for s in station_datasets if s in station_coords])
IDX_PM25 = 0  # PM2.5 is variable 0 -> single input function
all_train_x, all_train_y = [], []
all_test_x, all_test_y_raw = [], []
for sname in valid_stations:
    ds = station_datasets[sname]
    tx = ds['train_x'][:, :, IDX_PM25:IDX_PM25+1]   # PM2.5 only -> ONE function
    tex = ds['test_x'][:, :, IDX_PM25:IDX_PM25+1]
    nt, ne = tx.shape[0], tex.shape[0]
    all_train_x.append(tx.reshape(nt, -1))       # 24-dim single function, NO coords
    all_train_y.append(ds['train_y'])
    all_test_x.append(tex.reshape(ne, -1))
    all_test_y_raw.append(ds['test_y_raw'])

all_train_x = np.concatenate(all_train_x).astype(np.float32)
all_train_y = np.concatenate(all_train_y).astype(np.float32)
all_test_x = np.concatenate(all_test_x).astype(np.float32)
all_test_y_raw = np.concatenate(all_test_y_raw)

train_x_t = torch.tensor(all_train_x).to(device)
train_y_t = torch.tensor(all_train_y).to(device)
test_x_t = torch.tensor(all_test_x).to(device)
print(f"\nTrain windows: {train_x_t.shape[0]} | Test windows: {test_x_t.shape[0]} | Stations: {len(valid_stations)}")
print(f"Branch input dim: {train_x_t.shape[1]} (PM2.5 single function x 24 sensor hours, no coords)")

mask_all = ~np.isnan(all_test_y_raw)

# ============================================================
# MODELS
# ============================================================
P = 128

class DeepONet_Original(nn.Module):
    """Lu et al. 2021, unstacked DeepONet. Plain coordinate trunk, dot product + b0."""
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
            nn.Linear(128, p), nn.ReLU(),   # activation on trunk output (Lu 2021)
        )
        self.b0 = nn.Parameter(torch.zeros(1))

    def _coords(self):
        t = torch.arange(1, 73, dtype=torch.float32, device=self.b0.device).unsqueeze(-1)
        return t / 72.0   # normalized time in (0,1]

    def forward(self, x):
        b = self.branch(x)                 # (N, p)
        T = self.trunk(self._coords())     # (72, p)
        return torch.matmul(b, T.T) + self.b0

class FEDONet_Paper(nn.Module):
    """Sojitra et al. 2025. Frozen random Fourier-feature trunk embedding."""
    def __init__(self, branch_in=24, p=P, m_fourier=128, sigma=1.0):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(branch_in, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, p),
        )
        # frozen random Fourier embedding on the 1-D time coordinate
        B = torch.randn(m_fourier, 1) * sigma           # B_ij ~ N(0, sigma^2)
        self.register_buffer('B', B)                    # fixed, non-trainable
        self.trunk = nn.Sequential(
            nn.Linear(2 * m_fourier, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, p), nn.ReLU(),
        )
        self.b0 = nn.Parameter(torch.zeros(1))

    def _phi(self):
        t = torch.arange(1, 73, dtype=torch.float32, device=self.b0.device).unsqueeze(-1) / 72.0  # (72,1)
        proj = 2 * math.pi * t @ self.B.T               # (72, m)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (72, 2m)

    def forward(self, x):
        b = self.branch(x)
        T = self.trunk(self._phi())
        return torch.matmul(b, T.T) + self.b0

# ============================================================
# TRAIN / EVAL helper
# ============================================================
def train_eval(model, tag, epochs=2000):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)      # papers: Adam
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = nn.MSELoss()
    print(f"\n{'='*60}\nTraining {tag}\n{'='*60}")
    t0 = time.time()
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = crit(model(train_x_t), train_y_t)
        loss.backward(); opt.step(); sch.step()
        if (ep + 1) % 500 == 0:
            print(f"  Epoch {ep+1}/{epochs} | Loss {loss.item():.4f} | {time.time()-t0:.0f}s")
    model.eval()
    with torch.no_grad():
        pred = model(test_x_t).cpu().numpy()
    ncu_h = np.array([np.sqrt(np.mean((all_test_y_raw[:, h][mask_all[:, h]] - pred[:, h][mask_all[:, h]])**2))
                      for h in range(72) if mask_all[:, h].sum() > 0])
    rmse = float(ncu_h.mean())
    res = pred[mask_all] - all_test_y_raw[mask_all]
    mae = float(np.mean(np.abs(res)))
    seg = {"1-12": (0, 12), "13-24": (12, 24), "25-48": (24, 48), "49-72": (48, 72)}
    segs = {k: float(ncu_h[a:b].mean()) for k, (a, b) in seg.items()}
    print(f"  -> NCU-RMSE={rmse:.4f} | MAE={mae:.4f} | segs " +
          " ".join(f"{k}:{v:.2f}" for k, v in segs.items()))
    return {"tag": tag, "rmse": rmse, "mae": mae, **{f"seg_{k}": v for k, v in segs.items()}}

results = []
torch.manual_seed(42)
results.append(train_eval(DeepONet_Original(), "Original DeepONet (Lu 2021)"))

for sigma in [1.0, 5.0, 10.0]:
    torch.manual_seed(42)
    results.append(train_eval(FEDONet_Paper(sigma=sigma), f"FEDONet (Sojitra 2025), sigma={sigma}"))

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*72}\nSUMMARY — DeepONet baselines (same data/eval; my model = 7.3250)\n{'='*72}")
print(f"{'method':<38}{'NCU-RMSE':>10}{'MAE':>9}")
for r in results:
    print(f"{r['tag']:<38}{r['rmse']:>10.4f}{r['mae']:>9.4f}")
best_fed = min([r for r in results if 'FEDONet' in r['tag']], key=lambda r: r['rmse'])
print(f"\nBest FEDONet: {best_fed['tag']}  RMSE={best_fed['rmse']:.4f}")
print(f"My conditioned model (exp_raw_input.py): 7.3250")
print(f"SOTA NCU CNN-BASE (uses CMAQ): 6.88")

import csv
out_csv = r"C:\Users\user\Desktop\HW-NCHU\meeting\ccproject_deeponet\code\models\deeponet_baselines_results.csv"
with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    for r in results:
        w.writerow(r)
print(f"\nSaved: {out_csv}\nDone!")
