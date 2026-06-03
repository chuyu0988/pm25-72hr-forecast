"""
Four SIMPLE (non-DeepONet) baselines  —  lower-bound comparison
================================================================
IDENTICAL test set to the model / DeepONet baselines: windows require all 5
input vars present (raw-primary, FPCA fills gaps) + station coords -> 71 stations,
23,189 test windows. Predictions use PM2.5 only; pure-raw test target; NCU RMSE+MAE.

  1. Persistence        : y_hat(h) = last observed input PM2.5 (input hour 23), held 72h.
  2. Diurnal persistence: y_hat(h) = input PM2.5 at hour (h mod 24)  [windows start 00:00].
  3. Climatology        : training mean PM2.5 by (station, month, hour-of-day).
  4. Ridge regression   : linear map PM2.5x24 -> 72h, fit jointly on train.

Reference: my model 7.3250 / DeepONet baseline ~7.83 / SOTA 6.88.
"""

import pandas as pd
import numpy as np
from functools import reduce
import sys
sys.stdout.reconfigure(line_buffering=True)

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
    df = pd.read_csv(path); cols = [c for c in df.columns if c not in META]; df = df[cols]
    df['PublishTime'] = pd.to_datetime(df['PublishTime'])
    return df.melt(id_vars=['PublishTime'], var_name='Station', value_name=value_name)

print("Loading FPCA + RAW for all 5 vars (to match the model's window criterion)...")
fpca_melts = [load_melt(fpca_files[v], v) for v in VARS]
df_all = reduce(lambda l, r: pd.merge(l, r, on=['PublishTime', 'Station'], how='inner'), fpca_melts)
for v in VARS:
    df_all = pd.merge(df_all, load_melt(raw_files[v], v + "_RAW"), on=['PublishTime', 'Station'], how='left')
for v in VARS:
    df_all[v + "_H"] = df_all[v + "_RAW"].fillna(df_all[v])
df_all = df_all.sort_values(by=['Station', 'PublishTime'])

station_info = pd.read_csv(f"{base_path}\\station_info\\station .csv", encoding='utf-8-sig')
station_coords = {r['SITE_NAME']: (r['lat'], r['lon']) for _, r in station_info.iterrows()}

INPUT_HOURS, OUTPUT_HOURS = 24, 72
TOTAL_WINDOW = INPUT_HOURS + OUTPUT_HOURS
STEP_SIZE = 24
SPLIT_DATE = pd.Timestamp('2025-01-01'); TRAIN_START = pd.Timestamp('2018-01-01'); TEST_END = pd.Timestamp('2025-11-30')
H_COLS = [v + "_H" for v in VARS]   # 5 hybrid input cols (order PM25,U,V,RH,TEMP)

# climatology table (train period, station x month x hour) from raw-primary PM2.5
print("Building climatology (train, station x month x hour)...")
tr = df_all[(df_all['PublishTime'] >= TRAIN_START) & (df_all['PublishTime'] < SPLIT_DATE)].copy()
tr['month'] = tr['PublishTime'].dt.month; tr['hour'] = tr['PublishTime'].dt.hour
clim = tr.groupby(['Station', 'month', 'hour'])['PM25_H'].mean()
station_mean = tr.groupby('Station')['PM25_H'].mean(); global_mean = tr['PM25_H'].mean()

print("Building windows (same criterion as model: 5-var input non-NaN + coords)...")
tr_X, tr_Y = [], []
te_X, te_Y, te_station, te_start = [], [], [], []
for station in df_all['Station'].unique():
    if station not in station_coords:
        continue
    s = df_all[df_all['Station'] == station].set_index('PublishTime').sort_index()
    s = s[H_COLS + ['PM25_RAW']].asfreq('h')
    vals = s.values; times = s.index   # cols: 0..4 = H vars (PM25 at 0), 5 = PM25_RAW
    ns = len(vals) - TOTAL_WINDOW + 1
    if ns <= 0:
        continue
    for i in range(0, ns, STEP_SIZE):
        w = vals[i:i+TOTAL_WINDOW]; ct = times[i]
        xin5 = w[:INPUT_HOURS, 0:5]            # 5-var input
        if np.isnan(xin5).any():               # same validity criterion as model
            continue
        pm_in = w[:INPUT_HOURS, 0]             # PM2.5 input channel (24)
        if TRAIN_START <= ct < SPLIT_DATE:
            y = w[INPUT_HOURS:, 0]             # raw-primary PM2.5 target
            if np.isnan(y).any():
                continue
            tr_X.append(pm_in); tr_Y.append(y)
        elif SPLIT_DATE <= ct <= (TEST_END - pd.Timedelta(hours=TOTAL_WINDOW)):
            y = w[INPUT_HOURS:, 5]             # pure raw PM2.5 target
            if np.isnan(y).all():
                continue
            te_X.append(pm_in); te_Y.append(y); te_station.append(station); te_start.append(ct)

tr_X = np.array(tr_X, np.float32); tr_Y = np.array(tr_Y, np.float32)
te_X = np.array(te_X, np.float32); te_Y = np.array(te_Y, np.float64)
te_start = pd.to_datetime(pd.Series(te_start)).reset_index(drop=True)
print(f"Train windows {tr_X.shape} | Test windows {te_X.shape} | Stations {len(set(te_station))}")

mask = ~np.isnan(te_Y)
def report(tag, pred):
    h = np.array([np.sqrt(np.mean((te_Y[:, k][mask[:, k]] - pred[:, k][mask[:, k]])**2))
                  for k in range(72) if mask[:, k].sum() > 0])
    rmse = float(h.mean()); res = pred[mask] - te_Y[mask]; mae = float(np.mean(np.abs(res)))
    seg = {"1-12": (0, 12), "13-24": (12, 24), "25-48": (24, 48), "49-72": (48, 72)}
    segs = " ".join(f"{k}:{h[a:b].mean():.2f}" for k, (a, b) in seg.items())
    print(f"  [{tag:22s}] NCU-RMSE={rmse:.4f} | MAE={mae:.4f} | segs {segs}")
    return rmse, mae

results = {}
print("\nComputing baselines...")
results['Persistence'] = report("1 Persistence", np.repeat(te_X[:, -1:], 72, axis=1))
results['Diurnal persistence'] = report("2 Diurnal persistence", te_X[:, np.arange(72) % 24])

pred = np.empty((len(te_X), 72), np.float64)
months3 = np.stack([(te_start + pd.Timedelta(days=d)).dt.month.values for d in (1, 2, 3)], axis=1)
hours = np.arange(72) % 24; dayoff = np.arange(72) // 24
for i, st in enumerate(te_station):
    smean = station_mean.get(st, global_mean)
    for hh in range(72):
        try:
            v = clim.loc[(st, months3[i, dayoff[hh]], hours[hh])]
            pred[i, hh] = v if not np.isnan(v) else smean
        except KeyError:
            pred[i, hh] = smean
results['Climatology'] = report("3 Climatology", pred)

alpha = 1.0
Xtr = np.concatenate([tr_X, np.ones((len(tr_X), 1), np.float32)], axis=1).astype(np.float64)
A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1]); A[-1, -1] -= alpha
W = np.linalg.solve(A, Xtr.T @ tr_Y.astype(np.float64))
Xte = np.concatenate([te_X, np.ones((len(te_X), 1), np.float32)], axis=1).astype(np.float64)
results['Ridge (PM2.5x24)'] = report("4 Ridge regression", Xte @ W)

print(f"\n{'='*64}\nSUMMARY — simple baselines (my model 7.3250 / DeepONet ~7.83 / SOTA 6.88)\n{'='*64}")
print(f"{'baseline':<24}{'NCU-RMSE':>10}{'MAE':>9}")
for k, (r, a) in results.items():
    print(f"{k:<24}{r:>10.4f}{a:>9.4f}")
print("Done!")
