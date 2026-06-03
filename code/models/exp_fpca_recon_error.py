"""
FPCA reconstruction error per variable
======================================
At every cell where the RAW observation exists, compare it against the FPCA
reconstruction. Reports RMSE / MAE / correlation / normalized-RMSE (RMSE / std)
for each of the 5 variables, split into:
  - overall
  - train period (<2025; FPCA basis is fit here)
  - test  period (2025; projected onto the train basis via predict())

This quantifies the imputation quality / "reconstruction floor". Only measurable
where raw is observed (no ground truth on the FPCA-filled missing cells).
"""

import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(line_buffering=True)

base = r"C:\Users\user\Desktop\HW-NCHU\meeting\ccproject_deeponet\data"
VARS = {
    "PM2.5":   ("fpca_processed\\PM2.5_FPCA_2025.csv",   "raw\\PM2.5.csv"),
    "WIND_U":  ("fpca_processed\\WIND_U_FPCA_2025.csv",  "raw\\WIND_U.csv"),
    "WIND_V":  ("fpca_processed\\WIND_V_FPCA_2025.csv",  "raw\\WIND_V.csv"),
    "RH":      ("fpca_processed\\RH_FPCA_2025.csv",      "raw\\RH.csv"),
    "AMB_TEMP":("fpca_processed\\AMB_TEMP_FPCA_2025.csv","raw\\AMB_TEMP.csv"),
}
META = ['date', 'Time', 'year', 'SubjectID']

def melt(path, name):
    df = pd.read_csv(f"{base}\\{path}")
    cols = [c for c in df.columns if c not in META]
    df = df[cols]
    df['PublishTime'] = pd.to_datetime(df['PublishTime'])
    return df.melt(id_vars=['PublishTime'], var_name='Station', value_name=name)

def stats(raw, fp):
    m = ~np.isnan(raw) & ~np.isnan(fp)
    if m.sum() == 0:
        return None
    r, f = raw[m], fp[m]
    rmse = float(np.sqrt(np.mean((f - r) ** 2)))
    mae = float(np.mean(np.abs(f - r)))
    corr = float(np.corrcoef(r, f)[0, 1])
    sd = float(np.std(r))
    return dict(n=int(m.sum()), rmse=rmse, mae=mae, corr=corr, std=sd, nrmse=rmse / sd if sd > 0 else np.nan)

rows = []
for name, (fp_path, raw_path) in VARS.items():
    print(f"Loading {name}...")
    fp = melt(fp_path, "fp"); rw = melt(raw_path, "raw")
    df = pd.merge(fp, rw, on=['PublishTime', 'Station'], how='inner')
    yr = df['PublishTime'].dt.year
    for split, sel in [("overall", slice(None)), ("train(<2025)", (yr < 2025).values), ("test(2025)", (yr >= 2025).values)]:
        d = df if split == "overall" else df[sel]
        s = stats(d["raw"].values.astype(float), d["fp"].values.astype(float))
        if s:
            rows.append((name, split, s))

print(f"\n{'='*86}")
print(f"{'variable':<10}{'split':<14}{'n':>10}{'RMSE':>9}{'MAE':>9}{'corr':>7}{'std':>9}{'nRMSE':>8}")
print('=' * 86)
last = None
for name, split, s in rows:
    nm = name if name != last else ""
    last = name
    print(f"{nm:<10}{split:<14}{s['n']:>10,}{s['rmse']:>9.3f}{s['mae']:>9.3f}{s['corr']:>7.3f}{s['std']:>9.2f}{s['nrmse']:>8.3f}")
print('=' * 86)
print("nRMSE = RMSE / std(raw)  (scale-free; lower = better reconstruction)")
print("Note: measurable only on observed cells; FPCA-filled missing cells have no ground truth.")
print("Done!")
