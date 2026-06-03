"""
Functional-data-analysis baselines: FLM, FAM, GPR (FoFGPR)
==========================================================
Function-on-function regression baselines per the operator-learning slides.
Input function x(s) = PM2.5 over the past 24h (s=0..23); output Y(t) = PM2.5
over the next 72h (t=1..72). Same protocol/test set as the model:
raw-primary input (FPCA fills gaps), pure-raw test target, 71 stations,
23,189 test windows; report NCU-style RMSE + MAE.

FLM (slide 3):   A(x)(t) = alpha(t) + integral beta(s,t) x(s) ds
   -> discretized penalized linear FoFR; beta smoothed in s (2nd-diff penalty).

FAM (slides 9-12): Y(t) = integral F(x(s),s,t) ds,
   F(x,s,t) = sum_{l,l',k} B_{X,l}(x) B_{S,l'}(s) theta_{l l' k} phi_k(t)
   -> x expanded in an RBF value-basis B_X (captures nonlinearity) x s-basis B_S;
      feature Z_i = sum_s B_X(x_i(s)) (x) B_S(s); ridge-regress Y on Z.

GPR / FoFGPR (slides 14-18): FPCA of Y gives scores zeta_l; regress each zeta_l
   on input X via GPR with squared-exponential kernel (eq 6), hyperparameters by
   MLE (eq, slide 16); predict via eq (5) and reconstruct
   Yhat*(t) = mu_Y(t) + sum_l zeta*_l psi_l(t).  GP fit on a TRAIN SUBSET (M=3000)
   since the n x n covariance cannot use all 176k points (shared kernel across all
   score components -> one Cholesky per hyperparameter; MLE by grid over lengthscale).

Last slide (homework) ignored per instruction.
"""

import pandas as pd
import numpy as np
from functools import reduce
import sys, time
sys.stdout.reconfigure(line_buffering=True)
rng = np.random.default_rng(42)

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

print("Loading FPCA + RAW (5 vars, to match model's window criterion)...")
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
H_COLS = [v + "_H" for v in VARS]

print("Building windows (5-var input non-NaN + coords)...")
tr_X, tr_Y, te_X, te_Y = [], [], [], []
for station in df_all['Station'].unique():
    if station not in station_coords:
        continue
    s = df_all[df_all['Station'] == station].set_index('PublishTime').sort_index()
    s = s[H_COLS + ['PM25_RAW']].asfreq('h'); vals = s.values; times = s.index
    ns = len(vals) - TOTAL_WINDOW + 1
    if ns <= 0:
        continue
    for i in range(0, ns, STEP_SIZE):
        w = vals[i:i+TOTAL_WINDOW]; ct = times[i]
        if np.isnan(w[:INPUT_HOURS, 0:5]).any():
            continue
        pm_in = w[:INPUT_HOURS, 0]
        if TRAIN_START <= ct < SPLIT_DATE:
            y = w[INPUT_HOURS:, 0]
            if np.isnan(y).any():
                continue
            tr_X.append(pm_in); tr_Y.append(y)
        elif SPLIT_DATE <= ct <= (TEST_END - pd.Timedelta(hours=TOTAL_WINDOW)):
            y = w[INPUT_HOURS:, 5]
            if np.isnan(y).all():
                continue
            te_X.append(pm_in); te_Y.append(y)

tr_X = np.array(tr_X, np.float64); tr_Y = np.array(tr_Y, np.float64)
te_X = np.array(te_X, np.float64); te_Y = np.array(te_Y, np.float64)
print(f"Train {tr_X.shape} | Test {te_X.shape}")
mask = ~np.isnan(te_Y)

def report(tag, pred):
    h = np.array([np.sqrt(np.mean((te_Y[:, k][mask[:, k]] - pred[:, k][mask[:, k]])**2))
                  for k in range(72) if mask[:, k].sum() > 0])
    rmse = float(h.mean()); res = pred[mask] - te_Y[mask]; mae = float(np.mean(np.abs(res)))
    seg = {"1-12": (0, 12), "13-24": (12, 24), "25-48": (24, 48), "49-72": (48, 72)}
    segs = " ".join(f"{k}:{h[a:b].mean():.2f}" for k, (a, b) in seg.items())
    print(f"  [{tag:18s}] NCU-RMSE={rmse:.4f} | MAE={mae:.4f} | segs {segs}")
    return rmse, mae

results = {}
muY = tr_Y.mean(0)                       # alpha(t) / mu_Y(t)
mX = tr_X.mean(0)
Xc = tr_X - mX; Xte_c = te_X - mX
Yc = tr_Y - muY

# ===================== FLM =====================
# Y(t) = alpha(t) + sum_s beta(s,t) x(s);  beta smoothed in s (2nd-diff penalty)
print("\nFitting FLM (penalized functional linear)...")
D = np.zeros((INPUT_HOURS - 2, INPUT_HOURS))
for i in range(INPUT_HOURS - 2):
    D[i, i], D[i, i+1], D[i, i+2] = 1.0, -2.0, 1.0
Ps = D.T @ D
lam_s, eps = 10.0, 1e-3
A = Xc.T @ Xc + lam_s * Ps + eps * np.eye(INPUT_HOURS)
B = np.linalg.solve(A, Xc.T @ Yc)        # (24, 72) = beta(s,t)
pred_flm = muY + Xte_c @ B
results['FLM'] = report("FLM", pred_flm)

# ===================== FAM =====================
# F(x,s,t)=sum B_X,l(x) B_S,l'(s) theta phi_k(t);  Y=int F ds
# RBF value-basis over x, RBF basis over s; feature Z_i = sum_s B_X(x_i(s)) ⊗ B_S(s)
print("Fitting FAM (functional additive, RBF x-basis x s-basis)...")
Kx, Ks = 6, 6
xc_centers = np.quantile(tr_X.ravel(), np.linspace(0.05, 0.95, Kx))
xw = np.median(np.diff(xc_centers)) + 1e-6
def BX(v):  # v: (...,) -> (..., Kx)
    return np.exp(-0.5 * ((v[..., None] - xc_centers) / xw) ** 2)
s_grid = np.arange(INPUT_HOURS, dtype=float)
sc_centers = np.linspace(0, INPUT_HOURS - 1, Ks)
sw = (INPUT_HOURS - 1) / (Ks - 1)
Bs = np.exp(-0.5 * ((s_grid[:, None] - sc_centers) / sw) ** 2)   # (24, Ks)
def fam_feat(Xarr):
    bx = BX(Xarr)                                                # (N,24,Kx)
    Z = np.einsum('nsx,sk->nxk', bx, Bs).reshape(Xarr.shape[0], Kx * Ks)  # (N,Kx*Ks)
    return Z
Ztr = fam_feat(tr_X); Zte = fam_feat(te_X)
Zm = Ztr.mean(0); Ztr_c = Ztr - Zm; Zte_c = Zte - Zm
ridge = 1e-1
Theta = np.linalg.solve(Ztr_c.T @ Ztr_c + ridge * np.eye(Kx * Ks), Ztr_c.T @ Yc)  # (Kx*Ks,72)
pred_fam = muY + Zte_c @ Theta
results['FAM'] = report("FAM", pred_fam)

# ===================== GPR (FoFGPR) =====================
# FPCA of Y -> scores; per-score GPR with SE kernel; MLE lengthscale; reconstruct.
print("Fitting GPR / FoFGPR (FPCA scores + SE-kernel GP on M=3000 subset)...")
L = 10
U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
Phi = Vt[:L].T                                   # (72, L) eigenfunctions psi_l
var_expl = (S[:L]**2).sum() / (S**2).sum()
Ztr_sc = Yc @ Phi                                # (N, L) train scores
sc_std = Ztr_sc.std(0); Ztr_scn = Ztr_sc / sc_std  # standardize per component

# FPCA truncation floor (best possible with L comps, using TRUE test scores)
te_Y_fill = np.where(mask, te_Y, muY[None, :])
proj = muY + (te_Y_fill - muY) @ Phi @ Phi.T
report("GPR L-trunc floor", proj)

# standardize input for distance, subsample M
Xstd = tr_X.std(0) + 1e-6
Xtr_n = (tr_X - mX) / Xstd; Xte_n = (te_X - mX) / Xstd
M = 3000
sub = rng.choice(len(Xtr_n), size=min(M, len(Xtr_n)), replace=False)
Xs = Xtr_n[sub]; Zs = Ztr_scn[sub]               # (M,24),(M,L)
# pairwise sq dist within subset
def sqdist(A, Bm):
    return (A**2).sum(1)[:, None] + (Bm**2).sum(1)[None, :] - 2 * A @ Bm.T
Dss = sqdist(Xs, Xs); Dss = np.maximum(Dss, 0)
med = np.median(Dss[Dss > 0])
best = None
for ls in [0.25, 0.5, 1.0, 2.0, 4.0]:
    for s2 in [1e-2, 1e-1, 3e-1]:
        ell2 = ls * med
        Kmm = np.exp(-0.5 * Dss / ell2) + s2 * np.eye(len(Xs))
        try:
            Lc = np.linalg.cholesky(Kmm)
        except np.linalg.LinAlgError:
            continue
        logdet = 2 * np.log(np.diag(Lc)).sum()
        alpha = np.linalg.solve(Kmm, Zs)         # (M,L) shared kernel
        quad = (Zs * alpha).sum()                # sum over comps of zeta^T K^-1 zeta
        ll = -0.5 * quad - 0.5 * L * logdet      # (+ const) summed over L
        if best is None or ll > best[0]:
            best = (ll, ell2, s2, alpha)
_, ell2, s2, alpha = best
print(f"  GPR MLE: ell^2={ell2:.3f} (x median), noise={s2}, var_expl(L={L})={var_expl:.3f}")
Kxm = np.exp(-0.5 * sqdist(Xte_n, Xs) / ell2)    # (Nte, M)
Zte_scn = Kxm @ alpha                            # predicted standardized scores
Zte_sc = Zte_scn * sc_std
pred_gpr = muY + Zte_sc @ Phi.T
results['GPR (FoFGPR)'] = report("GPR (FoFGPR)", pred_gpr)

print(f"\n{'='*64}\nSUMMARY — FDA baselines (my model 7.3250 / DeepONet ~7.83 / SOTA 6.88)\n{'='*64}")
print(f"{'method':<20}{'NCU-RMSE':>10}{'MAE':>9}")
for k, (r, a) in results.items():
    print(f"{k:<20}{r:>10.4f}{a:>9.4f}")
print("Done!")
