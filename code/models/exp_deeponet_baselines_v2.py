"""
DeepONet baselines v2 — architecture matched to user's Station-independent notebook
====================================================================================
Fixes the v1 mistake (I had dropped dropout + weight-decay and trained 2000 ep,
which over-fit the tiny per-station data). This version uses the user's REAL setup:

  Branch:  Linear(24,512)-ReLU-Dropout(0.2)-Linear(512,256)-ReLU-Dropout(0.2)-Linear(256,p)
  Trunk(orig):     Linear(1,128)-ReLU-Linear(128,128)-ReLU-Linear(128,p)   (no final act)
  Trunk(FEDONet):  Fourier(frozen random, paper 2509.12344) -> Linear(2m,128)-ReLU-...-Linear(128,p)
  Optimizer: AdamW(lr=1e-3, weight_decay=1e-2) + Cosine(T_max=1000)
  Epochs: 1000 (full-batch), MSE, p=128, trunk t = linspace(0,1,72)

Single input function = PM2.5 x 24h (24-dim), station-independent trunk (time only).
Runs Original DeepONet and FEDONet(frozen sigma=10), each BOTH joint and per-station.
Metric: NCU-style pooled RMSE (= paper metric) + MAE, directly comparable.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math, time, sys
from functools import reduce

sys.stdout.reconfigure(line_buffering=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

base_path = r"C:\Users\user\Desktop\HW-NCHU\meeting\ccproject_deeponet\data"
VARS = ["PM25", "WIND_U", "WIND_V", "RH", "TEMP"]
fpca_files = {"PM25":f"{base_path}\\fpca_processed\\PM2.5_FPCA_2025.csv","WIND_U":f"{base_path}\\fpca_processed\\WIND_U_FPCA_2025.csv","WIND_V":f"{base_path}\\fpca_processed\\WIND_V_FPCA_2025.csv","RH":f"{base_path}\\fpca_processed\\RH_FPCA_2025.csv","TEMP":f"{base_path}\\fpca_processed\\AMB_TEMP_FPCA_2025.csv"}
raw_files = {"PM25":f"{base_path}\\raw\\PM2.5.csv","WIND_U":f"{base_path}\\raw\\WIND_U.csv","WIND_V":f"{base_path}\\raw\\WIND_V.csv","RH":f"{base_path}\\raw\\RH.csv","TEMP":f"{base_path}\\raw\\AMB_TEMP.csv"}
META = ['date','Time','year','SubjectID']

def load_melt(path, value_name):
    df = pd.read_csv(path); cols=[c for c in df.columns if c not in META]; df=df[cols]
    df['PublishTime']=pd.to_datetime(df['PublishTime'])
    return df.melt(id_vars=['PublishTime'], var_name='Station', value_name=value_name)

print("Loading FPCA features...")
fpca_melts=[load_melt(fpca_files[v],v) for v in VARS]
df_fpca=reduce(lambda l,r: pd.merge(l,r,on=['PublishTime','Station'],how='inner'), fpca_melts)
print("Loading RAW and merging...")
df_all=df_fpca
for v in VARS:
    df_all=pd.merge(df_all, load_melt(raw_files[v], v+"_RAW"), on=['PublishTime','Station'], how='left')
for v in VARS:
    df_all[v+"_H"]=df_all[v+"_RAW"].fillna(df_all[v])
df_all=df_all.sort_values(by=['Station','PublishTime'])

station_info=pd.read_csv(f"{base_path}\\station_info\\station .csv", encoding='utf-8-sig')
station_coords={r['SITE_NAME']:(r['lat'],r['lon']) for _,r in station_info.iterrows()}

INPUT_HOURS,OUTPUT_HOURS=24,72
TOTAL_WINDOW=INPUT_HOURS+OUTPUT_HOURS
STEP_SIZE=24
SPLIT_DATE=pd.Timestamp('2025-01-01'); TRAIN_START=pd.Timestamp('2018-01-01'); TEST_END=pd.Timestamp('2025-11-30')
MATRIX_COLS=[v+"_H" for v in VARS]+["PM25_RAW"]; IDX_PM25_H,IDX_PM25_RAW=0,5

print("\nBuilding windows...")
station_datasets={}
for station in df_all['Station'].unique():
    df_s=df_all[df_all['Station']==station].set_index('PublishTime').sort_index()
    df_s=df_s[MATRIX_COLS].asfreq('h'); dv=df_s.values; tm=df_s.index
    trX,trY,teX,teY=[],[],[],[]
    ns=len(dv)-TOTAL_WINDOW+1
    if ns<=0: continue
    for i in range(0,ns,STEP_SIZE):
        w=dv[i:i+TOTAL_WINDOW]; ct=tm[i]; xw=w[:INPUT_HOURS,0:5]
        if np.isnan(xw).any(): continue
        if TRAIN_START<=ct<SPLIT_DATE:
            yw=w[INPUT_HOURS:,IDX_PM25_H]
            if np.isnan(yw).any(): continue
            trX.append(xw[:,0]); trY.append(yw)
        elif SPLIT_DATE<=ct<=(TEST_END-pd.Timedelta(hours=TOTAL_WINDOW)):
            yw=w[INPUT_HOURS:,IDX_PM25_RAW]
            if np.isnan(yw).all(): continue
            teX.append(xw[:,0]); teY.append(yw)
    if len(trX)>0 and len(teX)>0:
        station_datasets[station]={'train_x':np.array(trX,dtype=np.float32),'train_y':np.array(trY,dtype=np.float32),
                                   'test_x':np.array(teX,dtype=np.float32),'test_y_raw':np.array(teY)}

valid=sorted([s for s in station_datasets if s in station_coords])
print(f"Stations: {len(valid)}")

# joint arrays
jtrX=np.concatenate([station_datasets[s]['train_x'] for s in valid]).astype(np.float32)
jtrY=np.concatenate([station_datasets[s]['train_y'] for s in valid]).astype(np.float32)
jteX=np.concatenate([station_datasets[s]['test_x'] for s in valid]).astype(np.float32)
jteY=np.concatenate([station_datasets[s]['test_y_raw'] for s in valid])
print(f"Joint train {jtrX.shape} test {jteX.shape}")

P=128; EPOCHS=1000
trunk_t=torch.linspace(0,1,72).unsqueeze(-1).to(device)

class Branch(nn.Module):
    def __init__(self, din=24, p=P):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(din,512),nn.ReLU(),nn.Dropout(0.2),
                               nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.2),
                               nn.Linear(256,p))
    def forward(self,x): return self.net(x)

class DeepONet_Original(nn.Module):
    def __init__(self,p=P):
        super().__init__()
        self.branch=Branch(24,p)
        self.trunk=nn.Sequential(nn.Linear(1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,p))
        self.bias=nn.Parameter(torch.zeros(1))
    def forward(self,x):
        return torch.matmul(self.branch(x), self.trunk(trunk_t).T)+self.bias

class FEDONet_Frozen(nn.Module):
    def __init__(self,p=P,m=128,sigma=10.0):
        super().__init__()
        self.branch=Branch(24,p)
        self.register_buffer('B', torch.randn(m,1)*sigma)
        self.trunk=nn.Sequential(nn.Linear(2*m,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,p))
        self.bias=nn.Parameter(torch.zeros(1))
    def _phi(self):
        proj=2*math.pi*trunk_t@self.B.T
        return torch.cat([torch.sin(proj),torch.cos(proj)],dim=-1)
    def forward(self,x):
        return torch.matmul(self.branch(x), self.trunk(self._phi()).T)+self.bias

def fit_predict(ctor, xtr, ytr, xte, epochs=EPOCHS):
    torch.manual_seed(42)
    m=ctor().to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-2)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-5)
    crit=nn.MSELoss(); m.train()
    for _ in range(epochs):
        opt.zero_grad(); loss=crit(m(xtr),ytr); loss.backward(); opt.step(); sch.step()
    m.eval()
    with torch.no_grad(): return m(xte).cpu().numpy()

def ncu(pred,true):
    mask=~np.isnan(true)
    h=np.array([np.sqrt(np.mean((true[:,k][mask[:,k]]-pred[:,k][mask[:,k]])**2)) for k in range(72) if mask[:,k].sum()>0])
    res=pred[mask]-true[mask]
    seg={"1-12":(0,12),"13-24":(12,24),"25-48":(24,48),"49-72":(48,72)}
    return float(h.mean()), float(np.mean(np.abs(res))), {k:float(h[a:b].mean()) for k,(a,b) in seg.items()}

def run_joint(ctor,tag):
    t0=time.time()
    xtr=torch.tensor(jtrX).to(device); ytr=torch.tensor(jtrY).to(device); xte=torch.tensor(jteX).to(device)
    pred=fit_predict(ctor,xtr,ytr,xte)
    r,a,s=ncu(pred,jteY)
    print(f"[JOINT {tag}] NCU-RMSE={r:.4f} MAE={a:.4f} segs "+" ".join(f"{k}:{v:.2f}" for k,v in s.items())+f"  ({time.time()-t0:.0f}s)")
    return r,a

def run_perstation(ctor,tag):
    t0=time.time(); preds,trues,psr=[],[],[]
    for k,sn in enumerate(valid):
        ds=station_datasets[sn]
        xtr=torch.tensor(ds['train_x']).to(device); ytr=torch.tensor(ds['train_y']).to(device); xte=torch.tensor(ds['test_x']).to(device)
        p=fit_predict(ctor,xtr,ytr,xte); tr=ds['test_y_raw']
        preds.append(p); trues.append(tr)
        mm=~np.isnan(tr)
        if mm.sum()>0: psr.append(np.sqrt(np.mean((p[mm]-tr[mm])**2)))
        if (k+1)%25==0: print(f"   {tag}: {k+1}/{len(valid)} ({time.time()-t0:.0f}s)")
    pred=np.concatenate(preds); true=np.concatenate(trues)
    r,a,s=ncu(pred,true); psr=np.array(psr)
    print(f"[PER-STATION {tag}] NCU-RMSE={r:.4f} MAE={a:.4f} segs "+" ".join(f"{k}:{v:.2f}" for k,v in s.items()))
    print(f"   per-station RMSE: mean={psr.mean():.3f} median={np.median(psr):.3f} min={psr.min():.3f} max={psr.max():.3f}  ({time.time()-t0:.0f}s)")
    return r,a

print(f"\n{'='*64}\nArchitecture matched to notebook (dropout0.2 + AdamW wd1e-2 + 1000ep)\n{'='*64}")
res={}
res['joint_orig']=run_joint(DeepONet_Original,"Original DeepONet")
res['joint_fed'] =run_joint(FEDONet_Frozen,"FEDONet frozen sig=10")
res['ps_orig']   =run_perstation(DeepONet_Original,"Original DeepONet")
res['ps_fed']    =run_perstation(FEDONet_Frozen,"FEDONet frozen sig=10")

print(f"\n{'='*64}\nSUMMARY (my model=7.3250, SOTA=6.88)\n{'='*64}")
print(f"{'method':<34}{'NCU-RMSE':>10}{'MAE':>9}")
for k,name in [('joint_orig','Original DeepONet (joint)'),('joint_fed','FEDONet (joint)'),
               ('ps_orig','Original DeepONet (per-station)'),('ps_fed','FEDONet (per-station)')]:
    print(f"{name:<34}{res[k][0]:>10.4f}{res[k][1]:>9.4f}")
print("Done!")
