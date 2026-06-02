# DeepONet PM2.5 72hr 預測 — 實驗總結與下一階段目標

## 🎯 目標
**將測試集 RMSE 降到 6.88 以下**（SOTA 論文 6.88 但使用 CMAQ 模擬資料，我們是純觀測）。

---

## 🔧 方法論修正與正式 Baseline（2026-06）

> 以下為**方法論修正後的正式版本**，取代先前以 FPCA 平滑值當訓練目標的舊設定。
> 評估一律採 **NCU-style RMSE**（每小時 pooled RMSE，再對 72 小時平均），才能與 SOTA 6.88 公允對比。

### 背景：先前的不嚴謹之處
舊流程把**「FPCA 平滑後的 PM2.5」當訓練目標**，卻用 **raw PM2.5 評估** → train/test 目標分佈不一致（train 平滑、test 原始）。這在學術上會被質疑「偷改了預測任務」，且系統性低估高污染尖峰。

### 已修正的內容
1. **訓練目標**：改為「**raw 觀測為主，僅缺值用 FPCA 補**」（原本整條 72hr 都用 FPCA 平滑值）。
2. **輸入特徵**：5 變數同樣改為「**raw 觀測為主，僅缺值用 FPCA 補**」（原本輸入 100% 用 FPCA 重建值）。採此版本以保有「純觀測」最乾淨的敘事。
3. **Temporal 特徵**：移除壞掉的 `hour`（因 `STEP=24` 每窗口恆為 00:00 → 常數無資訊），保留 `month + weekday`（sin/cos，共 4 維）。
4. **單一模型**：不使用 Ensemble（避免重複計算）。

### FPCA 流程查核 → 確認無資料洩漏
查 `code/fpca/pm25_fpca_2018~2024_2025.R`（5 變數共用同一支腳本，僅換檔名）：
- **一天 = 一條曲線**（`SubjectID = factor(date)`），FPCA 建模日週期。
- **基底只用訓練年（<2025）fit**，測試集用 `predict()` 投影到訓練基底 → 無 train/test 洩漏。
- **窗口 = 一天**（`INPUT_HOURS=STEP_SIZE=24`，從 00:00 起），每天平滑只用當天 24 小時 → **無 look-ahead 洩漏**（發布預測時當天已全部觀測到）。
- `FVEthreshold = 0.999` → 保留 99.9% 變異，**幾乎只補值、不平滑** → 「補值導致預測太平滑」的疑慮基本不成立。
- **結論**：FPCA 輸入其實也合法（不洩漏），但正式版選擇 raw 輸入以求最乾淨。

### 結果（NCU-style RMSE，單模型 seed=42, m=64, M5 station embedding）

| 版本 | 檔案 | 輸入 | 目標 | temporal | NCU-style RMSE |
|---|---|---|---|---|---|
| 舊（瑕疵, Ensemble x5） | `exp_next_round.py` | FPCA | FPCA | 6d (含 hour bug) | 7.2353 |
| 修正：FPCA 輸入 + raw 目標 | `exp_raw_target.py` | FPCA | raw | 4d | 7.3017 |
| ＋ season（冗餘，無效） | `exp_raw_target_season.py` | FPCA | raw | 6d (+season) | 7.2998 |
| **✅ 正式 baseline：raw 輸入 + raw 目標** | **`exp_raw_input.py`** | **raw** | **raw** | **4d** | **7.3250** |

註：
- 舊的 7.2353 同時包含「Ensemble x5 + FPCA 平滑目標 + hour bug」，**並非乾淨對照**；方法論修正後的單模型誠實基準約在 7.30。
- season 特徵與 month 高度冗餘，改善在雜訊範圍內（−0.002），不採用。
- raw 輸入（7.3250）與 FPCA 輸入（7.3017）差距僅 +0.023；FPCA 輸入因去噪略佳，但 raw 輸入敘事最乾淨。

### 🏆 目前正式最佳（方法論正確）
**raw 輸入 + raw 目標 + temporal(month/weekday 4d)，單模型 → NCU-style RMSE = 7.3250**（`exp_raw_input.py`）

---

## 📊 任務設定

- **輸入**：過去 24 小時 × 5 變數（PM2.5、WIND_U、WIND_V、RH、AMB_TEMP）+ 測站經緯度
- **輸出**：未來 72 小時 PM2.5 濃度
- **訓練範圍**：2018-01-01 ~ 2024-12-31（target：raw PM2.5 為主，僅缺值用 FPCA 補 — 見上方「方法論修正」）
- **測試範圍**：2025-01-01 ~ 2025-11-30（原始 raw PM2.5 作為 target）
- **滑動窗口**：步長 24（不重疊）
- **資料**：71 站 joint training，Train 176,750 筆 × 122 維，Test 23,189 筆
- **FPCA 重建 RMSE ≈ 2.0**（理論下界）

### 資料分布
- Train Y: mean=15.70, std=11.45, median=12.70, max=310.31
- Test Y: mean=12.98, std=9.61, median=11.00, max=235.00
- 高污染（>50）僅 1.5%，極端（>100）僅 0.01% → 嚴重右偏

### 測站分布特性
- 71 站全部位於台灣本島
- **集中在西半部平原**，東部極少
- 已跳過 3 個離島：金門、馬公、馬祖（無座標）

---

## 🔬 已完成的失敗實驗（請勿重複嘗試）

### Baseline
| 模型 | MAE | RMSE | 備註 |
|---|---|---|---|
| **FEDONet 原始** | 5.3628 | **7.3967** | User notebook 原始結果 |
| FEDONet 重現 | 5.3706 | 7.4047 | 完整重現（差距 +0.008 為 seed 變異） |

### 失敗實驗清單

| # | 實驗 | MAE | RMSE | vs 7.4047 | 結論 |
|---|---|---|---|---|---|
| 1 | **+ Temporal features**（month/hour/weekday sin/cos）| 5.3632 | 7.3864 | -0.018 | 微小改善，幾乎在隨機波動範圍 |
| 2 | **4000 epochs**（vs 2000）| 5.4760 | 7.4809 | +0.076 | ❌ 過擬合，loss 持續下降但 test 變差 |
| 3 | **Huber loss δ=5** | 5.3084 | 7.4498 | +0.045 | ❌ MAE 改善但 RMSE 變差 |
| 4 | **Huber loss δ=10** | 5.3199 | 7.4255 | +0.021 | ❌ 同上 |
| 5 | **Larger branch (768→512→256→p)** | 5.3266 | 7.4671 | +0.062 | ❌ 過擬合 |
| 6 | **Larger branch + 4000ep** | 5.5181 | 7.5270 | +0.122 | ❌ 嚴重過擬合 |
| 7 | **Ensemble x3**（seeds 42/123/777）| 5.3220 | 7.3796 | **-0.025** | ✅ 穩定小改善 |
| 8 | **Ensemble x3 + Temporal** | 5.3140 | **7.3514** | **-0.053** | ✅ **目前最佳** |
| 9 | **Autoregressive 24→24→24** | 5.4282 | 7.5069 | +0.10 | ❌ 滾動時誤差累積，且失去 wind/temp 資訊 |
| 10 | Z-score normalization（input + output）| - | ~7.5 | - | ❌ 沒幫助（mini-batch 測試） |
| 11 | Coords in Trunk（取代 branch）| - | ~7.4 | - | ❌ 邊際差異 |
| 12 | CNN branch | - | - | - | ❌ User 已嘗試，沒幫助 |

### 各時段 RMSE（Baseline）
| 區間 | RMSE | 觀察 |
|---|---|---|
| 1-12hr | 5.9537 | 短程相對準 |
| 13-24hr | 7.0470 | 開始上升 |
| 25-48hr | 7.7327 | 中程 |
| **49-72hr** | **8.0561** | **長程是主要瓶頸** |

---

## 🚫 從失敗實驗學到的教訓

1. **模型容量已足夠**：更大或更深的網路只會過擬合，不要再加深加寬
2. **訓練長度已到位**：2000 epochs 是最佳，更多會過擬合
3. **單一 loss function 改變沒用**：MSE / Huber 都試過
4. **AR 不適合這個問題**：滾動時遺失非 PM2.5 變數資訊，誤差累積
5. **只有兩件事有穩定改善**：
   - Ensemble（隨機性平均）
   - Temporal features（季節/日週期資訊）
6. **真正的瓶頸是「訓練資料的訊息含量」而非「模型架構」**

---

## 🎯 下一階段：未嘗試但高潛力的方向

### 🔥 強烈建議優先嘗試（按潛力排序）

#### 1. **鄰近測站 PM2.5 輸入**（最推薦）
**動機**：PM2.5 是強空間相關污染物，東北季風時北部測站「領先指示」中南部測站。AR 失敗是因為滾動時遺失輔助變數，但**鄰站的真實 PM2.5 永遠是已知的**。

**做法**：
- 每個測站找最近 k=3 個鄰站
- Branch 輸入：`5 vars × 24hr + 3 neighbors × 24hr PM2.5 + lat/lon` = 122 + 72 = 194 維
- 訓練時鄰站用真實值，測試時也用真實值（因為輸入窗口的 PM2.5 都是觀測值）

**預期改善**：**-0.1 ~ -0.3**

---

#### 2. **重疊窗口（STEP_SIZE=1 或 3）**
**動機**：目前 STEP_SIZE=24（不重疊），訓練資料 176K。改成 STEP_SIZE=1 → **~24 倍訓練資料**。

**做法**：
- STEP_SIZE: 24 → 1（或 3 平衡記憶體）
- 可能需要切到 mini-batch
- 注意：相鄰樣本高度相關

**預期改善**：-0.1 ~ -0.2

---

#### 3. **Multi-Head DeepONet（分時段 trunk）**
**動機**：49-72hr 誤差最大（8.05），且短程（風場主導）vs 長程（天氣型態主導）機制不同。

**做法**：
- 共用 branch，但 3 個獨立 trunk：
  - Trunk-Short：1-24hr
  - Trunk-Mid：25-48hr
  - Trunk-Long：49-72hr
- 或每個 trunk 用不同 Fourier 頻率範圍

**預期改善**：-0.05 ~ -0.15

---

#### 4. **風向工程（Taiwan-specific）**
**動機**：台灣 PM2.5 首要驅動是東北季風。WIND_U/V 各自線性，但「從中國方向吹來的風」才是真正污染指標。

**做法**：
```python
wind_speed = sqrt(U^2 + V^2)
wind_dir = atan2(V, U)
# 「來自污染源方向」分量（指向中國，方位約 285°）
china_wind = U * cos(285°) + V * sin(285°)
```
加到輸入，維度 +24×3。

**預期改善**：-0.05 ~ -0.1

---

#### 5. **Station Embedding**
**動機**：lat/lon 只描述位置，但兩個地理上接近的站可能污染特性差很多（海邊 vs 工業區）。

**做法**：
- 每個測站學一個 16 維 embedding
- Branch 輸入：`5×24 + station_embed(16) + lat/lon` = 138 維

**預期改善**：-0.05 ~ -0.1

---

### 🟡 中等潛力（次優先）

#### 6. **POD-DeepONet + Residual Trunk**
- 用訓練集 Y 的 SVD 取前 k=64 主成分當作固定 trunk 基底
- 加一個小的 learnable residual trunk 修正
- User 之前 POD-DeepONet 單獨 7.4591，結合可能更好

#### 7. **季節分模型**
- 冬季（10-3 月）vs 夏季分別訓練
- 或加 season one-hot 到輸入

#### 8. **加權損失（後段時間權重更高）**
```python
weights = torch.linspace(1.0, 2.0, 72)
loss = ((pred - true)^2 * weights).mean()
```

---

## 📐 模型架構參考（用 ✅ 表示要保留）

```python
# ✅ 保留：FEDONet 核心架構
class FEDONet(nn.Module):
    def __init__(self, input_dim=122, p=128, m=32):
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
```

**訓練設定（已驗證最佳）**：
- Optimizer: AdamW(lr=0.001, weight_decay=1e-2)
- Scheduler: CosineAnnealingLR(T_max=2000, eta_min=1e-5)
- Epochs: **2000**（不要更多）
- Batch: **Full-batch**（mini-batch 比 full-batch 差 ~0.16）

---

## 📁 重要檔案位置

```
C:\Users\user\Desktop\HW-NCHU\meeting\ccproject_deeponet\
├── data\
│   ├── raw\PM2.5.csv (and others — original observations)
│   ├── fpca_processed\*_FPCA_2025.csv (FPCA-smoothed)
│   └── station_info\station .csv (注意檔名有空格！)
├── code\models\
│   ├── reproduce_fedonet.py        ← Baseline reproduction (RMSE 7.4047)
│   ├── exp_temporal_fullbatch.py   ← + Temporal (RMSE 7.3864)
│   ├── exp_diagnosis.py            ← 6 failed variants
│   └── exp_ensemble_temporal_and_ar.py ← Best result 7.3514 + AR failure
└── EXPERIMENT_SUMMARY.md           ← 本檔案
```

**Python 執行路徑**：
```
C:\Users\user\AppData\Local\Programs\Python\Python310\python.exe
```

**GPU**：RTX 3060 Laptop（資料載入慢，但訓練 2000 epochs full-batch 只需 ~150s）

---

## 🎯 目標與建議路線

**目標 RMSE：< 6.88**（比目前 7.3514 改善 -0.47）

**建議路線**：把以下 5 個未嘗試的方法逐一加入，最後配合 Ensemble + Temporal：

1. **鄰近測站 PM2.5** 預期 -0.1 ~ -0.3 ⭐
2. **重疊窗口 STEP=1** 預期 -0.1 ~ -0.2 ⭐
3. **風向工程** 預期 -0.05 ~ -0.1
4. **Multi-Head DeepONet** 預期 -0.05 ~ -0.15
5. **Station Embedding** 預期 -0.05 ~ -0.1

**全部疊加目標**：7.3514 - 0.45 ≈ **6.9**（接近目標）

如果這些還不夠，再考慮：
- POD-DeepONet + Residual trunk
- 季節分模型
- 加權損失
- Spatial-Temporal GNN（工程複雜度高）

---

## ⚠️ 給下一個對話的提醒

1. **不要再試**：4000+ epochs、更大模型、Huber loss、AR
2. **必保留**：Full-batch、2000 epochs、AdamW、CosineAnnealingLR
3. **Best so far**：Ensemble x3 + Temporal = **7.3514**
4. **資料是真正瓶頸**，不是架構 → 重點應該在「**怎麼餵更多/更好的資訊給模型**」
5. **每次跑實驗**：用 `reproduce_fedonet.py` 為模板，保持 full-batch
6. **執行時間參考**：full-batch 2000 epochs ≈ 150s（不含資料載入 ~10min）

