# 新實驗紀錄 — 目標：最低 RMSE（對齊 SOTA 論文指標）

> 開始日期：2026-06-03
> 目標：在「與 SOTA 論文相同的 RMSE 算法」下，把測試 RMSE 壓到最低（SOTA = 6.88）。

---

## 🔒 固定協定（每個實驗都必須一致，否則不可比較）

### 評估指標
- **RMSE 算法 = 與 SOTA 論文相同 ✅ 已核對**（Lee et al. 2024, Atmospheric Environment 338, 120835）。
  - 論文 Eq.(5) 為標準 pooled RMSE；headline 6.88 = **每個預測小時把所有站 pool 起來算 RMSE_h，再對 72 小時取平均**。
  - 證據：Table 2 ALL=6.88 ≠ 各區等權平均(6.35) → 確認是「跨站 pool」而非「各站 RMSE 再平均」。
  - **與本專案 NCU-style 完全相同**：`for h in 72: RMSE_h=sqrt(mean_pooled((true-pred)²)); RMSE=mean(RMSE_h)`。
- 主要回報：**NCU-style RMSE**（headline）。輔助：MAE、分時段 RMSE（1-12 / 13-24 / 25-48 / 49-72hr）。

### ⚠️ 與論文 6.88 比較的「不對等」警語（解讀時必讀）
論文 CNN-BASE 的 6.88 **不是純觀測**：
- 它把 **WRF-CMAQ 的 72hr PM2.5 物理預報**當核心輸入（Table 1：Predicted PM2.5 ✓）。CMAQ 系統單獨 RMSE=10.48，CNN 結合 CMAQ+觀測才降到 6.88。
- 測試集不同：論文 75 站、2019/10–2021/09；本專案 71 站、2025。
- 論文丟棄缺值樣本（測試丟 8.62%）、輸入做 min-max[0,1] 正規化。
→ 本專案是**純觀測**達到 7.3250，逼近「用了 CMAQ」的 6.88。算法可比，但資訊條件不對等，0.45 差距有一部分來自 CMAQ 資訊優勢，非單純模型優劣。**「純觀測、不需數值模式」本身是賣點。**

### 資料協定（不可動）
- **輸入**：用**真實觀測值**，**空值才用 FPCA 補**（`raw.fillna(fpca)`）。
  - 輸入變數可自由選擇：**只用 PM2.5** 或 **全部 5 變數（PM2.5/WIND_U/WIND_V/RH/TEMP）皆可**。
  - 可額外加工程特徵（鄰站 PM2.5、風向、temporal…），但底層觀測一律 raw-primary。
- **訓練目標**：raw 為主、缺值用 FPCA 補。
- **測試目標**：**純 raw 觀測**（不補 FPCA，缺值不計分）。誠實評估。
- 訓練期 2018-01-01 ~ 2024-12-31；測試期 2025-01-01 ~ 2025-11-30。
- 71 站 joint training；滑動窗口 24→72，STEP=24（除非實驗本身就是要改 STEP）。

### 訓練設定（沿用已驗證最佳，除非實驗目的就是改它）
- AdamW(lr=1e-3, weight_decay=1e-2)、CosineAnnealingLR(T_max=2000, eta_min=1e-5)、MSE、**full-batch、2000 epochs、seed=42**。
- 模型基準：FEDONet + M5 station embedding(32d) + temporal(month/weekday 4d) + m=64。

### 紀律
- **一次只改一個變因**，保留乾淨對照。
- 改善若在 ±0.02 內視為雜訊，不採用。
- 每個實驗存：腳本路徑、RMSE/MAE、分時段、與 baseline 的差、結論。

---

## 📊 結果表

| # | 實驗 | 腳本 | 輸入 | 變因 | NCU-RMSE | MAE | Δvs baseline | 結論 |
|---|---|---|---|---|---|---|---|---|
| 0 | **Baseline（正式最佳）** | `code/models/exp_raw_input.py` | raw 5var | — | **7.3250** | **5.2707** | — | 方法論正確的基準線；分時段 5.89/6.98/7.62/7.92；輸入98%+raw、目標僅1.30%用FPCA補 |
| 2 | min-max 正規化[0,1] | `code/models/exp_minmax_norm.py` | raw 5var | 輸入 min-max(train-fit) | 7.5505 | 5.6197 | **+0.2255** | ❌ 變差。原始資料有壞值(PM2.5負、RH>100、TEMP-135)撐大範圍、壓縮真實訊號 |
| 3 | 經緯度平移到原點(SW=0,0) | `code/models/exp_coord_origin.py` | raw 5var | lat-=21.96, lon-=120.20 | 7.3234 | 5.3259 | −0.0016 | ➖ 雜訊範圍內(±0.02)，無實質改善但無害、較乾淨；train loss 略降 |

### 📉 簡單 baselines（非神經網路下界，同一 71 站 / 23,189 筆測試集）
> PM2.5 單變數、純 raw 測試目標、NCU-RMSE + MAE。`exp_simple_baselines.py`

| baseline | 說明 | NCU-RMSE | MAE |
|---|---|---|---|
| Persistence | 重複輸入窗最後一個觀測值 72h | 9.3743 | 6.6497 |
| Diurnal persistence | 複製輸入當天 24h 日週期 ×3 | 9.9319 | 7.0646 |
| Climatology | 訓練期 站×月×時 歷史平均 | 8.3935 | 6.4308 |
| **Ridge 回歸** | PM2.5×24 線性映射→72h | **7.8536** | 5.8169 |

**重要發現**：**Ridge 線性回歸 (7.85) ≈ vanilla DeepONet (7.83)**！在「PM2.5 單變數」這個輸入下，DeepONet 的算子結構幾乎沒有比線性好——這強烈說明你的改善（7.3250）**不是來自 DeepONet 架構本身，而是來自「多變數輸入 + station 條件化」**。這對論文敘事很關鍵。
（另：diurnal persistence 比單純 persistence 還差，因為 72h 尺度下硬複製日週期會累積相位誤差。）

### 🏛️ DeepONet baselines（架構對齊 notebook：Branch Dropout0.2 + AdamW wd1e-2 + 1000ep）
> 單一輸入函數（PM2.5×24h，無座標）、只差 trunk 與訓練方式。`exp_deeponet_baselines_v2.py`
> ⚠️ v1 (`exp_deeponet_baselines.py`/`_per_station.py`) 因拿掉 dropout+wd 且訓練 2000ep 而**過擬合**，
> 其數字（joint 8.13/7.96、per-station 10.38/11.23）**作廢**，以下 v2 為準。

| 方法 | 訓練方式 | trunk | NCU-RMSE | MAE |
|---|---|---|---|---|
| 原始 DeepONet (Lu 2021) | joint | 純座標 MLP | **7.8330** | 5.7489 |
| FEDONet (Sojitra 2025) σ=10 | joint | random Fourier(固定) | **7.8107** | 5.7414 |
| 原始 DeepONet | per-station (71模型) | 純座標 MLP | 8.0520 | 5.8232 |
| FEDONet σ=10 | per-station (71模型) | random Fourier(固定) | 8.0445 | 5.8185 |

**觀察**：
- **joint 僅略優於 per-station**（7.83 vs 8.05，≈0.22）——per-station 在正確正則化下可用，沒有災難性過擬合。
- **Fourier trunk 在此 setup 幫助很小**（7.81 vs 7.83，雜訊內）；資料/正則化才是主導。
- per-station 各站 RMSE 分布：mean 8.03, median 8.29, min 4.63, max 10.16。

**消融階梯（修正後）**：DeepONet/FEDONet baseline ≈ **7.81–7.83** → **+多變數+station embed+temporal（我的）7.3250**（−0.5）→ SOTA(CNN+CMAQ) 6.88。
我的模型在純觀測下仍明顯優於 published DeepONet baseline 約 0.5。

### 📐 經典 FDA baselines：FLM / FAM / GPR（function-on-function，同測試集）
> 依 operator-learning 投影片公式實作。PM2.5 單一輸入函數(24h)→72h；`exp_fda_baselines.py`

| 方法 | 公式 | NCU-RMSE | MAE |
|---|---|---|---|
| **FLM** | A(x)(t)=α(t)+∫β(s,t)x(s)ds（penalized 線性） | 7.8536 | 5.8169 |
| **FAM** | Y(t)=∫F(x(s),s,t)ds，F=ΣB_X(x)B_S(s)θφ_k(t)（RBF基底） | 7.8832 | 5.8248 |
| **GPR (FoFGPR)** | FPCA分數→SE kernel GPR(式6,MLE)→式(5)重建；GP用M=3000子集 | 8.1837 | 5.9886 |

註：
- **FLM (7.8536) = Ridge (7.8536) 完全相同** → 一致性驗證：FLM 本質就是離散化的線性 function-on-function 回歸。
- FAM 的非線性（僅 PM2.5）幾乎沒幫助（7.88，略差）。
- GPR 受限於「只能用 3000 子集 + 10 分量分數迴歸」(n×n 無法用全部 17 萬)；FPCA 10 分量截斷下界僅 3.29，所以瓶頸在 GP 分數預測、非重建基底。

**綜合**：所有「PM2.5 單變數」的方法——線性(Ridge/FLM)、非線性FDA(FAM)、GP(FoFGPR)、DeepONet/FEDONet——全部卡在 **7.81–8.18**。你的模型 **7.3250** 靠「多變數輸入 + station 條件化」突破此高原（−0.5）。

### 分時段 RMSE 參考（baseline，FPCA 輸入版本，僅供量級參考）
| 區間 | RMSE |
|---|---|
| 1-12hr | 5.95 |
| 13-24hr | 7.05 |
| 25-48hr | 7.73 |
| 49-72hr | 8.06 |

---

## 🧭 候選實驗（依殘差分析的潛力排序）

殘差分析結論：誤差集中在「冬季區域傳輸事件 × 西部高污染站」，且系統性**低估高污染尖峰**。

1. **鄰站 PM2.5 輸入**（k=3 最近鄰，+72 維 raw）— 直接打中冬季暗帶，預期 −0.1~−0.3 ⭐最優先
2. **重疊窗口 STEP=3**（增訓練資料、補高污染稀少樣本）— 預期 −0.1~−0.2
3. **高污染加權 / log-target**（解決尖峰低估的系統偏差）— 預期 −0.05~−0.15
4. 風向工程（china_wind 分量）— 與 1 部分重疊，預期 −0.05~−0.1
5. 逐站偏差校正 — 收尾用

> 註：最差的局地工業站（二林、麥寮）bias≈0 屬高變異局地排放，純觀測接近不可約下界，不要當成模型缺陷。

---

## ⚠️ 待辦
- [ ] 取得 SOTA 論文（放 `paper/`），核對 RMSE 精確定義與比較條件（資料/時長/區域/站數）。
- [ ] 確認指標後，必要時重算 baseline 並更新本表。
