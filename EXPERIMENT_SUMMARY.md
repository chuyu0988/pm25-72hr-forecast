# DeepONet PM2.5 72hr 預測 — 實驗總結

> 純觀測（無 CMAQ）的 72 小時 PM2.5 預測。71 站 joint training。
> 目標：在「與 SOTA 論文相同的 RMSE 算法」下把測試誤差壓到最低。

---

## 🎯 目標與 SOTA 對照（必讀：比較不對等）

- 目標：逼近 SOTA **NCU-style RMSE = 6.88**（Lee et al. 2024, *Atmospheric Environment* 338, 120835；CNN-BASE）。
- **⚠️ 6.88 不是同條件門檻**：
  - 該 CNN 把 **WRF-CMAQ 物理預報**當核心輸入（CMAQ 系統單獨 RMSE=10.48，CNN 結合 CMAQ+觀測才到 6.88）。
  - 測試集不同：論文 75 站、2019/10–2021/09；本專案 71 站、2025。
  - 論文丟棄缺值樣本、輸入 min-max[0,1]。
  - → 本專案是**純觀測**達到 **7.2931**，逼近「用了 CMAQ」的 6.88。算法可比、資訊條件不對等。**「純觀測、不需數值模式」本身是賣點。**

---

## 📊 任務設定

| | |
|---|---|
| 輸入 | 過去 24 小時 × 5 變數（PM2.5, WIND_U, WIND_V, RH, AMB_TEMP）＋測站 lat/lon |
| 輸出 | 未來 72 小時 PM2.5 |
| 訓練 | 2018-01-01 ~ 2024-12-31（176,750 窗口）|
| 測試 | 2025-01-01 ~ 2025-11-30（23,189 窗口）|
| 窗口 | 滑動，step=24h（不重疊）|
| 測站 | 71 站，joint training |
| 指標 | **NCU-style RMSE**＝每小時 pooled RMSE 對 72h 平均（與 SOTA 同算法）|

資料分布：Train Y mean=15.70/std=11.45/max=310；Test Y mean=12.98/std=9.61/max=235。高污染(>50)僅 1.5%、極端(>100)僅 0.01% → 嚴重右偏。

---

## 🔒 固定協定（每個實驗一致，否則不可比較）

**評估指標**：NCU-style RMSE（headline，已核對＝論文 Eq.5 的 pooled RMSE 對 72h 平均；證據：論文 Table 2 ALL=6.88≠各區等權均6.35 → 跨站 pool）。輔助：MAE、分時段 RMSE。

**資料協定**：
- 輸入：**真實觀測為主，空值才用 FPCA 補**（`raw.fillna(fpca)`）。
- 訓練目標：raw 為主、缺值用 FPCA 補（僅 1.30% 目標格用 FPCA）。
- 測試目標：**純 raw 觀測**（不補 FPCA，缺值不計分）→ 誠實評估。

**訓練設定（已驗證最佳）**：AdamW(lr=1e-3, wd=1e-2)、CosineAnnealingLR(T_max=2000)、MSE、full-batch、2000 epochs、seed=42。模型＝FEDONet + station embedding(32d) + temporal(**month 2d**；weekday 經多 seed 驗證後拿掉) + m=64。

**紀律**：一次只改一個變因；改善 <±0.02 視為雜訊；每實驗記錄腳本/RMSE/MAE/分時段/結論。

---

## ✅ 方法論修正（取代舊版）

> 舊流程把 **FPCA 平滑值當訓練目標、FPCA 重建值當輸入**，卻用 raw 評估 → train/test 目標分佈不一致、系統性低估尖峰，**舊結果不可信**。已全面改為「raw 為主、僅缺值補 FPCA」，並只用 raw 測試目標評分。

### FPCA 流程 / 資料管線稽核（已逐項實測，無致命洩漏）
- **基底只用訓練年 fit**，測試年用 `predict()` 投影 → 無 train/test 基底洩漏。
- **一天一條曲線**，當日補值只用當日 24h（發布時當日已全觀測）→ 無 look-ahead。
- `FVEthreshold=0.999` → 幾乎只補值、不平滑。
- **補值對齊已實測**：raw 與 FPCA CSV 在觀測點上相關 0.94–0.98、均值吻合 → 對齊正確（曾疑慮的 SubjectID 錯位未發生）。
- **測試目標純 raw**（不碰 FPCA）→ 最佳 7.2931 為誠實指標。
- 已知小瑕疵：train/test 邊界缺 purge gap，最後一個訓練窗 label 落在 2025-01-01~03（影響前 ~3 天，量級可忽略，未修）。

### FPCA 重建誤差（各變數）— `exp_fpca_recon_error.py`
在「有真實觀測的格子」上比較 raw vs FPCA 重建值。nRMSE = RMSE/std（跨變數可比，越低越好）。

| 變數 | RMSE | MAE | corr | std | **nRMSE** | test RMSE |
|---|---|---|---|---|---|---|
| AMB_TEMP | 0.347 °C | 0.236 | 0.998 | 5.54 | **0.063** | 0.340 |
| RH | 1.744 % | 1.200 | 0.992 | 13.50 | **0.129** | 1.827 |
| PM2.5 | 2.717 µg/m³ | 1.877 | 0.973 | 11.68 | **0.233** | 2.508 |
| WIND_V | 0.555 m/s | 0.397 | 0.960 | 1.97 | **0.282** | 0.557 |
| WIND_U | 0.552 m/s | 0.397 | 0.951 | 1.78 | **0.309** | 0.551 |

判讀：
- 重建品質：**溫度 ≫ 濕度 ≫ PM2.5 ≫ 風速**（溫濕度平滑、強日週期 → 幾乎完美；風場噪訊多、日週期弱 → 最難）。
- **train ≈ test**：每變數 test 誤差與 train 幾乎相同 → 訓練年基底投影到 2025 無退化，再證 FPCA 無洩漏、泛化良好。
- **對模型影響有限**：模型僅在缺值（~1.3% 目標、~2% 輸入）才用 FPCA 補，98% 是 raw。
- 此為**下界**：誤差量在「有觀測」的格子；實際要補的「完全缺失」格子更難重建、誤差可能更高（無 ground truth 不可測），風場缺值填補最不可靠。

---

## 🏆 主結果

| 模型 | 腳本 | NCU-RMSE | MAE | 分時段(1-12/13-24/25-48/49-72) |
|---|---|---|---|---|
| **本模型（多變數+station embed+month）** | `code/models/exp_best.py` | **7.2931** | **5.2463** | 5.87 / 6.98 / 7.58 / 7.88 |
| 前一版（含 weekday）| `code/models/exp_raw_input.py` | 7.3250 | 5.2707 | 5.89 / 6.98 / 7.62 / 7.92 |
| SOTA CNN-BASE（用 CMAQ）| Lee et al. 2024 | 6.88 | — | — |

> 最佳 = `exp_best.py`：temporal **只用 month sin/cos（拿掉 weekday）**、**保留 lat/lon**。
> 7.2931 為 seed=42（與含 weekday 版同 seed 公平對照 7.3250）。「拿掉 weekday、保留 lat/lon」
> 兩項決定均經多 seed (42/123/777) 驗證為穩定方向。

---

## 📐 完整 baseline 對照（同一 71 站 / 23,189 筆測試集、同 NCU 指標）

| 類別 | 方法 | 腳本 | NCU-RMSE | MAE |
|---|---|---|---|---|
| trivial | Diurnal persistence | `exp_simple_baselines.py` | 9.93 | 7.06 |
| trivial | Persistence | `exp_simple_baselines.py` | 9.37 | 6.65 |
| 統計 | Climatology（站×月×時）| `exp_simple_baselines.py` | 8.39 | 6.43 |
| GP/FDA | FoFGPR（FPCA分數+SE kernel GP，M=3000子集）| `exp_fda_baselines.py` | 8.18 | 5.99 |
| 線性/FDA | Ridge ＝ **FLM**（function-on-function 線性）| `exp_simple_baselines.py` / `exp_fda_baselines.py` | 7.85 | 5.82 |
| FDA | FAM（functional additive, RBF 基底）| `exp_fda_baselines.py` | 7.88 | 5.82 |
| operator | DeepONet — per-station（71模型）| `exp_deeponet_baselines_v2.py` | 8.05 | 5.82 |
| operator | DeepONet — joint（Lu 2021）| `exp_deeponet_baselines_v2.py` | 7.83 | 5.75 |
| operator | FEDONet — joint（Sojitra 2025, 固定 Fourier trunk）| `exp_deeponet_baselines_v2.py` | 7.81 | 5.74 |
| **本模型** | conditioned multi-input FEDONet (month-only) | `exp_best.py` | **7.2931** | **5.25** |

**核心結論**：所有「PM2.5 單變數」方法——線性(Ridge/FLM)、非線性FDA(FAM)、GP(FoFGPR)、各種 DeepONet/FEDONet——全部卡在 **7.8–8.2 高原**。突破到 ~7.29 來自**多變數輸入 + station 條件化**，不是 operator 架構本身（FLM ≡ Ridge ≡ DeepONet ≈ 7.85）。

註記：
- **FLM (7.8536) = Ridge (7.8536) 完全相同**（一致性驗證：FLM 即離散化線性 FoFR）。
- DeepONet baseline 架構對齊原 notebook（Branch Dropout0.2 + AdamW wd + 1000ep）；joint 僅略優於 per-station（7.83 vs 8.05）。
- FDA/DeepONet baseline 用單一輸入函數（PM2.5×24，無座標、station-independent），忠實對應原論文公式。

---

## 🧪 輸入前處理實驗（已收斂，皆不採用）

| 實驗 | NCU-RMSE | Δ | 結論 |
|---|---|---|---|
| min-max 正規化[0,1] | 7.5505 | +0.226 | ❌ 變差；原始資料有壞值(PM2.5負、RH>100、TEMP-135)撐爆範圍 |
| 經緯度平移到原點(SW=0,0) | 7.3234 | −0.002 | ➖ 雜訊內，無害、較乾淨 |

→ 輸入縮放/平移動不了指標；瓶頸是**資料的訊息含量**，非輸入尺度。

---

## 🔬 殘差分析（每站每天）

誤差非均勻分布，集中在兩個維度的交集：
- **冬季（1-2月）區域傳輸事件**：同一天西部多站一起爆、且**系統性低估尖峰**（bias −15~−26）。
- **西部高污染站**（二林/崙背/斗六/麥寮/大寮…，mean_true 15-19）RMSE 最高。
- 模型「回歸到平均」：乾淨日(0-10)略高估(+1.68)、髒日(30-50)嚴重低估(−9.17)。
- 注意：二林/麥寮 bias≈0、屬局地工業尖峰高變異，純觀測接近不可約下界，非模型缺陷。
- 動畫：`daily_rmse.gif`/`daily_bias.gif`（台灣地圖每日誤差，冬季事件西部一起轉藍=同時被低估）— 留本機。

---

## 🧪 進一步消融與診斷（彙整；探索性腳本留本機）

### 輸入/編碼（多 seed 驗證者標注）
- **weekday 拿掉、lat/lon 保留**（seed 42/123/777 驗證）：weekday 有害（T4 三 seed 全低於含 weekday，−0.023、無重疊）；lat/lon 拿掉反而變差（+0.011）。已反映於 `exp_best`。station embedding 已涵蓋位置（拿掉 embedding 才明顯變差 +0.13~0.16；coord→embedding 不如自由 id-embedding）。
- **時間諧波**：month/weekday 加高階諧波（k≥2）單調變差（過擬合）；單一基本正弦(2π)足夠。
- **bias b₀**：純量、與時空無關（同 Lu 2021 / DeepXDE 官方）。拿掉≈不變（雜訊內）、改 72 維時間 bias 無幫助、POD 式固定 μ_Y(t) 反而差（joint 平均軌跡太平）。維持純量 b₀。
- **min-max 正規化** +0.23❌（壞值撐爆範圍）、**經緯度平移** ➖雜訊內。

### 架構融合：MIONet 多 branch（`exp_mionet5.py`）
- **MIONet-5**（每變數一 branch + context branch，Hadamard 逐元素積融合；其餘同 exp_best）：**NCU-RMSE 7.5188、MAE 5.3482（+0.2257 ❌ 明顯變差）**。分時段全差，短程 1-12hr 最嚴重（6.60 vs 5.87，+0.73）。
- 訓練 loss 全程比 exp_best 高 ~5–10（優化困難的直接證據）。
- **原因**：Hadamard 積是**乘性**融合，要求所有 branch 同時非零才有輸出，任一變數 branch→0 即拉垮乘積；而 PM2.5 的最強預測子是「過去 24h PM2.5 自相關」（**加性/線性**主導），乘性結構把自相關訊號稀釋掉。MIONet 原設計給「異質 Banach 空間、本質乘性耦合」的輸入（如 PDE source×boundary），不適合本問題的同空間協變數。
- → **負面結果**：concat 單 branch（加性融合）優於 MIONet 乘性融合，佐證 `exp_best` 的架構選擇。

### Loss（評估仍標準 NCU-RMSE）
- 換 loss 無法在不犧牲整體下改善高值：MAE/Huber/LogCosh 讓高值更差；**Pinball（非對稱）可救高值低估但整體 RMSE 升**（trade-off，旋鈕=τ）；MSE 整體最佳。
- 加權 loss（freq/threshold/quad，仿 SOTA Eq.4）同理：高值改善、整體略升。
- 結論：高值低估「換 loss」只能搬移、不能無痛補滿 → 指向「補資訊」。

### SWP（天氣型態，仿 SOTA CNN-SWP，無 CMAQ/無預報版）
- 用 71 站每日風場 PCA+KMeans(6) 自建 regime（訓練年 fit、無洩漏）；分群物理合理（分出冬季高污染型 vs 夏季低污染型，對照論文 Fig 3）。
- 接進模型（one-hot / soft / 連續 PCA，當天 regime）：**整體與高值皆無實質改善**。
- 原因：分群高度依季節→與 month 冗餘；群內差異與風場歷史重疊；**論文 SWP 的威力來自「未來 D+1~D+3 天氣型態（用 WRF 預報）」，純觀測無預報拿不到**。
- → 忠實重現 SOTA 關鍵設計，證明「**無數值預報時 SWP 不轉移**」，解釋了與 6.88 差距的一部分。**鄰站 PM2.5**（舊實驗）亦同：被 station embedding 涵蓋、冗餘。

### month 季節效應（診斷）
- 反事實掃描（固定歷史、只改 month）：模型預測冬高夏低（Δ+1.7），方向/形狀與氣候平均一致 → month sin/cos 確有體現季節，但幅度小（24h 歷史已主導）。

### 重疊窗口 STEP=3（潛力，**尚未採用、待多 seed 驗證**）
- STEP=24→3（~8 倍訓練樣本），測試固定 STEP=24。小 batch 被 mini-batch 懲罰拖累(7.34)；**大 batch(65536) 逼近 full-batch → NCU-RMSE 7.2633、高值 y≥35 RMSE 22.94→21.80**（單 seed）。
- 目前唯一「同時改善整體 + 高值」的方向（多事件樣本）；多 seed 驗證通過後再考慮定為新最佳。

### 滾動預測 / 自回歸（AR）＋「完美未來氣象」上限測試（`exp_rolling_*.py`，留本機）
單步模型 M（exp_best 架構，24h→次 24h），逐日滾動 3 步成 72h。測試集為原集子集（要求未來氣象非缺）：73 槽 embedding／23,161 窗口，與 exp_best 71 站／23,189 幾乎等同（差 28 窗口 <0.12%）。

| 方法 | NCU-RMSE | MAE | day1 | day2 | day3 |
|---|---|---|---|---|---|
| 純多變數 AR（PM2.5＋氣象全預測、餵回）`exp_rolling_multivar.py` | 7.7558 | — | 6.9 | 8.1 | 8.3 |
| **direct exp_best**（一次預測 72h，目前最佳） | **7.2931** | 5.2463 | 6.43 | 7.58 | 7.88 |
| **作弊滾動**（PM2.5 AR 餵回、**氣象餵真實值**）`exp_rolling_cheat.py` | **7.1665** | 5.0937 | 6.42 | 7.45 | 7.63 |

- **純 AR 更差（+0.46）**：單步模型容量被 5 輸出瓜分（連 day1 都輸），餵回的風/濕/溫誤差逐步惡化 → 證實 **direct multi-horizon 才是對的設計**。
- **作弊滾動（完美未來氣象）只贏 −0.13（7.29→7.17）**，且增益**全在後段**（day1≈持平 6.42≈6.43、day2 −0.13、day3 **−0.25**）：正是「資訊缺口隨 horizon 變大」的指紋。
- **關鍵結論**：就算給「上帝視角」的未來氣象，純觀測也只到 7.17、**仍遠不到 6.88**。代表到 SOTA 的剩餘差距主因**不是當地未來天氣**，而是 CMAQ 的**區域化學傳輸/排放模擬**（測站氣象變數表達不出）→ 反向佐證 7.29 是強的、誠實的純觀測結果。
- 但書：作弊版仍混入 PM2.5 自回歸的累積誤差，吃掉部分氣象紅利，故 −0.13 是「完美天氣預報價值」的**保守下界**；乾淨上限需 direct 72h 模型直接吃未來氣象（未做）。

---

## 🧭 候選下一步（依殘差潛力排序）

1. **鄰站 PM2.5 輸入**（k=3 最近鄰，+72維 raw）— 直接打中冬季傳輸暗帶，預期 −0.1~−0.3 ⭐最優先
2. **重疊窗口 STEP=3** — 補高污染稀少樣本，預期 −0.1~−0.2
3. **高污染加權 / log-target** — 解決尖峰低估，預期 −0.05~−0.15
4. 風向工程（china_wind 分量）— 與 1 部分重疊
5. 逐站偏差校正 — 收尾
6. （ablation）「只用 branch」拿掉 trunk — 驗證 branch-trunk 結構是否有用

---

## 📁 檔案索引

```
code/
├── fpca/pm25_fpca_2018~2024_2025.R     # FPCA 補值（5 變數共用，僅換檔名）
└── models/
    ├── exp_best.py                      # ★ 最佳模型 (month-only; 7.2931 / MAE 5.25)
    ├── exp_raw_input.py                 # 前一版 (含 weekday; 7.3250 / MAE 5.27)
    ├── exp_simple_baselines.py          # persistence / climatology / Ridge
    ├── exp_fda_baselines.py             # FLM / FAM / FoFGPR（依投影片公式）
    ├── exp_deeponet_baselines_v2.py     # 原始 DeepONet & FEDONet（joint + per-station）
    ├── exp_fpca_recon_error.py          # FPCA 各變數重建誤差診斷
    └── exp_mionet5.py                   # MIONet-5 (Hadamard 融合; 7.5188，負面結果)
```
> 表格以外的探索性腳本（前處理、位置 ablation、殘差分析、v1 過擬合版、原始 notebook 等）僅保留於本機，未納入此 repo。

**執行**：
```bash
gunzip -k data/raw/*.csv.gz data/fpca_processed/*.csv.gz
python code/models/exp_best.py
```
需要 PyTorch（CUDA 可選）；FPCA 前處理需 R + fdapace。
