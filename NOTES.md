# RL-based Congestion Window Control — 開發筆記

## 專案說明

RL-based Congestion Window Control in Simulated Wireless Networks
使用 DQN 與 DDPG 在模擬無線網路環境中自動調整 TCP congestion window。

---

## 目前進度

- DQN 和 DDPG 都訓練完成，結果穩定
- 四模式比較（DQN / DDPG / Reno / Cubic）已跑完
- 所有程式碼已 push 到 GitHub agent branch
- 待完成：撰寫報告

---

## 檔案結構

```
RL_FinalProject/
├── agent.py              # DQN Agent + DDPG Agent
├── train_dqn.py          # DQN 訓練主程式（socket 版）
├── train_ddpg.py         # DDPG 訓練主程式（socket 版）
├── compare.py            # 四模式比較展示（需先啟動 env_server.py）
├── client.py             # 環境 client（昶安提供）
├── env_server.py         # 環境 server（昶安提供，已補上 DDPG cwnd 處理）
├── main.py               # 單模式測試（昶安提供）
├── environment/
│   ├── __init__.py
│   ├── env.py            # TCPEnv 主體（max_steps=200）
│   ├── agent_sender.py   # Agent 用的 Sender（支援 apply(cwnd=...)）
│   ├── reno_sender.py    # TCP Reno
│   ├── cublic_sender.py  # TCP CUBIC
│   ├── sender.py         # BaseSender
│   ├── router.py         # 路由器（bandwidth / queue / loss）
│   ├── receiver.py       # 封包接收端（計算 AoI）
│   └── packet.py         # 封包定義
├── image/                # 訓練結果與比較圖（自動產生）
├── logs/                 # 比較模擬的詳細 log（自動產生）
└── pyproject.toml        # 記得加 torch, gymnasium, stable-baselines3, numpy
```

---

## 兩階段架構

**訓練階段：** 透過 socket 連 env_server，需要先開 server。

```
train_dqn.py / train_ddpg.py
  → TCPEnvClient → env_server.py → TCPEnv
  → 儲存 dqn_agent.pth / ddpg_agent.pth
```

**展示比較階段：** 同樣透過 socket，四模式同時跑，公平比較。

```
env_server.py（server）
      ↕ socket / JSON
compare.py（載入訓練好的模型，四模式同時跑）
  → DQN / DDPG / Reno / Cubic
  → 相同 seed，輸出比較圖與 log
```

---

## API 約定

| 項目 | 格式 | 說明 |
|------|------|------|
| Action（DQN） | int ∈ {0, 1, 2} | 0=減少 / 1=不變 / 2=增加 cwnd |
| Action（DDPG） | float ∈ [1.0, 100.0] | 直接輸出目標 cwnd 值 |
| State | np.ndarray shape=(5,) | [cwnd, throughput, aoi, loss_rate, queue]，正規化後各值 0.0~1.0 |
| Reward | float | 由環境計算後回傳，agent 不自己算 |

### 正規化上限

```python
MAX_CWND       = 100.0
MAX_THROUGHPUT = 80.0
MAX_AOI        = 30.0
loss_rate               # 本來就是 0.0~1.0
MAX_QUEUE      = 300.0
```

### Reset 初始 State

```python
[0.1, 0.0, 0.0, 0.0, 0.0]
# cwnd=10（BaseSender 預設值），其他全 0
```

### Reward 公式（environment/env.py）

```python
reward = throughput - 0.1 * rtt - 5 * loss_rate - 0.5 * aoi
# loss_rate 係數從 20 調整為 5（避免 DQN 學到過度保守策略）
```

### info 結構

env_server.py 的 normalize_info() 攤平後（單層）：
```python
info["cwnd"]
info["throughput"]
info["loss_rate"]
info["aoi"]
info["queue"]
```

---

## Client API

```python
client = TCPEnvClient()
resp = client.create(mode="agent", seed=42)   # 建立 session
session_id = resp["session_id"]
client.reset(session_id)                       # reset 環境

# DQN
_, reward, done, info = client.step(session_id, action=0, cwnd=None)

# DDPG
_, reward, done, info = client.step(session_id, action=None, cwnd=15.3)
```

### 已知問題

client.py 的 step() 原本用 `res.get("done", False)`，但 env_server.py 回傳的是 `terminated` 和 `truncated`，導致 done 永遠是 False，while 迴圈不會結束。

修正方法（已修正）：
```python
done = res.get("terminated", False) or res.get("truncated", False)
```

---

## Agent 設計（agent.py）

### DQN

**架構：**
```
State (5,) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(3) → Q values
```

**超參數：**
```python
LR               = 1e-3
GAMMA            = 0.99
BATCH_SIZE       = 64
BUFFER_SIZE      = 10_000
LEARNING_STARTS  = 500
TARGET_UPDATE_FREQ = 500
EPS_START        = 1.0
EPS_END          = 0.05
EPS_DECAY        = 0.99
GRAD_CLIP        = 1.0
N_EPISODES       = 1000
```

---

### DDPG

**Actor 架構：**
```
State (5,) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(1) → Sigmoid → [1.0, 100.0]
```

**Critic 架構：**
```
[State(5) + Action(1)] → Linear(256) → ReLU → Linear(256) → ReLU → Linear(1) → Q value
```

**超參數：**
```python
LR_ACTOR          = 1e-4
LR_CRITIC         = 1e-3
GAMMA             = 0.99
TAU               = 0.005
BATCH_SIZE        = 64
BUFFER_SIZE       = 100_000
LEARNING_STARTS   = 2000
GRAD_CLIP         = 1.0
NOISE_SIGMA_START = 3.0
NOISE_SIGMA_END   = 0.3
NOISE_DECAY       = 0.99
N_EPISODES        = 1000
```

---

## 調整記錄

### Reward 係數

| 參數 | 初始值 | 最終值 | 原因 |
|------|--------|--------|------|
| loss_rate 係數 | 20 | 5 | DQN 學到過度保守策略，throughput 只有 ~1 |

### DDPG 參數

| 參數 | 初始值 | 最終值 | 原因 |
|------|--------|--------|------|
| NOISE_SIGMA_START | 10.0 | 3.0 | 前期 CWND 暴跌，noise 過大 |
| NOISE_SIGMA_END | 0.5 | 0.3 | 配合 sigma_start 調整 |
| LEARNING_STARTS | 1000 | 2000 | 讓 buffer 累積更多樣本再開始訓練 |
| GRAD_CLIP | 無 | 1.0 | Critic Loss 持續上漲，加 clipping 緩解 |

### DQN 參數

| 參數 | 初始值 | 最終值 | 原因 |
|------|--------|--------|------|
| EPS_DECAY | 0.995 | 0.99 | 收斂更快 |
| GRAD_CLIP | 無 | 1.0 | 避免梯度爆炸 |

### State 維度調整

| 版本 | State | 原因 |
|------|-------|------|
| 初始 | [throughput, rtt, loss_rate]（3維） | mock env 時期 |
| 最終 | [cwnd, throughput, aoi, loss_rate, queue]（5維） | 接上真實環境 |

---

## 訓練結果（最終版）

### DQN（N_EPISODES=1000）

- Reward：收斂到 ~450
- Throughput：~14 pkts
- Loss Rate：~0.025
- AoI：接近 0

### DDPG（N_EPISODES=1000）

- Reward：收斂到 ~50
- Throughput：~15 pkts
- Loss Rate：~0.03
- AoI：接近 0
- CWND：穩定在 ~15
- Critic Loss：持續上漲（Q value overestimation，標準 DDPG 已知問題）

---

## 比較結果（四模式，150 steps）

| 指標 | DQN | DDPG | Reno | Cubic |
|------|-----|------|------|-------|
| Throughput | 低（卡底部） | ~15 穩定 | ~5 保守 | ~15（大幅震盪） |
| AoI | 低 | 低 | 低 | 高（最高 ~20） |
| Loss Rate | 偶爾衝高 | 低穩定 | 低穩定 | 高（最高 ~0.85） |
| CWND | 壓在底部 | ~15 穩定 | 緩慢爬升 | 大幅震盪（最高 ~220） |

### 已知問題說明

**DQN 在展示時卡住：**
- cwnd 掉到 1.0 後無法恢復
- 原因：離散 action 幅度固定（×0.7 / ×1.2），高度隨機的無線環境持續觸發減少
- 報告討論點：DDPG 能直接輸出目標 cwnd 跳脫局部狀態，體現連續 action 的優勢

**DDPG Critic Loss 持續上漲：**
- Q value overestimation，標準 DDPG 的已知限制
- Reward 已收斂，訓練本身沒有問題

---

## 執行流程

### 環境建立

```bash
uv venv
.venv\Scripts\activate     # Windows
uv sync
```

### 訓練（需先開 server）

```bash
# Terminal 1
python env_server.py

# Terminal 2
python train_dqn.py    # 產生 dqn_agent.pth
# 或
python train_ddpg.py   # 產生 ddpg_agent.pth
```

### 展示比較

```bash
# Terminal 1
python env_server.py

# Terminal 2
python compare.py
```

### 輸出檔案

| 檔案 | 說明 |
|------|------|
| dqn_agent.pth | DQN 模型權重（不進 git） |
| ddpg_agent.pth | DDPG 模型權重（不進 git） |
| image/dqn_training_*.png | DQN 訓練曲線 |
| image/ddpg_training_*.png | DDPG 訓練曲線 |
| image/compare_*.png | 四模式比較圖 |
| logs/*_{MODE}.txt | 各模式詳細 log（不進 git） |

---

## 開發環境

- Python 3.12（uv）
- Formatter：Ruff
- 虛擬環境：.venv

### 依賴（pyproject.toml 記得加）

```toml
dependencies = [
    "numpy>=2.4.5",
    "torch",
    "gymnasium>=0.29",
    "stable-baselines3>=2.3",
    "matplotlib>=3.10.9",
]
```

---

## TODO

- [ ] 撰寫報告：結果分析、圖表說明
