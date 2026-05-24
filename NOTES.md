# RL-based Congestion Window Control — 開發筆記

## 專案說明

RL-based Congestion Window Control in Simulated Wireless Networks
使用 DQN 與 DDPG 在模擬無線網路環境中自動調整 TCP congestion window。

---

## 檔案結構

```
RL_FinalProject/
├── agent.py              # DQN Agent + DDPG Agent
├── env_wrapper.py        # 包裝 TCPEnv 成 Gym 介面（訓練用）
├── train_dqn.py          # DQN 訓練主程式
├── train_ddpg.py         # DDPG 訓練主程式
├── compare.py            # 四模式比較展示（需先啟動 env_server.py）
├── client.py             # 環境 client（昶安提供）
├── env_server.py         # 環境 server（昶安提供，已補上 DDPG cwnd 處理）
├── main.py               # 單模式測試（昶安提供）
├── environment/
│   ├── __init__.py
│   ├── env.py            # TCPEnv 主體
│   ├── agent_sender.py   # Agent 用的 Sender（支援 apply(cwnd=...)）
│   ├── reno_sender.py    # TCP Reno
│   ├── cublic_sender.py  # TCP CUBIC
│   ├── sender.py         # BaseSender
│   ├── router.py         # 路由器（bandwidth / queue / loss）
│   ├── receiver.py       # 封包接收端（計算 AoI）
│   └── packet.py         # 封包定義
├── image/                # 訓練結果與比較圖（自動產生）
├── logs/                 # 比較模擬的詳細 log（自動產生）
└── pyproject.toml
```

---

## 兩階段架構

**訓練階段（方案 A）：** 直接 import TCPEnv，不需要 server，速度快。

```
train_dqn.py / train_ddpg.py
  → DQNEnvWrapper / DDPGEnvWrapper（直接 import TCPEnv）
  → 儲存 dqn_agent.pth / ddpg_agent.pth
```

**展示比較階段（方案 B）：** 透過 socket 與 env_server 溝通，四模式同時跑，公平比較。

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
| Action（DQN） | `int` ∈ {0, 1, 2} | 0=減少 / 1=不變 / 2=增加 cwnd |
| Action（DDPG） | `float` ∈ [1.0, 100.0] | 直接輸出目標 cwnd 值 |
| State | `np.ndarray` shape=(5,) | `[cwnd, throughput, aoi, loss_rate, queue]`，正規化後各值 0.0~1.0 |
| Reward | `float` | 由環境計算後回傳，agent 不自己算 |
| 介面 | `gymnasium.Env` | `reset()` / `step()` 符合 Gym 規範 |

### 正規化上限

```python
MAX_CWND       = 100.0
MAX_THROUGHPUT = 80.0    # router.base_bandwidth 上限
MAX_AOI        = 30.0    # 觀測值上限（保留緩衝）
loss_rate               # 本來就是 0.0~1.0，不需正規化
MAX_QUEUE      = 300.0   # router.queue_limit
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

### info 結構（environment/env.py 巢狀）

訓練時（env_wrapper.py 直接讀）：
```python
info["sender"]["cwnd"]
info["network"]["throughput"]
info["network"]["loss_rate"]
info["receiver"]["aoi"]
info["router"]["queue_size"]
```

展示時（env_server.py 的 normalize_info() 攤平後）：
```python
info["cwnd"]
info["throughput"]
info["loss_rate"]
info["aoi"]
info["queue"]
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

**設計重點：**
- Double network：q_net（訓練）+ target_net（凍結），每 500 步同步
- Replay Buffer：隨機抽樣打破時間相關性
- ε-greedy：初始完全探索，逐漸轉向利用，最低保留 5%
- Gradient clipping：避免梯度爆炸

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

**設計重點：**
- 四個網路：Actor × 2 + Critic × 2（online + target 各一）
- Soft update：每步以 TAU=0.005 更新 target network
- Gaussian Noise：探索用，σ 從 3.0 慢慢降到 0.3
- Critic Gradient clipping：緩解 Q value overestimation

---

## 環境整合（env_wrapper.py）

### 處理的問題

| 問題 | 解法 |
|------|------|
| `step()` 回傳格式不符 | Wrapper 直接接 5 個回傳值（terminated / truncated） |
| `reset()` 只回傳 `state` | Wrapper 補上空的 `info` dict |
| 沒有繼承 `gymnasium.Env` | Wrapper 繼承並補上 `action_space` / `observation_space` |
| reset 時無網路數據 | 初始 state 固定為 `[0.1, 0.0, 0.0, 0.0, 0.0]` |
| DQN / DDPG action 格式不同 | 分成 `DQNEnvWrapper` / `DDPGEnvWrapper` 兩個 class |
| `env.step()` 不支援 cwnd 參數 | DDPGEnvWrapper 直接設定 `self._env.sender.cwnd = cwnd` 再呼叫 `step(action=None)` |

### DQN vs DDPG Wrapper 差異

| | DQNEnvWrapper | DDPGEnvWrapper |
|---|---|---|
| `action_space` | `Discrete(3)` | `Box([1.0], [100.0])` |
| cwnd 設定方式 | `env.step(action=action)` | `sender.cwnd = cwnd` → `env.step(action=None)` |

### env_server.py 的 DDPG cwnd 處理

展示比較時，`env_server.py` 的 step 區塊補上 cwnd 讀取：

```python
cwnd = request.get("cwnd")   # DDPG 用
if cwnd is not None:
    env.sender.cwnd = float(cwnd)

state, reward, terminated, truncated, info = env.step(action)
```

原因：`env.step()` 只接 `action` 參數，DDPG 要傳 cwnd 只能直接設定 `sender.cwnd`。

---

## 調整記錄

### Reward 係數

| 參數 | 初始值 | 最終值 | 原因 |
|------|--------|--------|------|
| `loss_rate` 係數 | 20 | 5 | DQN 學到過度保守策略，寧願不傳也不冒險 |

### DDPG 參數

| 參數 | 初始值 | 最終值 | 原因 |
|------|--------|--------|------|
| `NOISE_SIGMA_START` | 10.0 | 3.0 | 前期 CWND 暴跌，noise 過大 |
| `NOISE_SIGMA_END` | 0.5 | 0.3 | 配合 sigma_start 調整 |
| `LEARNING_STARTS` | 1000 | 2000 | 讓 buffer 累積更多樣本再開始訓練 |
| `GRAD_CLIP` | 無 | 1.0 | Critic Loss 持續上漲，加 clipping 緩解 |

### DQN 參數

| 參數 | 初始值 | 最終值 | 原因 |
|------|--------|--------|------|
| `EPS_DECAY` | 0.995 | 0.99 | 收斂更快 |
| `GRAD_CLIP` | 無 | 1.0 | 避免梯度爆炸 |

### Normalization Constants

| 參數 | 初始值 | 最終值 | 原因 |
|------|--------|--------|------|
| `MAX_THROUGHPUT` | 100 | 80 | 配合真實環境 `router.base_bandwidth` |
| `MAX_CWND` | — | 100.0 | 對應 DDPG action 上限 |
| `MAX_AOI` | — | 30.0 | 從 CUBIC log 觀察到最高約 23，取 30 留緩衝 |
| `MAX_QUEUE` | — | 300.0 | 對應 `router.queue_limit` |

### State 維度調整

| 版本 | State | 原因 |
|------|-------|------|
| 初始 | `[throughput, rtt, loss_rate]`（3維） | mock env 時期 |
| **最終** | `[cwnd, throughput, aoi, loss_rate, queue]`（5維） | 接上真實環境 |

---

## 訓練結果（最終版）

### DQN

- Reward：收斂到 ~450
- Throughput：~14 pkts
- Loss Rate：~0.025
- AoI：接近 0

### DDPG

- Reward：收斂到 ~50
- Throughput：~15 pkts
- Loss Rate：~0.03
- AoI：接近 0
- CWND：穩定在 ~15
- Critic Loss：持續上漲（Q value overestimation，標準 DDPG 已知問題）

---

## 比較結果（四模式）

| 指標 | DQN | DDPG | Reno | Cubic |
|------|-----|------|------|-------|
| Throughput | 低（卡底部） | ~15 穩定 | ~5 保守 | ~15（大幅震盪） |
| AoI | 低 | 低 | 低 | 高（持續累積，最高 ~20） |
| Loss Rate | 偶爾衝高 | 低穩定 | 低穩定 | 高（爆衝時最高 ~0.85） |
| CWND | 壓在底部 | ~15 穩定 | 緩慢爬升 | 大幅震盪（最高 ~220） |

### 已知問題說明

**DQN 在展示時卡住：**
- cwnd 掉到 1.0 後無法恢復
- 原因：離散 action 幅度固定（×0.7 / ×1.2），高度隨機的無線環境持續觸發減少，無法有效爬升
- 說明：這是 DQN 離散 action 的設計限制，DDPG 能直接輸出目標 cwnd 跳脫局部狀態，體現連續 action 的優勢

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

### 訓練

```bash
python train_dqn.py    # 產生 dqn_agent.pth
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
| `dqn_agent.pth` | DQN 模型權重（不進 git） |
| `ddpg_agent.pth` | DDPG 模型權重（不進 git） |
| `image/dqn_training_*.png` | DQN 訓練曲線 |
| `image/ddpg_training_*.png` | DDPG 訓練曲線 |
| `image/compare_*.png` | 四模式比較圖 |
| `logs/*_{MODE}.txt` | 各模式詳細 log（不進 git） |

---

## 開發環境

- Python 3.12（uv）
- Formatter：Ruff（存檔自動修正）
- 虛擬環境：`.venv`

### 依賴

```toml
dependencies = [
    "matplotlib>=3.10.9",
    "numpy>=2.4.5",
    "torch",
    "gymnasium>=0.29",
    "stable-baselines3>=2.3",
]
```

---

## TODO

- [ ] 撰寫報告：結果分析、圖表說明
