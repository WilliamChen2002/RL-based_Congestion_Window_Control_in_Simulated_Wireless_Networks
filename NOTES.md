# RL-based Congestion Window Control — 開發筆記

## 專案說明
RL-based Congestion Window Control in Simulated Wireless Networks
使用 DQN 在模擬無線網路環境中自動調整 TCP congestion window。

---

## 檔案結構

```
RL-based_Congestion_Window_Control/
├── agent.py              # DQN Agent 本體
├── train.py              # 訓練主程式
├── environment/
│   ├── __init__.py       # 補上 TCPEnvWrapper export
│   ├── env_wrapper.py    # 包裝真實環境成 Gym 介面
│   ├── tcp.py            # import 改成相對路徑
│   ├── sender.py         # import 改成相對路徑
│   ├── router.py
│   ├── receiver.py
│   └── packet.py
├── image/                # 訓練結果圖片
├── pyproject.toml        # 專案設定、依賴、Ruff 設定
└── .vscode/
    └── settings.json     # Workspace 共用設定（Ruff formatter）
```

---

## API 約定

| 項目 | 格式 | 說明 |
|------|------|------|
| Action | `int` ∈ {0, 1, 2} | 0=減少 / 1=不變 / 2=增加 cwnd |
| State | `np.ndarray` shape=(3,) | `[throughput, rtt, loss_rate]`，正規化後各值 0.0~1.0 |
| Reward | `float` | 由環境計算後回傳，agent 不自己算 |
| 介面 | `gymnasium.Env` | `reset()` / `step()` 符合 Gym 規範 |

### 正規化上限（依真實環境設定）
```
MAX_THROUGHPUT = 20.0   # router.bandwidth 上限
MAX_RTT        = 75.0   # base_delay(50) + max queue_delay(50*0.5=25)
loss_rate               # 本來就是 0.0~1.0，不需正規化
```

### Reward 公式（由真實環境計算）
```python
reward = throughput - 0.1 * rtt - 20 * loss_rate
```

---

## Agent 設計（agent.py）

### 演算法：DQN
- 時間緊迫選 DQN，action 離散、實作成熟、收斂相對快
- 未來有餘裕可換 PPO

### 架構
```
State (3,) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(3) → Q values
```

### 超參數（最終版）
```python
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 10_000
LEARNING_STARTS = 500    # buffer 累積到 500 步才開始訓練
TARGET_UPDATE_FREQ = 500
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.99         # 調整過，見下方調整記錄
```

---

## 環境整合（env_wrapper.py）

真實環境介面與 Gym 規範不符，使用 Wrapper 在外層轉換，不動真實環境的 code。

### 處理的問題
| 問題 | 解法 |
|------|------|
| `step()` 回傳 4 個值 | Wrapper 把 `done` 拆成 `terminated` / `truncated` |
| `reset()` 只回傳 `state` | Wrapper 補上空的 `info` dict |
| 沒有繼承 `gymnasium.Env` | Wrapper 繼承並補上 `action_space` / `observation_space` |
| State 格式不符 | Wrapper 從 `info` 取原始值並正規化成 `[throughput, rtt, loss_rate]` |

### 修改真實環境的部分
- `environment/tcp.py`：`from environment import ...` 改成相對路徑
- `environment/sender.py`：`from environment import Packet` 改成相對路徑
- `environment/__init__.py`：補上 `from .env_wrapper import TCPEnvWrapper`

原因：絕對路徑會觸發 `__init__.py` 造成 circular import，改相對路徑直接找檔案跳過。

---

## 調整記錄

### N_EPISODES
| 值 | 結果 |
|----|------|
| 300 | 收斂不夠充分 |
| **1000** | 收斂穩定，採用 |

### EPS_DECAY
| 值 | 結果 |
|----|------|
| 0.998 | 探索期太長，RTT episode 400~600 明顯衝高，淘汰 |
| 0.995 | RTT episode 500~600 衝高，次佳 |
| **0.99** | 收斂最快，episode 100 前就穩定，採用 |

### 正規化上限
| 參數 | 初始值 | 最終值 | 原因 |
|------|--------|--------|------|
| MAX_THROUGHPUT | 100 Mbps | 20 Mbps | 配合真實環境 router.bandwidth |
| MAX_RTT | 500 ms | 75 ms | 配合真實環境實際範圍 |

---

## 訓練結果（真實環境，EPS_DECAY=0.99，N_EPISODES=1000）

收斂後表現（episode 100 之後）：
- Reward：從 -100 收斂到 300+
- Throughput：收斂在 17 Mbps（router.bandwidth=20 的 85%）
- RTT：維持在 51ms 左右
- Loss：穩定在 0.05 左右

---

## 開發環境

- Python 3.12.12（uv）
- Formatter：Ruff（存檔自動修正）
- 虛擬環境：`.venv`（`uv venv` + `uv sync`）

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

### .vscode/settings.json（Workspace 共用）
```jsonc
{
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff"
    },
    "editor.codeActionsOnSave": {
        "source.fixAll": "explicit",
        "source.organizeImports": "explicit"
    }
}
```

---

## 執行流程

### 環境建立
```bash
uv venv                    # 建立虛擬環境
.venv\Scripts\activate     # 啟動（Windows）
uv sync                    # 安裝依賴
```

### 訓練
```bash
python train.py
```

流程：
```
train.py
  ├── TCPEnvWrapper()      建立環境（包裝真實環境）
  ├── check_env()          驗證 Gym 介面
  ├── DQNAgent()           建立 agent
  └── 訓練迴圈（N_EPISODES=1000）
        ├── env.reset()
        ├── agent.select_action()   ε-greedy 選 action
        ├── env.step()              環境執行 action，回傳 state / reward
        ├── agent.store()           存入 replay buffer
        ├── agent.train_step()      從 buffer 抽樣更新 Q network
        └── agent.decay_epsilon()   每個 episode 結束降低探索率
  ├── agent.save()         儲存模型（dqn_agent.pth）
  └── plot()               畫圖（image/training_result_時間戳.png）
```

### 輸出檔案
| 檔案 | 說明 |
|------|------|
| `dqn_agent.pth` | 訓練好的模型權重（不進 git） |
| `image/training_result_YYYYMMDD_HHMMSS.png` | 訓練曲線，每次跑都會產生新檔案 |

### 載入已訓練模型
```python
from agent import DQNAgent
agent = DQNAgent()
agent.load("dqn_agent.pth")
```

---
