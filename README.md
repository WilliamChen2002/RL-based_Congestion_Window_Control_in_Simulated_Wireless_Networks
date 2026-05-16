**專題名稱**

RL-based Congestion Window Control in Simulated Wireless Networks

組長：M1454025陳昶安，組員：M1454006黃洸昱

**選定方向**

方向 A：Self-evolving
Agent（自我進化型 Agent）

**動機與背景**

傳統 TCP congestion control 演算法多採用固定規則進行 congestion window 調整，然而在無線網路環境中，packet loss不一定代表網路壅塞，可能來自於無線干擾、訊號衰減或頻寬波動，因此傳統 TCP congestion control演算法容易造成效能和吞吐量的下降。

**系統架構設計圖（含模組劃分）**

![]()![structure](image/README/structure.png)

實驗環境，模擬簡化版的TCP網路環境，包含：頻寬（Bandwidth）、RTT、Packet
Loss、Queue Length與AoI等。

API：操作實驗環境的橋接器，用來組合分工的約定。

代理人：據 throughput、RTT、AoI 與 packet loss 計算
reward，作為 Agent 學習依據並繪製 cwnd、throughput、RTT 與
packet loss 等變化圖表，用於分析 RL Agent 行為與效能。

**預期功能與技術方案**

本專題希望可以設計一個Agent，使其能夠根據 Throughput與AoI等資訊，自動調整 congestion window，提升整體網路效能與適應能力。

附註：AoI(Age of Infromation)：一個新的網路指標來驗就，2012提出。

**分工說明（每位組員的負責項目）**

陳昶安：模擬環境與API設計，代理人演算法討論，報告撰寫

黃洸昱：API討論，代理人設計與系統整合，PPT製作

**預計時程表（Gantt 圖或表格）**

| 周次            | 工作任務                     |
| --------------- | ---------------------------- |
| 第一周5/11~5/18 | 建立TCP模擬環境、設計Agent   |
| 第二周5/19~5/26 | 開始實驗，進行訓練與參數調整 |
| 第三周5/26~6/1  | 撰寫報告：結果分析、圖表繪製 |
|                 |                              |

**預期困難與解決方案**

問題：時間上難以設計完整可信的網路模擬環境(NS3, mini-net之類的)。

解決方案：自行設計一個簡易的網路環境模擬器或Gym。
