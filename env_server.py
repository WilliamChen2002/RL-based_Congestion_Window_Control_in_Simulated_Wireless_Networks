# env_server.py
import asyncio
import json
import uuid
import random
import numpy as np

# 使用你原本的 TCPEnv
from environment import TCPEnv  # ← 請確認 import 路徑正確

# 儲存多個獨立環境
sessions: dict[str, TCPEnv] = {}


async def handle_client(reader, writer):
    addr = writer.get_extra_info("peername")

    try:
        while True:
            data = await reader.readline()
            if not data:
                break

            request = json.loads(data.decode().strip())
            command = request.get("command")

            if command == "create":
                mode = request.get("mode", "agent")
                seed = request.get("seed", random.randint(1, 100000))

                # 固定隨機種子，讓比較更公平
                random.seed(seed)
                np.random.seed(seed)

                session_id = str(uuid.uuid4())
                sessions[session_id] = TCPEnv(mode=mode)  # 使用你原本的 TCPEnv

                response = {
                    "status": "ok",
                    "command": "create",
                    "session_id": session_id,
                    "mode": mode,
                    "seed": seed,
                }

            elif command == "reset":
                session_id = request.get("session_id")
                env = sessions.get(session_id)
                if env:
                    env.reset()  # 使用你原本的 reset
                    response = {
                        "status": "ok",
                        "command": "reset",
                        "state": env.get_state().tolist(),
                    }
                else:
                    response = {"status": "error", "message": "Session not found"}

            elif command == "step":
                session_id = request.get("session_id")
                action = request.get("action")
                env = sessions.get(session_id)

                if env:
                    # 使用你原本的 step 方法
                    next_state, reward, done, info = env.step(
                        action=action,
                        cwnd=request.get("cwnd"),
                        coef=request.get("coef"),
                    )
                    response = {
                        "status": "ok",
                        "command": "step",
                        "state": next_state.tolist(),
                        "reward": reward,
                        "done": done,
                        "info": info,
                    }
                else:
                    response = {"status": "error", "message": "Session not found"}

            else:
                response = {"status": "error", "message": "Unknown command"}

            writer.write((json.dumps(response) + "\n").encode())
            await writer.drain()

    finally:
        writer.close()
        await writer.wait_closed()


async def main():
    server = await asyncio.start_server(handle_client, "127.0.0.1", 8888)
    print("🚀 TCP Environment Server 已啟動")
    print("使用獨立 TCPEnv 實例（每個 mode 有自己的 Router）")
    print("隨機種子固定 → 公平比較\n")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
