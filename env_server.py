import asyncio
import json
import random
import uuid

from environment import TCPEnv

sessions = {}


<<<<<<< HEAD
# =========================================================
# SAFE INFO NORMALIZER
# =========================================================
=======
>>>>>>> main
def normalize_info(info, state):

    return {
        # ===== network =====
        "throughput": float(info.get("network", {}).get("throughput", 0)),
        "loss_rate": float(info.get("network", {}).get("loss_rate", 0)),
        "rtt": float(info.get("network", {}).get("rtt", 0)),
        "bandwidth": float(info.get("network", {}).get("bandwidth", 0)),
        # ===== sender =====
        "cwnd": float(info.get("sender", {}).get("cwnd", 0)),
        # ===== receiver =====
        "aoi": float(info.get("receiver", {}).get("aoi", 0)),
        # ===== router =====
        "queue": int(info.get("router", {}).get("queue_size", 0)),
    }


<<<<<<< HEAD
# =========================================================
# CLIENT HANDLER
# =========================================================
=======
>>>>>>> main
async def handle_client(reader, writer):

    try:
        while True:
            data = await reader.readline()
            if not data:
                break

            request = json.loads(data.decode().strip())
            cmd = request.get("command")

            # ================= CREATE =================
            if cmd == "create":
                mode = request.get("mode", "agent")
                seed = request.get("seed", random.randint(0, 99999))

                session_id = str(uuid.uuid4())

                env = TCPEnv(mode=mode, seed=seed)
                sessions[session_id] = env

                response = {
                    "status": "ok",
                    "session_id": session_id,
                    "mode": mode,
                    "seed": seed,
                }

<<<<<<< HEAD
            # ================= RESET =================
=======
>>>>>>> main
            elif cmd == "reset":
                session_id = request.get("session_id")
                env = sessions.get(session_id)

                if env:
                    state, info = env.reset()

                    response = {
                        "status": "ok",
                        "state": state.tolist(),
                        "info": normalize_info(info, state),
                    }

                else:
                    response = {"status": "error", "msg": "no session"}

<<<<<<< HEAD
            # ================= STEP =================
            elif cmd == "step":
                session_id = request.get("session_id")
                action = request.get("action")
                cwnd = request.get("cwnd")   # DDPG 用，直接設定目標 cwnd

                env = sessions.get(session_id)

                if env:
                    # DDPG 傳來的 cwnd 直接設定到 sender，再讓環境跑一步
                    if cwnd is not None:
                        env.sender.cwnd = float(cwnd)

                    state, reward, terminated, truncated, info = env.step(action)
=======
            elif cmd == "step":
                session_id = request.get("session_id")
                action = request.get("action")
                cwnd = request.get("cwnd")
                env = sessions.get(session_id)

                if env:
                    state, reward, terminated, truncated, info = env.step(action, cwnd)
>>>>>>> main

                    response = {
                        "status": "ok",
                        "state": state.tolist(),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
<<<<<<< HEAD
                        "done": bool(terminated or truncated),
=======
>>>>>>> main
                        "info": normalize_info(info, state),
                    }

                else:
                    response = {"status": "error", "msg": "no session"}

            else:
                response = {"status": "error", "msg": "unknown command"}

            writer.write((json.dumps(response) + "\n").encode())
            await writer.drain()

    finally:
        writer.close()
        await writer.wait_closed()


<<<<<<< HEAD
# =========================================================
# MAIN
# =========================================================
=======
>>>>>>> main
async def main():

    server = await asyncio.start_server(handle_client, "127.0.0.1", 8888)

<<<<<<< HEAD
    print("🚀 TCP RL Env Server running (stable schema enabled)")
=======
    print("TCP RL Env Server running (stable schema enabled)")
>>>>>>> main

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
