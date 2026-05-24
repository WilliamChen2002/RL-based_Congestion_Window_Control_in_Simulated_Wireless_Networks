import asyncio
import json
import random
import uuid

from environment import TCPEnv

sessions = {}


# =========================================================
# SAFE INFO NORMALIZER（關鍵）
# =========================================================
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


# =========================================================
# CLIENT HANDLER
# =========================================================
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

            # ================= RESET =================
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

            # ================= STEP =================
            elif cmd == "step":
                session_id = request.get("session_id")
                action = request.get("action")

                env = sessions.get(session_id)

                if env:
                    state, reward, terminated, truncated, info = env.step(action)

                    response = {
                        "status": "ok",
                        "state": state.tolist(),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        # 🔥 KEY FIX: stable schema
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


# =========================================================
# MAIN
# =========================================================
async def main():

    server = await asyncio.start_server(handle_client, "127.0.0.1", 8888)

    print("🚀 TCP RL Env Server running (stable schema enabled)")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
