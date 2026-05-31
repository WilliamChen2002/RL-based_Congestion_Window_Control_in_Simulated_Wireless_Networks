import asyncio
import json
import random
import uuid

from environment import TCPEnv

sessions = {}


def normalize_info(info, state):

    return {
        "throughput": float(info.get("network", {}).get("throughput", 0)),
        "loss_rate": float(info.get("network", {}).get("loss_rate", 0)),
        "rtt": float(info.get("network", {}).get("rtt", 0)),
        "bandwidth": float(info.get("network", {}).get("bandwidth", 0)),
        "cwnd": float(info.get("sender", {}).get("cwnd", 0)),
        "aoi": float(info.get("receiver", {}).get("aoi", 0)),
        "queue": int(info.get("router", {}).get("queue_size", 0)),
    }


async def handle_client(reader, writer):

    try:
        while True:
            data = await reader.readline()
            if not data:
                break

            request = json.loads(data.decode().strip())
            cmd = request.get("command")

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

            elif cmd == "step":
                session_id = request.get("session_id")
                action = request.get("action")
                cwnd = request.get("cwnd")  # DDPG 用，直接設定目標 cwnd
                aoi_request = request.get("aoi_request")

                env = sessions.get(session_id)

                if env:
                    state, reward, terminated, truncated, info = env.step(
                        action, cwnd, aoi_request
                    )

                    response = {
                        "status": "ok",
                        "state": state.tolist(),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "done": bool(terminated or truncated),
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


async def main():

    server = await asyncio.start_server(handle_client, "0.0.0.0", 8888)

    print("Environment Server Start.")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
