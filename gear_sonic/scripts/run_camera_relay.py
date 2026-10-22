"""Fan out one robot camera connection on loopback without decoding images."""

from dataclasses import dataclass
import time

import tyro
import zmq


@dataclass
class CameraRelayConfig:
    camera_host: str = "192.168.123.164"
    camera_port: int = 5555
    local_port: int = 5558


def main(config: CameraRelayConfig) -> None:
    if not 1 <= config.camera_port <= 65535 or not 1 <= config.local_port <= 65535:
        raise ValueError("Camera ports must be in [1, 65535]")
    context = zmq.Context()
    upstream = context.socket(zmq.SUB)
    downstream = context.socket(zmq.PUB)
    upstream.setsockopt(zmq.SUBSCRIBE, b"")
    # Recording needs every group, including short bursts after USB/network
    # jitter. Preview subscribers can still conflate their local copy.
    upstream.setsockopt(zmq.RCVHWM, 3)
    downstream.setsockopt(zmq.SNDHWM, 3)
    upstream.setsockopt(zmq.LINGER, 0)
    downstream.setsockopt(zmq.LINGER, 0)
    try:
        downstream.bind(f"tcp://127.0.0.1:{config.local_port}")
        upstream.connect(f"tcp://{config.camera_host}:{config.camera_port}")
        print(f"Camera relay: {config.camera_host}:{config.camera_port} -> "
              f"127.0.0.1:{config.local_port}; original payloads unchanged", flush=True)
        count = 0
        received_bytes = 0
        last_log = time.monotonic()
        last_frame = None
        while True:
            if upstream.poll(100):
                payload = upstream.recv()
                downstream.send(payload)
                count += 1
                received_bytes += len(payload)
                last_frame = time.monotonic()
            now = time.monotonic()
            if now - last_log >= 2:
                age = "missing" if last_frame is None else f"{now - last_frame:.3f}s"
                print(f"[CameraRelay] frames={count} "
                      f"input={received_bytes * 8 / (now - last_log) / 1e6:.2f}Mbps "
                      f"receive_age={age}", flush=True)
                count = received_bytes = 0
                last_log = now
    except KeyboardInterrupt:
        print("Camera relay stopped", flush=True)
    finally:
        upstream.close()
        downstream.close()
        context.term()


if __name__ == "__main__":
    main(tyro.cli(CameraRelayConfig))
