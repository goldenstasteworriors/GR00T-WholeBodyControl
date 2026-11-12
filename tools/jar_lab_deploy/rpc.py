"""Bounded ZMQ multipart wire protocol; no pickle or executable deserialization."""
import json
import time

import numpy as np
import zmq


def encode_request(state, images, ages, steps=1):
    arrays = [np.asarray(state, dtype="<f4"), np.asarray(images, dtype=np.uint8), np.asarray(ages, dtype="<f4")]
    header = dict(version=1, steps=steps, shapes=[list(a.shape) for a in arrays])
    return [json.dumps(header).encode()] + [np.ascontiguousarray(a).tobytes() for a in arrays]


def decode_request(parts):
    if len(parts) != 4 or sum(map(len, parts)) > 9 * 1024 * 1024:
        raise ValueError("Invalid multipart count/size")
    header = json.loads(parts[0])
    shapes = header["shapes"]
    if (header["version"] != 1 or len(shapes) != 3 or shapes[0] != [16]
            or len(shapes[1]) != 5 or shapes[1][1:] != [3, 3, 240, 240]
            or not 1 <= shapes[1][0] <= 4 or shapes[2] != [shapes[1][0], 3]):
        raise ValueError("Invalid request schema")
    arrays = [np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
              for payload, dtype, shape in zip(parts[1:], ("<f4", "u1", "<f4"), shapes)]
    return arrays, header["steps"]


class Client:
    def __init__(self, endpoint="tcp://127.0.0.1:5560", timeout_ms=15000):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(endpoint)

    def predict(self, state, images, ages, steps=1):
        started = time.monotonic()
        self.socket.send_multipart(encode_request(state, images, ages, steps))
        parts = self.socket.recv_multipart()
        header = json.loads(parts[0])
        if not header.get("ok"):
            raise RuntimeError(header.get("error", "Policy server failed"))
        positions = np.frombuffer(parts[1], dtype="<f4").reshape(header["shape"]).copy()
        if positions.shape != (steps, 16) or not np.isfinite(positions).all():
            raise ValueError("Invalid response")
        return dict(positions=positions, round_trip_seconds=time.monotonic() - started, **header)

    def status(self):
        self.socket.send_json(dict(type="status",version=1))
        result = self.socket.recv_json()
        if not result.get("ok"):
            raise RuntimeError(result.get("error","Policy not ready"))
        return result

    def close(self):
        self.socket.close()
        self.context.term()
