"""Receive the existing three-view composed-camera protocol without hardware drivers."""
import base64
from collections import deque
import threading
import time

import cv2
import msgpack
import msgpack_numpy
import numpy as np
import zmq

from contract import CAMERAS


class Cameras:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.frames = deque(maxlen=256)
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.error = None
        self.capture_times = None
        self.last_advance = np.zeros(3)
        self.thread = threading.Thread(target=self._receive, name="camera-input")
        self.thread.start()

    def _receive(self):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.endpoint)
        try:
            while not self.stopped.is_set():
                if not socket.poll(100):
                    continue
                payload = msgpack.unpackb(socket.recv(), object_hook=msgpack_numpy.decode)
                pictures, times = [], []
                for name in CAMERAS:
                    image = payload["images"][name]
                    if isinstance(image, str):
                        image = base64.b64decode(image)
                    if isinstance(image, (bytes, bytearray)):
                        image = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if image is None:
                            raise ValueError("Camera JPEG could not be decoded")
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = np.asarray(image)
                    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                        raise ValueError("Camera is not RGB uint8")
                    pictures.append(cv2.resize(image, (240, 240), interpolation=cv2.INTER_AREA).transpose(2, 0, 1))
                    times.append(float(payload["timestamps"][name]))
                times = np.asarray(times)
                if not np.isfinite(times).all():
                    raise ValueError("Invalid camera capture times")
                received = time.monotonic()
                with self.lock:
                    advanced = np.ones(3, dtype=bool) if self.capture_times is None else times > self.capture_times
                    if self.capture_times is not None and np.any(times < self.capture_times - 1e-5):
                        raise ValueError("Camera capture clock moved backwards")
                    self.last_advance[advanced] = received
                    self.capture_times = times.copy()
                    if advanced.any():
                        self.frames.append((received, np.stack(pictures), times - times.max()))
                    self.error = None
        except Exception as error:
            self.error = str(error)
        finally:
            socket.close()
            context.term()

    def history(self, max_age=.25):
        with self.lock:
            frames = list(self.frames)
            last_advance = self.last_advance.copy()
        if self.error:
            raise RuntimeError(self.error)
        if not frames or time.monotonic() - min(last_advance) > max_age:
            raise TimeoutError("Waiting for three fresh cameras")
        received_times = np.asarray([frame[0] for frame in frames])
        targets = received_times[-1] - np.array([24, 16, 8, 0]) / 30
        indices = np.unique(np.maximum(0, np.searchsorted(received_times, targets, side="right") - 1))
        pictures = np.stack([frames[i][1] for i in indices])
        # Sample causal 30Hz anchors by time, so packet loss does not compress history.
        ages = np.stack([frames[i][0] - received_times[-1] + frames[i][2] for i in indices]).astype(np.float32)
        return pictures, ages

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=2)
