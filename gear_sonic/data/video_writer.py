import os
import queue
import sys
import threading

import av
import numpy as np


class VideoWriter:
    _STOP = object()
    _CANCEL = object()

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "h264",
        buffer_size: int = 50,
    ):
        self.output_path = output_path
        self._first_frame = True

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.queue = queue.Queue(maxsize=buffer_size)
        self.container = av.open(output_path, mode="w")
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.width = width
        self.stream.height = height
        self._worker_error: BaseException | None = None
        self._closed = False
        self._stop_lock = threading.Lock()
        self._thread = threading.Thread(target=self._writer_worker, daemon=True)
        self._thread.start()

    def _assert_dimensions(self, frame: np.ndarray) -> None:
        assert (
            frame.shape[1] == self.stream.width and frame.shape[0] == self.stream.height
        ), (
            f"Incorrect frame dimensions. Input dimensions: {frame.shape[1]}x{frame.shape[0]}. "
            f"Expected dimensions: {self.stream.width}x{self.stream.height}"
        )

    def add_frame(self, frame: np.ndarray) -> None:
        self._assert_dimensions(frame)
        if self._closed:
            raise RuntimeError("Cannot add a frame after the video writer has stopped")
        if self._worker_error is not None:
            raise RuntimeError("Video writer worker failed") from self._worker_error
        self.queue.put(frame)

    def _writer_worker(self) -> None:
        try:
            while True:
                frame = self.queue.get()
                if frame is self._STOP:
                    self._flush_stream()
                    break
                if frame is self._CANCEL:
                    break

                self._assert_dimensions(frame)
                frame = av.VideoFrame.from_ndarray(frame, format="rgb24")

                if self._first_frame:
                    stderr_fd = sys.stderr.fileno()
                    old_stderr = os.dup(stderr_fd)
                    devnull = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(devnull, stderr_fd)
                    try:
                        packets = self.stream.encode(frame)
                        for packet in packets:
                            self.container.mux(packet)
                    finally:
                        os.dup2(old_stderr, stderr_fd)
                        os.close(old_stderr)
                        os.close(devnull)
                        self._first_frame = False
                else:
                    packets = self.stream.encode(frame)
                    for packet in packets:
                        self.container.mux(packet)
        except BaseException as exc:
            self._worker_error = exc
        finally:
            try:
                self.container.close()
            except BaseException as exc:
                if self._worker_error is None:
                    self._worker_error = exc
            self._closed = True

    def _flush_stream(self) -> None:
        packets = self.stream.encode()
        for packet in packets:
            self.container.mux(packet)

    def stop(self) -> str:
        """Drain queued frames, then flush and close inside the writer thread."""
        with self._stop_lock:
            if not self._closed:
                print("Waiting for video writer to drain and flush...")
                self.queue.put(self._STOP)
                self._thread.join()
            if self._worker_error is not None:
                raise RuntimeError("Video writer worker failed") from self._worker_error
        return self.output_path

    def cancel(self) -> None:
        """Discard pending frames, stop the writer thread, and delete the output file."""
        with self._stop_lock:
            if not self._closed:
                while True:
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        break
                self.queue.put(self._CANCEL)
                self._thread.join()
            if os.path.exists(self.output_path):
                os.remove(self.output_path)

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            try:
                self.stop()
            except Exception:
                pass
