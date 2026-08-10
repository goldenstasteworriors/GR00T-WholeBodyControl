import os
import queue
import sys
import threading

import av
import numpy as np


class VideoWriter:
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
        self._closed = False
        self._container_closed = False
        self._worker_error: BaseException | None = None
        self._state_lock = threading.Lock()
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
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Can't add a frame to a closed video writer")
            if self._worker_error is not None:
                raise RuntimeError("Video writer worker failed") from self._worker_error
            self._assert_dimensions(frame)
            self.queue.put(frame)

    def _writer_worker(self) -> None:
        while True:
            frame = self.queue.get()
            try:
                if frame is None:
                    return
                if self._worker_error is not None:
                    continue

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
                self.queue.task_done()

    def _flush_stream(self) -> None:
        packets = self.stream.encode()
        for packet in packets:
            self.container.mux(packet)

    def _close_container(self) -> None:
        if not self._container_closed:
            self.container.close()
            self._container_closed = True
            self.stream = None
            self.container = None

    def _stop_worker(self, discard_pending: bool = False) -> bool:
        with self._state_lock:
            if self._closed:
                return False
            self._closed = True

            if discard_pending:
                while True:
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        break
                    else:
                        self.queue.task_done()

            self.queue.put(None)

        self.queue.join()
        self._thread.join()
        return True

    def stop(self) -> str:
        """Blocking call. Waits for queue to drain, flushes, and closes the container."""
        if not self._stop_worker():
            return self.output_path
        try:
            if self._worker_error is not None:
                raise RuntimeError("Video writer worker failed") from self._worker_error
            print("Video writer queue is empty, flushing stream...")
            self._flush_stream()
        finally:
            self._close_container()
        return self.output_path

    def cancel(self) -> None:
        """Immediately stops writing and deletes the output file."""
        self._stop_worker(discard_pending=True)
        self._close_container()
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def __del__(self) -> None:
        try:
            self._close_container()
        except Exception:
            pass
