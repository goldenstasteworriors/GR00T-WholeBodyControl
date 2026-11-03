"""Dashboard-only camera reconnection. No dependency on motion execution."""
import time

from deploy.navila.bridge import Camera


class ViewerCamera(Camera):
    def __init__(self, host, port, key):
        self.frame_count = 0
        self.last_received = None
        self.reconnects = 0
        super().__init__(host, port, key)

    def run(self):
        from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
        while not self.stop.is_set():
            client = None
            connected = time.monotonic()
            try:
                client = ComposedCameraClientSensor(self.host, self.port)
                while not self.stop.is_set():
                    item = client.read(blocking=False, new_only=True)
                    frame = item.get("images", {}).get(self.key) if item else None
                    if frame is not None:
                        self.sample = (frame.copy(), item.get("timestamps", {}).get(self.key))
                        self.last_received = time.monotonic()
                        self.frame_count += 1
                        self.error = None
                        connected = self.last_received
                    elif time.monotonic() - connected > 3:
                        self.error = "Camera disconnected; reconnecting video subscription"
                        self.sample = None
                        break
                    self.stop.wait(0.005)
            except Exception as exc:
                self.error = str(exc)
                self.sample = None
            finally:
                if client is not None:
                    client.close()
            if not self.stop.is_set():
                self.reconnects += 1
                self.stop.wait(0.2)

    def status(self):
        return dict(camera_frames=self.frame_count,
                    camera_frame_age_s=None if self.last_received is None else time.monotonic() - self.last_received,
                    camera_reconnects=self.reconnects,
                    camera_connected=self.sample is not None and self.error is None)
