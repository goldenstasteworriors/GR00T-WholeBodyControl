"""Read-only live camera and NaVILA prediction dashboard. No motion publisher."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import subprocess
import threading
import time

from deploy.navila.bridge import Camera, infer, parse_action, sample_history
from deploy.navila.navigation_monitor import NavigationMonitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default="Walk to the table and stop in front of it.")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--observe", action="store_true", help="Observe the real navigation process; no inference")
    args = parser.parse_args()
    camera = Camera("192.168.123.164", 5555, "ego_view")
    lock = threading.Lock()
    stop = threading.Event()
    state = dict(app="navila-preview", instruction=args.instruction, execute=False,
                 mode="observe" if args.observe else "preview",
                 phase="Waiting for camera", latest=None, history=[], error=None)
    model_frame = b""
    html = Path(__file__).with_name("viewer.html").read_bytes()

    def observe():
        monitor = NavigationMonitor()
        while not stop.is_set():
            try:
                observed = monitor.read()
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                observed = dict(phase="Waiting for navigation interface", error=str(exc))
            with lock:
                state.update(observed)
            stop.wait(0.4)

    def predict():
        nonlocal model_frame
        history = []
        step = 0
        try:
            while not stop.is_set():
                if camera.error:
                    raise RuntimeError(camera.error)
                if camera.sample is None:
                    stop.wait(0.1)
                    continue
                encoded, shape, stamp = camera.snapshot()
                history.append(encoded)
                with lock:
                    state["phase"] = "Inferring"
                response, latency = infer("127.0.0.1", 54321, sample_history(history), args.instruction)
                action, amount, error = None, None, None
                try:
                    action, amount = parse_action(response)
                except ValueError as exc:
                    error = str(exc)
                step += 1
                item = dict(step=step, response=response, action=action, amount=amount,
                            parse_error=error, round_trip_s=latency, time=time.time(),
                            camera_timestamp=stamp, execute=False)
                with lock:
                    model_frame = base64.b64decode(encoded)
                    state.update(latest=item, phase="Waiting for next prediction")
                    state["history"] = ([item] + state["history"])[:20]
                print(json.dumps(item), flush=True)
                stop.wait(0.5)
        except Exception as exc:
            with lock:
                state.update(error=str(exc), phase="Prediction stopped")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            status = 200
            if path == "/":
                body, mime = html, "text/html; charset=utf-8"
            elif path == "/api/state":
                with lock:
                    body = json.dumps({**state, "camera_error": camera.error}).encode()
                mime = "application/json"
            elif path == "/api/frame":
                body = base64.b64decode(camera.snapshot()[0]) if camera.sample else b""
                mime = "image/jpeg"
                status = 200 if body else 503
            elif path == "/api/model-frame":
                with lock:
                    body = model_frame
                mime = "image/jpeg"
                status = 200 if body else 503
            else:
                body, mime, status = b"Not found", "text/plain", 404
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=observe if args.observe else predict, daemon=True).start()
    print(f"{'Execution monitor' if args.observe else 'Preview only'}: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        stop.set()
        server.server_close()
        camera.close()


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    try:
        main()
    except KeyboardInterrupt:
        print("Preview closed. No robot command was sent.", flush=True)
