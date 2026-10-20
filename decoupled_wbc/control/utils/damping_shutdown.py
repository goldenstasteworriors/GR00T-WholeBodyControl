"""Explicit damping shutdown shared by keyboard, signals and normal exit."""

import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


class DampingShutdown:
    def __init__(self, env):
        self.env = env
        self.requested = threading.Event()
        self.reason = "control_exit"
        self.session = os.environ.get("DECOUPLED_WBC_TMUX_SESSION")

    def request(self, reason):
        if not self.requested.is_set():
            self.reason = reason
            self.requested.set()

    def install_signals(self):
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1):
            signal.signal(sig, lambda number, frame: self.request(f"signal:{number}"))
        if self.session:
            subprocess.run([
                "tmux", "set-environment", "-t", self.session,
                "DECOUPLED_WBC_CONTROL_PID", str(os.getpid()),
            ], check=True)

    def finish(self):
        started = time.monotonic()
        report = {"reason": self.reason, "pid": os.getpid(), "confirmed": False}
        try:
            self.env.damping_stop()
            report["confirmed"] = True
        except Exception as error:
            report["error"] = str(error)
        report["elapsed_s"] = time.monotonic() - started
        path = Path(__file__).resolve().parents[3] / ".runtime" / "collection_damping_stop.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        if self.session:
            subprocess.run(["tmux", "kill-session", "-t", self.session], capture_output=True)
        return report
