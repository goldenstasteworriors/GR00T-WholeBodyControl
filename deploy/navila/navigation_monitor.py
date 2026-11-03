"""Observe actual bridge stdout in its existing tmux pane. Never infer or publish."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def execution_state(text, instruction, dead=False, waiting=False, started=0):
    records = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(item, dict) and item.get("time", 0) >= started and "event" in item:
            records.append(item)
    predictions = [item for item in records if item["event"] == "inference" and item.get("execute") is True]
    history = [{**item, "camera_timestamp": item.get("camera_source_timestamp"), "parse_error": None}
               for item in reversed(predictions[-20:])]
    phase = "Waiting for real navigation" if waiting else "Starting navigation"
    segment = None
    error = None
    for item in records:
        event = item["event"]
        if event == "waiting_for_manual_standing":
            phase = "Waiting for standing"
        elif event == "inference":
            phase = "Prediction received"
            segment = None
        elif event == "segment":
            phase = "Movement segment issued"
            segment = item
        elif event == "error":
            phase, error = "Navigation error", item.get("error")
        elif event == "interrupted":
            phase = "Navigation interrupted"
    if dead and not error:
        phase = "Navigation process exited"
    return dict(app="navila-preview", mode="observe", execute=True, instruction=instruction,
                phase=phase, latest=history[0] if history else None, history=history,
                segment=segment, error=error, navigation_alive=not dead and not waiting,
                source="Actual execution process stdout; this viewer only observes")


class NavigationMonitor:
    def __init__(self):
        self.pid = None
        self.started = 0

    def read(self):
        layout = json.loads((ROOT / ".runtime/navila/one_click_layout.json").read_text())
        pane = layout["panes"]["navigation"]
        def tmux(*args):
            return subprocess.check_output(["tmux", *args], text=True, stderr=subprocess.DEVNULL)
        pid, dead, title = tmux("display-message", "-p", "-t", pane,
                                "#{pane_pid}|#{pane_dead}|#{pane_title}").strip().split("|", 2)
        if pid != self.pid:
            self.pid, self.started = pid, 0
            try:
                ticks = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
                boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                            if line.startswith("btime "))
                self.started = boot + ticks / os.sysconf("SC_CLK_TCK")
            except (OSError, ValueError, StopIteration):
                pass
        waiting = title.startswith("Waiting:") or "Preview" in title
        text = "" if waiting else tmux("capture-pane", "-p", "-J", "-t", pane, "-S", "-2000")
        return execution_state(text, layout["instruction"], dead == "1", waiting, self.started)
