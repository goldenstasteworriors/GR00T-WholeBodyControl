"""Manual SIGUSR1 damping request and controller acknowledgement. Standard library only."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time


def process_start(pid):
    # /proc stat field 22; comm may contain spaces/parentheses.
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + f".{os.getpid()}.pending")
    pending.write_text(json.dumps(data, indent=2) + "\n")
    pending.replace(path)


def register_controller(directory):
    registration = {"pid": os.getpid(), "start_ticks": process_start(os.getpid()),
                    "project": str(Path.cwd().resolve())}
    write_json(Path(directory) / "controller.json", registration)
    return registration


def acknowledge(directory, registration, diagnostic_read_only):
    write_json(Path(directory) / "ack.json", {
        **registration, "ack_monotonic_ns": time.monotonic_ns(),
        "state": "damping_latched", "diagnostic_read_only": diagnostic_read_only,
        "scope": "read-only test; no actuator writes" if diagnostic_read_only
                 else "controller damping requested; not measured physical standstill"})


def request_stop(directory, wait=3.0):
    directory = Path(directory)
    registration = json.loads((directory / "controller.json").read_text())
    pid = registration["pid"]
    # These checks select the intended process; they do not gate navigation.
    fd = os.pidfd_open(pid)
    try:
        if process_start(pid) != registration["start_ticks"]:
            raise RuntimeError("Controller registration is stale; PID belongs to a different process")
        if str(Path(f"/proc/{pid}/cwd").resolve()) != registration["project"]:
            raise RuntimeError("Registered process no longer belongs to this project")
        sent = time.monotonic_ns()
        signal.pidfd_send_signal(fd, signal.SIGUSR1)
    finally:
        os.close(fd)
    end = time.monotonic() + wait
    while time.monotonic() < end:
        try:
            ack = json.loads((directory / "ack.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            ack = {}
        if (ack.get("pid") == pid and ack.get("start_ticks") == registration["start_ticks"]
                and ack.get("ack_monotonic_ns", 0) >= sent):
            print(json.dumps(ack), flush=True)
            return ack
        time.sleep(0.01)
    raise TimeoutError("SIGUSR1 sent, but no controller acknowledgement; stop is not confirmed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / ".runtime/navila/estop")
    args = parser.parse_args()
    try:
        request_stop(args.state_dir)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ESTOP NOT CONFIRMED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
