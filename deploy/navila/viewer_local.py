"""Start the read-only ThinkPad dashboard and forward it to this workstation."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8767"


def current():
    try:
        with urlopen(URL + "/api/state", timeout=1) as response:
            state = json.load(response)
        return state if state.get("app") == "navila-preview" else None
    except (OSError, ValueError):
        return None


def show(state):
    print(f"Preview only: {URL}\nInstruction: {state['instruction']}", flush=True)
    subprocess.Popen(["xdg-open", URL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default="Walk to the table and stop in front of it.")
    args = parser.parse_args()
    state = current()
    if state:
        print("Reusing the running preview; its instruction is shown below.", flush=True)
        show(state)
        return 0
    subprocess.run(["bash", str(HERE / "local_relay.sh"), "start"], check=True)
    remote = shlex.join(["bash", "/home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/viewer.sh", "--instruction", args.instruction])
    child = subprocess.Popen(["ssh", "-tt", "-o", "ExitOnForwardFailure=yes", "-o", "BatchMode=yes", "-L", "127.0.0.1:8767:127.0.0.1:8767", "--", "thinkpad", remote])
    try:
        for _ in range(60):
            if child.poll() is not None:
                return child.returncode or 1
            state = current()
            if state:
                show(state)
                print("Keep this terminal open. Ctrl+C closes the preview only.", flush=True)
                return child.wait()
            time.sleep(0.5)
        print("Preview did not become available. See SSH output above.", flush=True)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
