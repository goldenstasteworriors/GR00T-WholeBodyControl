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
    label = "Execution monitor" if state.get("mode") == "observe" else "Preview only"
    print(f"{label}: {URL}\nInstruction: {state['instruction']}", flush=True)
    subprocess.Popen(["xdg-open", URL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    global URL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default="Walk to the table and stop in front of it.")
    parser.add_argument("--observe", action="store_true", help="Observe actual execution without making predictions")
    args = parser.parse_args()
    port = 8768 if args.observe else 8767
    URL = f"http://127.0.0.1:{port}"
    state = current()
    if state:
        print("Reusing the running dashboard; its instruction is shown below.", flush=True)
        show(state)
        return 0
    if not args.observe:
        subprocess.run(["bash", str(HERE / "local_relay.sh"), "start"], check=True)
    remote_args = ["bash", "/home/data-collection/GR00T-WholeBodyControl-worktrees/navila-g1-link/deploy/navila/viewer.sh", "--instruction", args.instruction, "--port", str(port)]
    if args.observe:
        remote_args.append("--observe")
    remote = shlex.join(remote_args)
    child = subprocess.Popen(["ssh", "-tt", "-o", "ExitOnForwardFailure=yes", "-o", "BatchMode=yes", "-L", f"127.0.0.1:{port}:127.0.0.1:{port}", "--", "thinkpad", remote])
    try:
        for _ in range(60):
            if child.poll() is not None:
                return child.returncode or 1
            state = current()
            if state:
                show(state)
                print("Keep this terminal open. Ctrl+C closes this dashboard only; robot control is unaffected.", flush=True)
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
