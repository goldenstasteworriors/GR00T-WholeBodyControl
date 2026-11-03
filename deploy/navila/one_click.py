"""Tmux launchers. Enter initializes; Enter after ready starts navigation."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAYOUT = ROOT / ".runtime/navila/one_click_layout.json"
TP_SESSION = "navila-g1"
PC_SESSION = "navila-workstation"


def call(args, **kwargs):
    return subprocess.run(args, cwd=ROOT, **kwargs)


def tmux(*args):
    return call(["tmux", *args], check=True, text=True, capture_output=True).stdout.strip()


def has_session(name):
    return call(["tmux", "has-session", "-t", name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def command(*args):
    return shlex.join([str(arg) for arg in args])


def idle_command(label):
    return command("python3", HERE / "one_click.py", "idle", label)


def launch_pane(pane, label, cmd):
    dead = tmux("display-message", "-p", "-t", pane, "#{pane_dead}")
    title = tmux("display-message", "-p", "-t", pane, "#{pane_title}")
    # Idempotent launch: do not replace an already running process.
    if dead != "1" and not title.startswith("Waiting:"):
        print(f"{label} is already running; keeping the current process.", flush=True)
        return False
    tmux("select-pane", "-t", pane, "-T", label)
    tmux("respawn-pane", "-k", "-t", pane, cmd)
    return True


def attach(session):
    for key in ("DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR"):
        if key in os.environ:
            tmux("set-environment", "-t", session, key, os.environ[key])
    if sys.stdin.isatty() and sys.stdout.isatty():
        action = "switch-client" if os.environ.get("TMUX") else "attach-session"
        return call(["tmux", action, "-t", session]).returncode
    print(f"Started. Open the interface: tmux attach-session -t {session}")
    return 0


def start_thinkpad(args):
    if not has_session(TP_SESSION):
        previous = json.loads(LAYOUT.read_text()) if LAYOUT.exists() else {}
        menu = tmux("new-session", "-d", "-P", "-F", "#{pane_id}",
                    "-s", TP_SESSION, "-n", "run", "-x", "160", "-y", "45",
                    idle_command("Menu"))
        tmux("set-option", "-t", TP_SESSION, "remain-on-exit", "on")
        tmux("set-option", "-t", TP_SESSION, "mouse", "on")
        control = tmux("split-window", "-h", "-P", "-F", "#{pane_id}",
                       "-t", menu, idle_command("Control: waiting for START"))
        camera = tmux("split-window", "-v", "-P", "-F", "#{pane_id}",
                      "-t", control, idle_command("Camera: waiting for START"))
        navigation = tmux("split-window", "-v", "-P", "-F", "#{pane_id}",
                          "-t", camera, idle_command("Output: PREVIEW, LIVE, or GO"))
        panes = {"menu": menu, "control": control, "camera": camera, "navigation": navigation}
        for label, pane in panes.items():
            tmux("select-pane", "-t", pane, "-T", "Waiting:" + label)
        tmux("select-layout", "-t", TP_SESSION + ":run", "tiled")
        LAYOUT.parent.mkdir(parents=True, exist_ok=True)
        LAYOUT.write_text(json.dumps({"panes": panes,
                                     "instruction": args.instruction or previous.get("instruction", "Walk to the table and stop in front of it."),
                                     "steps": args.steps if args.steps is not None else previous.get("steps", 20)},
                                    ensure_ascii=False, indent=2) + "\n")
        launch_pane(menu, "Menu / e+Enter: ESTOP",
                    command("python3", HERE / "one_click.py", "console-thinkpad"))
        tmux("select-pane", "-t", menu)
    else:
        print("Reusing the existing interface; no duplicate controller will be started.", flush=True)
        layout = json.loads(LAYOUT.read_text())
        if args.instruction is not None:
            layout["instruction"] = args.instruction
        if args.steps is not None:
            layout["steps"] = args.steps
        LAYOUT.write_text(json.dumps(layout, ensure_ascii=False, indent=2) + "\n")
        print(f"Next launch instruction: {layout['instruction']} | Steps: {layout['steps']}", flush=True)
        print("An already running navigation process keeps its original instruction.", flush=True)
        menu = layout["panes"]["menu"]
        if tmux("display-message", "-p", "-t", menu, "#{pane_dead}") == "1":
            launch_pane(menu, "Menu / e+Enter: ESTOP",
                        command("python3", HERE / "one_click.py", "console-thinkpad"))
        tmux("select-pane", "-t", menu)
    return attach(TP_SESSION)


def start_workstation():
    relay = ["bash", str(HERE / "local_relay.sh")]
    result = call(relay + ["status"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode:
        if has_session("navila-g1-relay"):
            call(relay + ["stop"], check=True)
        call(relay + ["start"], check=True)
    if not has_session(PC_SESSION):
        tmux("new-session", "-d", "-s", PC_SESSION, "-n", "estop",
             command("python3", HERE / "one_click.py", "console-workstation"))
        tmux("set-option", "-t", PC_SESSION, "remain-on-exit", "on")
    elif tmux("display-message", "-p", "-t", PC_SESSION + ":estop.0", "#{pane_dead}") == "1":
        tmux("respawn-pane", "-k", "-t", PC_SESSION + ":estop.0",
             command("python3", HERE / "one_click.py", "console-workstation"))
    return attach(PC_SESSION)


def stop_from_menu(local):
    script = "estop_local.sh" if local else "estop_thinkpad.sh"
    result = call(["bash", str(HERE / script)], stdin=subprocess.DEVNULL)
    if result.returncode:
        print("ESTOP NOT CONFIRMED. See the result above.", flush=True)


def controller_state(pane):
    if tmux("display-message", "-p", "-t", pane, "#{pane_dead}") == "1":
        return "exited"
    output = tmux("capture-pane", "-p", "-J", "-t", pane, "-S", "-2000")
    for line in reversed(output.splitlines()):
        for state in ("ready", "running", "damped", "initializing"):
            if f"[CollectionMode] {state}" in line:
                return state
    return "initializing"


def exit_thinkpad():
    """Explicit user exit: interrupt only this launcher's worker panes."""
    if not has_session(TP_SESSION):
        print("ThinkPad launcher is already closed.", flush=True)
        return 0
    if not LAYOUT.exists():
        print("No launcher layout on this host. Run exit_thinkpad.sh on ThinkPad or exit_local.sh on the workstation.", flush=True)
        return 1
    panes = json.loads(LAYOUT.read_text())["panes"]
    active = []
    for key in ("navigation", "camera", "control"):
        pane = panes[key]
        if (tmux("display-message", "-p", "-t", pane, "#{pane_dead}") != "1"
                and not tmux("display-message", "-p", "-t", pane, "#{pane_title}").startswith("Waiting:")):
            tmux("send-keys", "-t", pane, "C-c")
            active.append(pane)
    # Give the original controller's SIGINT damping/shutdown path time to finish.
    end = time.monotonic() + 8
    while active and time.monotonic() < end:
        active = [p for p in active if tmux("display-message", "-p", "-t", p, "#{pane_dead}") != "1"]
        if active:
            time.sleep(0.1)
    if active:
        print(f"Exit still pending in panes {active}. Ctrl+C was sent; interface remains open.", flush=True)
        return 1
    print("Workers exited. Closing ThinkPad interface. Server and relay remain running.", flush=True)
    tmux("kill-session", "-t", TP_SESSION)
    return 0


def console_thinkpad():
    layout = json.loads(LAYOUT.read_text())
    panes = layout["panes"]
    confirmed = False
    print("ThinkPad launcher. PREVIEW works with the saved image while disconnected.", flush=True)
    print("Press Enter: initialize real joints. Wait for ready, then Enter: stand and navigate.", flush=True)
    print("Stay in this menu; no Control-pane key is required.", flush=True)
    print("PREVIEW/LIVE: output only | e: damping ESTOP | x or Ctrl+C: exit workers and interface", flush=True)
    print("s: pane status | q: detach (keep processes) | START/GO remain available", flush=True)
    print("Control pane: d for ESTOP. No added navigation watchdog or action caps.", flush=True)
    print(f"Instruction: {layout['instruction']}\nSteps: {layout['steps']}", flush=True)
    while True:
        try:
            choice = input("Enter: initialize/start | PREVIEW/LIVE/e/x/s/q > ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            return exit_thinkpad()
        # Read settings for each new launch, including reused interfaces.
        layout = json.loads(LAYOUT.read_text())
        if choice == "":
            state = controller_state(panes["control"]) if confirmed else "exited"
            choice = "START" if state in {"exited", "damped"} else "GO"
        if choice in {"PREVIEW", "LIVE"}:
            args = ["bash", HERE / "preview.sh", "--instruction", layout["instruction"]]
            if choice == "LIVE":
                args += ["--live", "--steps", layout["steps"]]
            if launch_pane(panes["navigation"], "Preview only / no control", command(*args)):
                print("Read-only output started. No controller or movement command is created.", flush=True)
        elif choice == "START":
            state = controller_state(panes["control"]) if confirmed else "exited"
            if state == "damped":
                tmux("send-keys", "-t", panes["control"], "]")
                print("Reinitializing. Wait for ready, then press Enter here.", flush=True)
                continue
            if confirmed and state != "exited":
                print(f"Controller state: {state}. Press Enter after ready to start navigation.", flush=True)
                continue
            confirmed = True
            tmux("clear-history", "-t", panes["control"])
            launch_pane(panes["control"], "Control / d: ESTOP",
                        command("bash", HERE / "run_control.sh", "--real"))
            launch_pane(panes["camera"], "Camera",
                        command("bash", HERE / "one_click_camera.sh"))
            print("Initializing. Wait for ready in Control, then press Enter HERE to stand and navigate.", flush=True)
            tmux("select-pane", "-t", panes["menu"])
        elif choice == "GO":
            if not confirmed:
                print("Press Enter once to initialize first.", flush=True)
                continue
            state = controller_state(panes["control"])
            if state == "exited":
                confirmed = False
                print("Controller exited. See Control error. Reconnect if needed; Enter retries initialization; x exits.", flush=True)
                continue
            if state not in {"ready", "running"}:
                print(f"Controller state: {state}. Wait for ready, then press Enter again.", flush=True)
                continue
            if state == "ready":
                tmux("send-keys", "-t", panes["control"], "]")
            print(f"Navigation instruction: {layout['instruction']}", flush=True)
            launch_pane(panes["navigation"], "NaVILA navigation",
                        command("bash", HERE / "run_bridge.sh", "--instruction",
                                layout["instruction"], "--execute", "--steps", layout["steps"]))
            print("Navigation requested. Menu e+Enter or Control d requests ESTOP.", flush=True)
        elif choice == "E":
            stop_from_menu(False)
        elif choice in {"X", "EXIT"}:
            return exit_thinkpad()
        elif choice == "S":
            print(tmux("list-panes", "-t", TP_SESSION + ":run",
                       "-F", "#{pane_index} #{pane_title} alive=#{==:#{pane_dead},0} #{pane_current_command}"), flush=True)
        elif choice == "Q":
            call(["tmux", "detach-client"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("Enter: initialize/start | PREVIEW/LIVE: output only | e: ESTOP | x: exit", flush=True)


def workstation_preview(live):
    panes = tmux("list-panes", "-t", PC_SESSION + ":estop", "-F", "#{pane_id}").splitlines()
    if len(panes) == 1:
        pane = tmux("split-window", "-h", "-P", "-F", "#{pane_id}", "-t", panes[0],
                    idle_command("Preview output"))
        tmux("select-pane", "-t", pane, "-T", "Waiting:preview")
    else:
        pane = panes[1]
    args = ["bash", HERE / "preview_local.sh"]
    if live:
        args += ["--live", "--steps", "20"]
    launch_pane(pane, "Preview only / no control", command(*args))
    tmux("select-pane", "-t", panes[0])


def console_workstation():
    print("Relay is running. Robot control requires confirmation on ThinkPad.", flush=True)
    print("p: saved-image preview | l: live preview | e: ESTOP | s: status | q: detach", flush=True)
    print("Local ESTOP requires SSH to ThinkPad; it does not pass through server02.", flush=True)
    while True:
        try:
            choice = input("p / l / e / s / q > ").strip().lower()
        except EOFError:
            return 0
        if choice in {"p", "l"}:
            workstation_preview(choice == "l")
        elif choice == "e":
            stop_from_menu(True)
        elif choice == "s":
            call(["bash", str(HERE / "local_relay.sh"), "status"])
        elif choice == "q":
            call(["tmux", "detach-client"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    tp = sub.add_parser("thinkpad")
    tp.add_argument("--instruction", default=None, help="Task for the next launch; omitted keeps the existing task")
    tp.add_argument("--steps", type=int, default=None, help="Action budget; omitted keeps the existing value (new session: 20)")
    for name in ("workstation", "console-thinkpad", "console-workstation", "exit-thinkpad"):
        sub.add_parser(name)
    idle = sub.add_parser("idle")
    idle.add_argument("label")
    args = parser.parse_args()
    if args.mode == "thinkpad":
        return start_thinkpad(args)
    if args.mode == "workstation":
        return start_workstation()
    if args.mode == "console-thinkpad":
        return console_thinkpad()
    if args.mode == "console-workstation":
        return console_workstation()
    if args.mode == "exit-thinkpad":
        return exit_thinkpad()
    print(args.label, flush=True)
    print("No robot process has been started in this pane.", flush=True)
    while True:
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
