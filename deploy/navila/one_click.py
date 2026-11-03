"""Tmux launchers. No robot process starts before explicit START."""
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
        print(f"{label} 已在运行；保留当前进程。", flush=True)
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
    print(f"已启动；进入界面：tmux attach-session -t {session}")
    return 0


def start_thinkpad(args):
    if not has_session(TP_SESSION):
        menu = tmux("new-session", "-d", "-P", "-F", "#{pane_id}",
                    "-s", TP_SESSION, "-n", "run", "-x", "160", "-y", "45",
                    idle_command("菜单"))
        tmux("set-option", "-t", TP_SESSION, "remain-on-exit", "on")
        tmux("set-option", "-t", TP_SESSION, "mouse", "on")
        control = tmux("split-window", "-h", "-P", "-F", "#{pane_id}",
                       "-t", menu, idle_command("控制器：等待 START"))
        camera = tmux("split-window", "-v", "-P", "-F", "#{pane_id}",
                      "-t", control, idle_command("相机：等待 START"))
        navigation = tmux("split-window", "-v", "-P", "-F", "#{pane_id}",
                          "-t", camera, idle_command("推理：等待 GO"))
        panes = {"menu": menu, "control": control, "camera": camera, "navigation": navigation}
        for label, pane in panes.items():
            tmux("select-pane", "-t", pane, "-T", "Waiting:" + label)
        tmux("select-layout", "-t", TP_SESSION + ":run", "tiled")
        LAYOUT.parent.mkdir(parents=True, exist_ok=True)
        LAYOUT.write_text(json.dumps({"panes": panes, "instruction": args.instruction,
                                     "steps": args.steps}, ensure_ascii=False, indent=2) + "\n")
        launch_pane(menu, "菜单 / e 回车急停",
                    command("python3", HERE / "one_click.py", "console-thinkpad"))
        tmux("select-pane", "-t", menu)
    else:
        print("复用已有界面；不会重复启动控制器。", flush=True)
        menu = json.loads(LAYOUT.read_text())["panes"]["menu"]
        if tmux("display-message", "-p", "-t", menu, "#{pane_dead}") == "1":
            launch_pane(menu, "菜单 / e 回车急停",
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
    result = call(["bash", str(HERE / script)])
    if result.returncode:
        print("急停未确认；请查看上方结果。", flush=True)


def console_thinkpad():
    layout = json.loads(LAYOUT.read_text())
    panes = layout["panes"]
    confirmed = False
    print("ThinkPad 一键界面（机器人可暂不连接）", flush=True)
    print("未确认前，不启动控制、相机或导航进程。", flush=True)
    print("接回机器人后：START 确认初始化 → 控制窗 ready 后按 ] → 菜单 GO 导航。", flush=True)
    print("START 会启动真实关节初始化；不是只读模式。", flush=True)
    print("e 回车：阻尼急停 | s：窗口状态 | q：脱离界面（保留进程）", flush=True)
    print("控制窗 d 也是原急停；这里没有新增超时/限幅。", flush=True)
    print(f"任务：{layout['instruction']}\n最多段数：{layout['steps']}", flush=True)
    while True:
        try:
            choice = input("START / GO / e / s / q > ").strip().upper()
        except EOFError:
            return 0
        if choice == "START":
            confirmed = True
            launch_pane(panes["control"], "控制器 / d 急停",
                        command("bash", HERE / "run_control.sh", "--real"))
            launch_pane(panes["camera"], "相机",
                        command("bash", HERE / "one_click_camera.sh"))
            print("右上控制窗：等待 ready，按 ] 站立；左上菜单：GO 导航。", flush=True)
            tmux("select-pane", "-t", panes["control"])
        elif choice == "GO":
            if not confirmed:
                print("尚未确认初始化，请先输入 START。", flush=True)
                continue
            launch_pane(panes["navigation"], "NaVILA 导航",
                        command("bash", HERE / "run_bridge.sh", "--instruction",
                                layout["instruction"], "--execute", "--steps", layout["steps"]))
            print("导航已启动；菜单 e 回车或控制窗 d 可急停。", flush=True)
        elif choice == "E":
            stop_from_menu(False)
        elif choice == "S":
            print(tmux("list-panes", "-t", TP_SESSION + ":run",
                       "-F", "#{pane_index} #{pane_title} alive=#{==:#{pane_dead},0} #{pane_current_command}"), flush=True)
        elif choice == "Q":
            call(["tmux", "detach-client"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print("输入 START 或 GO 才会启动相应进程。", flush=True)


def console_workstation():
    print("本机中转已启动；机器人由 ThinkPad 确认后启动。", flush=True)
    print("e 回车：SSH 阻尼急停 | s：中转状态 | q：脱离界面（保留中转）", flush=True)
    print("本机急停需 SSH 连通；不经过 server02。", flush=True)
    while True:
        try:
            choice = input("e / s / q > ").strip().lower()
        except EOFError:
            return 0
        if choice == "e":
            stop_from_menu(True)
        elif choice == "s":
            call(["bash", str(HERE / "local_relay.sh"), "status"])
        elif choice == "q":
            call(["tmux", "detach-client"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    tp = sub.add_parser("thinkpad")
    tp.add_argument("--instruction", default="Find the beaker and stop in front of it.")
    tp.add_argument("--steps", type=int, default=20)
    for name in ("workstation", "console-thinkpad", "console-workstation"):
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
    print(args.label, flush=True)
    print("此窗口尚未启动机器人进程。", flush=True)
    while True:
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
