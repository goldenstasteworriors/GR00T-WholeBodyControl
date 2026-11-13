"""Observable, resumable sequential operation queue; run under supervisord."""
import argparse
import atexit
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


TASKS = ["TAKE_TUBE", "TAKE_DROPPER_AND_ASPIRATE", "DISPENSE_TO_HELD_TUBE",
         "RETURN_DROPPER", "RETURN_TUBE_TO_RACK", "PLACE_TUBE_IN_BATH"]
STOP = False
CHILD = None
PAUSED_KEEPALIVES = []


def project_tmp(root):
    root = Path(root).resolve()
    path = root/"runtime/tmp"
    if len(str(path).encode()) > 65:
        # Python multiprocessing appends pymp-*/listener-* Unix socket names.
        path = root.parent/("t_"+hashlib.sha256(str(root).encode()).hexdigest()[:6])
    if len(str(path).encode()) > 65:
        raise ValueError("Project temporary path is too long for multiprocessing Unix sockets")
    return path


def process_start(pid):
    return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]


def restore_keepalives():
    for pid, started in PAUSED_KEEPALIVES:
        try:
            if process_start(pid) == started:
                os.kill(pid, signal.SIGCONT)
        except (FileNotFoundError, ProcessLookupError):
            pass


def pause_keepalives(devices, control):
    """Pause only the verified matmul keeper on the selected devices."""
    keeper = "/share/personal/liyanjun/meilu/gpu_keepalive/matmul_keepalive.py"
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit():
            continue
        try:
            argv = (folder / "cmdline").read_bytes().decode().strip("\0").split("\0")
            if keeper not in argv or "--gpu" not in argv:
                continue
            device = argv[argv.index("--gpu") + 1]
            if device not in devices:
                continue
            state = next(line for line in (folder / "status").read_text().splitlines() if line.startswith("State:"))
            if "T (stopped)" in state:
                continue
            pid = int(folder.name)
            started = process_start(pid)
            os.kill(pid, signal.SIGSTOP)
            PAUSED_KEEPALIVES.append((pid, started))
        except (FileNotFoundError, ProcessLookupError):
            continue
    write_json(control / "paused-keepalives.json", dict(processes=PAUSED_KEEPALIVES, devices=devices,
               policy="SIGSTOP retains keeper allocation; SIGCONT on normal service exit/failure"))


def stop(signum, frame):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None:
        CHILD.send_signal(signal.SIGTERM)


def write_json(path, value):
    temporary = path.with_suffix(".new.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def wait_for_gpus(control, devices, idle_memory_mib, min_free_gib=0):
    """Resource gating inside the supervised training service; no chat timers."""
    announced = False
    while not STOP:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True)
        cards = [[int(x.strip()) for x in line.split(",")] for line in result.stdout.strip().splitlines()]
        cards = [row for row in cards if str(row[0]) in devices]
        if len(cards) != len(devices):
            raise RuntimeError("Configured GPU devices are unavailable")
        import shutil
        free_gib = shutil.disk_usage(control).free / 1024**3
        if all(memory < idle_memory_mib for _, memory in cards) and free_gib >= min_free_gib:
            return True
        write_json(control / "queue-status.json", dict(state="waiting_for_gpus" if free_gib >= min_free_gib else "waiting_for_storage",
                   memory_mib=cards, free_gib=free_gib, min_free_gib=min_free_gib,
                   utc=datetime.now(timezone.utc).isoformat(), reason="Resource gate; other tasks and existing files left untouched"))
        if not announced:
            print(json.dumps(dict(event="waiting_for_gpus", memory_mib=cards)), flush=True)
            announced = True
        time.sleep(30)
    return False


def main():
    global CHILD
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--official", required=True)
    parser.add_argument("--seconds", type=int, default=14400)
    parser.add_argument("--global-batch", type=int, default=16)
    parser.add_argument("--devices", default="1,2")
    parser.add_argument("--idle-memory-mib", type=int, default=12000)
    parser.add_argument("--pause-keepalive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-eval-samples", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--compact-checkpoints", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=0)
    args = parser.parse_args()
    recipe = ["--train-eval-samples", str(args.train_eval_samples), "--eval-every", str(args.eval_every),
              "--save-every", str(args.save_every)] + (["--compact-checkpoints"] if args.compact_checkpoints else [])
    devices = args.devices.split(",")
    if len(set(devices)) != len(devices) or not all(d.isdigit() for d in devices):
        raise ValueError("Invalid GPU selection")
    world = len(devices)
    root = Path(args.root).resolve()
    scripts = Path(__file__).resolve().parent
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    temporary = project_tmp(root)
    temporary.mkdir(parents=True, exist_ok=True)
    lock = (control / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.devices, CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
               TMPDIR=str(temporary), XDG_CACHE_HOME=str(root / "runtime/cache"),
               TORCHINDUCTOR_CACHE_DIR=str(root / "runtime/cache/inductor"),
               PYTHONPATH=str(root / "vendor") + ":" + str(scripts))
    if not args.dry_run:
        if not (control / "preflight.json").is_file() or not json.loads((control / "preflight.json").read_text())["passed"]:
            raise RuntimeError("CPU/data preflight must pass before starting this service")
        if args.pause_keepalive:
            atexit.register(restore_keepalives)
            pause_keepalives(devices, control)
        smoke = root / "smoke_2gpu"
        if not (smoke / "autoregressive_evaluation.json").exists():
            if not wait_for_gpus(control, devices, args.idle_memory_mib, args.min_free_gib):
                return
            smoke.mkdir(parents=True, exist_ok=True)
            command = [args.python, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world}",
                       str(scripts / "train.py"), "--cache", str(root / "cache"), "--task", TASKS[0],
                       "--official", args.official, "--output", str(smoke), "--max-seconds", "1200",
                       "--max-steps", "5", "--global-batch", str(args.global_batch)] + recipe
            smoke_status = json.loads((smoke / "status.json").read_text()) if (smoke / "status.json").exists() else {}
            if smoke_status.get("state") != "complete":
                if (smoke / "latest.json").exists():
                    command += ["--resume", json.loads((smoke / "latest.json").read_text())["path"]]
                write_json(control / "queue-status.json", dict(state="gpu_smoke", task=TASKS[0]))
                with (smoke / "launcher.log").open("a") as log:
                    CHILD = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                    rc = CHILD.wait()
                if STOP:
                    return
                if rc:
                    write_json(control / "queue-status.json", dict(state="gpu_smoke_failed", returncode=rc))
                    sys.exit(rc)
            readback = json.loads((smoke / "checkpoint-readback.json").read_text())
            if not readback["passed"] or readback["step"] != 5 or readback["world"] != world:
                raise RuntimeError("Five-step distributed checkpoint smoke did not pass")
            command = [args.python, str(scripts / "evaluate.py"), "--cache", str(root / "cache"),
                       "--task", TASKS[0], "--training", str(smoke), "--max-samples", "1"]
            with (smoke / "evaluation.log").open("a") as log:
                CHILD = subprocess.Popen(command, cwd=root, env=dict(env, CUDA_VISIBLE_DEVICES=devices[0]),
                                         stdout=log, stderr=subprocess.STDOUT)
                rc = CHILD.wait()
            if STOP:
                return
            if rc:
                write_json(control / "queue-status.json", dict(state="gpu_smoke_evaluation_failed", returncode=rc))
                sys.exit(rc)
    for task in TASKS:
        output = root / "runs" / task
        status_file = output / "status.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        if status.get("state") != "complete":
            command = [args.python, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world}",
                       str(scripts / "train.py"), "--cache", str(root / "cache"), "--task", task,
                       "--official", args.official, "--output", str(output),
                       "--max-seconds", str(args.seconds), "--global-batch", str(args.global_batch)] + recipe
            latest = output / "latest.json"
            if latest.exists():
                command += ["--resume", json.loads(latest.read_text())["path"]]
            print(json.dumps(dict(task=task, command=command, dry_run=args.dry_run)), flush=True)
            if not args.dry_run:
                if not wait_for_gpus(control, devices, args.idle_memory_mib, args.min_free_gib):
                    return
                output.mkdir(parents=True, exist_ok=True)
                write_json(control / "queue-status.json", dict(state="training", task=task, utc=datetime.now(timezone.utc).isoformat()))
                with (output / "launcher.log").open("a") as log:
                    CHILD = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                    rc = CHILD.wait()
                if STOP:
                    write_json(control / "queue-status.json", dict(state="stopped", task=task, returncode=rc))
                    return
                if rc:
                    write_json(control / "queue-status.json", dict(state="failed", task=task, returncode=rc))
                    sys.exit(rc)
                status = json.loads(status_file.read_text())
                if status.get("state") != "complete":
                    raise RuntimeError(f"Task did not finish budget: {task}")
        if STOP:
            return
        if not (output / "autoregressive_evaluation.json").exists():
            command = [args.python, str(scripts / "evaluate.py"), "--cache", str(root / "cache"),
                       "--task", task, "--training", str(output)]
            if args.dry_run:
                print(json.dumps(dict(task=task, evaluation=command)), flush=True)
                continue
            write_json(control / "queue-status.json", dict(state="evaluating", task=task))
            with (output / "evaluation.log").open("a") as log:
                CHILD = subprocess.Popen(command, cwd=root, env=dict(env, CUDA_VISIBLE_DEVICES=devices[0]),
                                         stdout=log, stderr=subprocess.STDOUT)
                rc = CHILD.wait()
            if STOP:
                return
            if rc:
                write_json(control / "queue-status.json", dict(state="evaluation_failed", task=task, returncode=rc))
                sys.exit(rc)
        if not args.dry_run:
            summaries = [json.loads(p.read_text()) for p in (root / "runs").glob("*/autoregressive_evaluation.json")]
            write_json(control / "evaluation-summary.json", summaries)
    if not args.dry_run:
        write_json(control / "queue-status.json", dict(state="complete", tasks=TASKS))


if __name__ == "__main__":
    main()
