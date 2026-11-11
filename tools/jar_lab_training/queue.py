"""Observable, resumable sequential operation queue; run under supervisord."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


TASKS = ["TAKE_TUBE", "TAKE_DROPPER_AND_ASPIRATE", "DISPENSE_TO_HELD_TUBE",
         "RETURN_DROPPER", "RETURN_TUBE_TO_RACK", "PLACE_TUBE_IN_BATH"]
STOP = False
CHILD = None


def stop(signum, frame):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None:
        CHILD.send_signal(signal.SIGTERM)


def write_json(path, value):
    temporary = path.with_suffix(".new.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    global CHILD
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--official", required=True)
    parser.add_argument("--seconds", type=int, default=14400)
    parser.add_argument("--global-batch", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    scripts = Path(__file__).resolve().parent
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
               TMPDIR=str(root / "runtime/tmp"), XDG_CACHE_HOME=str(root / "runtime/cache"),
               TORCHINDUCTOR_CACHE_DIR=str(root / "runtime/cache/inductor"),
               PYTHONPATH=str(root / "vendor") + ":" + str(scripts))
    for task in TASKS:
        output = root / "runs" / task
        status_file = output / "status.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        if status.get("state") != "complete":
            command = [args.python, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4",
                       str(scripts / "train.py"), "--cache", str(root / "cache"), "--task", task,
                       "--official", args.official, "--output", str(output),
                       "--max-seconds", str(args.seconds), "--global-batch", str(args.global_batch)]
            latest = output / "latest.json"
            if latest.exists():
                command += ["--resume", json.loads(latest.read_text())["path"]]
            print(json.dumps(dict(task=task, command=command, dry_run=args.dry_run)), flush=True)
            if not args.dry_run:
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
                CHILD = subprocess.Popen(command, cwd=root, env=dict(env, CUDA_VISIBLE_DEVICES="0"),
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
