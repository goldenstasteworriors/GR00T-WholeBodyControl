"""Server-managed uploads for the six-operation training queue, independent of SSH."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path("/share/personal/yekangjie/JAR_lab_primitives_20261111")
OUTPUT = ROOT / "uploads/modelscope_20261112"
CONTROL = ROOT / "control"
CONFIG = CONTROL / "ms-upload-supervisord.conf"
SOCKET = CONTROL / "ms-upload.sock"
PYTHON = OUTPUT / ".venv/bin/python"
PROGRAM = "jar-modelscope-upload"
STOP = False
CHILD = None


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def state(value):
    value.update(utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid())
    path = OUTPUT / "managed-status.json"
    pending = path.with_suffix(".new.json")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    pending.replace(path)


def stop(signum, frame):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None:
        CHILD.terminate()


def worker():
    global CHILD
    lock = (CONTROL / "ms-upload.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    expected = set(read(ROOT / "cache/complete.json")["tasks"])
    while not STOP:
        receipt = read(OUTPUT / "upload-receipt.json", {})
        uploaded = set(receipt.get("operations", [])) if receipt.get("sha256_verified") else set()
        ready = {task for task in expected
                 if read(ROOT / "runs" / task / "status.json", {}).get("state") == "complete"
                 and (ROOT / "runs" / task / "autoregressive_evaluation.json").is_file()}
        if expected <= uploaded:
            state(dict(state="complete", uploaded=sorted(uploaded), expected=sorted(expected)))
            return
        pending = ready - uploaded
        if pending:
            token = OUTPUT / ".private/token"
            if not token.is_file():
                state(dict(state="failed", reason="ModelScope token file is missing", pending=sorted(pending)))
                raise FileNotFoundError("ModelScope credential required for upload")
            state(dict(state="uploading", pending=sorted(pending), uploaded=sorted(uploaded)))
            command = [str(PYTHON), "-u", str(Path(__file__).with_name("upload_modelscope.py")),
                       "--root", str(ROOT), "--output", str(OUTPUT), "--token-file", str(token)]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(OUTPUT / "runtime"),
                       XDG_CACHE_HOME=str(OUTPUT / "runtime/cache"), MODELSCOPE_CACHE=str(OUTPUT / "runtime/modelscope"))
            with (OUTPUT / "managed-transfer.log").open("a") as log:
                CHILD = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                rc = CHILD.wait()
            if STOP:
                break
            if rc:
                state(dict(state="retry_pending", returncode=rc, pending=sorted(pending)))
                # Server-side retry for this bounded transfer, never a chat timer.
                for _ in range(12):
                    if STOP:
                        break
                    time.sleep(5)
            continue
        state(dict(state="waiting_for_completed_training", uploaded=sorted(uploaded),
                   remaining=sorted(expected - uploaded), expected=sorted(expected)))
        for _ in range(12):
            if STOP:
                break
            time.sleep(5)
    state(dict(state="stopped"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "status", "stop", "worker"))
    args = parser.parse_args()
    if args.action == "worker":
        worker()
        return
    supervisor = "/etc/aibox/bin/supervisord"
    ctl = "/etc/aibox/bin/supervisorctl"
    if args.action == "start":
        if SOCKET.exists():
            result = subprocess.run([ctl, "-c", str(CONFIG), "status"])
            if result.returncode:
                subprocess.run([ctl, "-c", str(CONFIG), "start", PROGRAM], check=True)
            return
        CONFIG.write_text(f"""[unix_http_server]
file={SOCKET}
chmod=0600

[supervisord]
logfile={CONTROL}/ms-upload-supervisord.log
logfile_maxbytes=5MB
logfile_backups=2
pidfile={CONTROL}/ms-upload-supervisord.pid
childlogdir={CONTROL}
nodaemon=false

[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix://{SOCKET}

[program:{PROGRAM}]
command=/etc/aibox/bin/python3.9 -u {Path(__file__).resolve()} worker
directory={ROOT}
autostart=true
autorestart=unexpected
exitcodes=0
startsecs=3
startretries=3
stopsignal=TERM
stopwaitsecs=60
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={OUTPUT}/managed-service.log
stdout_logfile_maxbytes=5MB
stdout_logfile_backups=2
environment=PYTHONDONTWRITEBYTECODE="1",TMPDIR="{OUTPUT}/runtime"
""")
        subprocess.run([supervisor, "-c", str(CONFIG)], check=True)
        subprocess.run([ctl, "-c", str(CONFIG), "status"], check=True)
    elif args.action == "stop":
        subprocess.run([ctl, "-c", str(CONFIG), "stop", PROGRAM], check=True)
    else:
        subprocess.run([ctl, "-c", str(CONFIG), "status"], check=True)
        print(json.dumps(read(OUTPUT / "managed-status.json", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
