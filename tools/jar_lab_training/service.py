"""Control a project-private supervisord instance without system configuration."""
import argparse
from pathlib import Path
import subprocess
import shlex


ROOT = Path("/share/personal/yekangjie/JAR_lab_primitives_20261111")
PYTHON = "/share/personal/liyanjun/yanjun/envs/jar/bin/python"
OFFICIAL = "/share/personal/liyanjun/yanjun/JAR/assets/olmo2-0425-1b-official"
SUPERVISORD = next((p for p in ("/root/miniforge/bin/supervisord", "/etc/aibox/bin/supervisord") if Path(p).is_file()), None)
SUPERVISORCTL = next((p for p in ("/root/miniforge/bin/supervisorctl", "/etc/aibox/bin/supervisorctl") if Path(p).is_file()), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "status", "stop"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seconds", type=int, default=14400)
    parser.add_argument("--devices", default="1,2")
    parser.add_argument("--train-eval-samples", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--compact-checkpoints", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    program = "jar-lab-20261111" if root == ROOT else "jar-" + root.name
    control = root / "control"
    config = control / "supervisord.conf"
    if args.action == "start":
        if (control / "supervisor.sock").exists():
            status = subprocess.run([SUPERVISORCTL, "-c", str(config), "status"])
            if status.returncode:
                subprocess.run([SUPERVISORCTL, "-c", str(config), "start", program], check=True)
            return
        code = Path(__file__).resolve().parent
        command = [PYTHON, "-u", str(code/"run_queue.py"), "--root", str(root), "--python", PYTHON,
                   "--official", OFFICIAL, "--seconds", str(args.seconds), "--global-batch", "16",
                   "--devices", args.devices, "--idle-memory-mib", "12000", "--pause-keepalive",
                   "--train-eval-samples", str(args.train_eval_samples), "--eval-every", str(args.eval_every),
                   "--save-every", str(args.save_every), "--min-free-gib", str(args.min_free_gib)]
        if args.compact_checkpoints:
            command.append("--compact-checkpoints")
        config.write_text(f"""[unix_http_server]
file={control}/supervisor.sock
chmod=0600

[supervisord]
logfile={control}/supervisord.log
logfile_maxbytes=5MB
logfile_backups=2
pidfile={control}/supervisord.pid
childlogdir={control}
nodaemon=false

[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix://{control}/supervisor.sock

[program:{program}]
command={shlex.join(command)}
directory={root}
autostart=true
autorestart=false
startsecs=5
startretries=0
stopsignal=TERM
stopwaitsecs=900
stopasgroup=false
killasgroup=true
redirect_stderr=true
stdout_logfile={control}/queue.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=2
environment=TMPDIR="{root}/runtime/tmp",XDG_CACHE_HOME="{root}/runtime/cache",PYTHONDONTWRITEBYTECODE="1"
""")
        subprocess.run([SUPERVISORD, "-c", str(config)], check=True)
        subprocess.run([SUPERVISORCTL, "-c", str(config), "status"], check=True)
    elif args.action == "stop":
        subprocess.run([SUPERVISORCTL, "-c", str(config), "stop", program], check=True)
    else:
        subprocess.run([SUPERVISORCTL, "-c", str(config), "status"], check=True)


if __name__ == "__main__":
    main()
