"""Control a project-private supervisord instance without system configuration."""
import argparse
from pathlib import Path
import subprocess


ROOT = Path("/share/personal/yekangjie/JAR_lab_primitives_20261111")
PYTHON = "/share/personal/liyanjun/yanjun/envs/jar/bin/python"
OFFICIAL = "/share/personal/liyanjun/yanjun/JAR/assets/olmo2-0425-1b-official"
SUPERVISORD = "/root/miniforge/bin/supervisord"
SUPERVISORCTL = "/root/miniforge/bin/supervisorctl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("start", "status", "stop"))
    args = parser.parse_args()
    control = ROOT / "control"
    config = control / "supervisord.conf"
    if args.action == "start":
        if (control / "supervisor.sock").exists():
            subprocess.run([SUPERVISORCTL, "-c", str(config), "status"], check=True)
            return
        code = Path(__file__).resolve().parent
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

[program:jar-lab-20261111]
command={PYTHON} -u {code}/queue.py --root {ROOT} --python {PYTHON} --official {OFFICIAL} --seconds 14400 --global-batch 32
directory={ROOT}
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
environment=TMPDIR="{ROOT}/runtime/tmp",XDG_CACHE_HOME="{ROOT}/runtime/cache",PYTHONDONTWRITEBYTECODE="1"
""")
        subprocess.run([SUPERVISORD, "-c", str(config)], check=True)
        subprocess.run([SUPERVISORCTL, "-c", str(config), "status"], check=True)
    elif args.action == "stop":
        subprocess.run([SUPERVISORCTL, "-c", str(config), "stop", "jar-lab-20261111"], check=True)
    else:
        subprocess.run([SUPERVISORCTL, "-c", str(config), "status"], check=True)


if __name__ == "__main__":
    main()
