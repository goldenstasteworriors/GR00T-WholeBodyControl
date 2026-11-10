"""Read/tune Revo2 hands through Unitree's official DDS service."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, render_template_string

from gear_sonic.utils.data_collection.brainco_hand import (
    DEFAULT_CONFIG, JOINT_NAMES, BraincoDdsClient, load_profiles, six,
)

PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BrainCo Revo2 手部调节</title>
<style>body{font:16px sans-serif;max-width:850px;margin:30px auto;padding:16px}
section{border:1px solid #bbb;padding:18px;margin:15px 0;border-radius:8px}
label{display:flex;gap:12px;margin:12px 0}input{flex:1}button{padding:9px;margin:5px}
pre{background:#eee;padding:12px;overflow:auto}</style>
<h1>强脑 Revo2 双手调节</h1>
<p>0=张开，1=闭合；按官方索引顺序。拖动只修改目标，点击发送才控制一只手。
启动模式：{{ '允许发送' if enabled else '禁止发送' }}。电流单位A，speed为归一化速度，均非力矩/弧度速度。</p>
<div id="hands"></div><pre id="result">就绪</pre><h2>真实反馈</h2><pre id="state"></pre>
<script>
const names={{ names|tojson }}, cfg={{ config|tojson }};
const values=s=>names.map((_,i)=>Number(document.getElementById(s+i).value));
function fill(s,pose){cfg[s][pose].forEach((v,i)=>{document.getElementById(s+i).value=v;
document.getElementById(s+i+'v').textContent=v})}
async function api(path,body){try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
const d=await r.json();document.getElementById('result').textContent=JSON.stringify(d,null,2);return d}
catch(e){document.getElementById('result').textContent=String(e);return {ok:false}}}
async function save(s,pose){const q=values(s),r=await api('/api/save',{side:s,pose,q});if(r.ok)cfg[s][pose]=q}
for(const side of ['left','right']){const panel=document.createElement('section');
panel.innerHTML='<h2>'+ (side==='left'?'左手':'右手')+'</h2>';
names.forEach((n,i)=>{const row=document.createElement('label');row.innerHTML=n+
'<input type="range" min="0" max="1" step="0.01" id="'+side+i+'"><output id="'+side+i+'v"></output>';
row.querySelector('input').oninput=e=>row.querySelector('output').textContent=e.target.value;panel.append(row)});
panel.insertAdjacentHTML('beforeend',`<button onclick="fill('${side}','open')">加载张开目标</button>
<button onclick="fill('${side}','grasp')">加载抓握目标</button>
<button onclick="api('/api/command',{side:'${side}',q:values('${side}')})">发送此手目标</button>
<button onclick="save('${side}','open')">保存为此手张开</button>
<button onclick="save('${side}','grasp')">保存为此手抓握</button>`);
document.getElementById('hands').append(panel);fill(side,'open')}
async function refresh(){try{const r=await fetch('/api/state');document.getElementById('state').textContent=JSON.stringify(await r.json(),null,2)}
catch(e){document.getElementById('state').textContent='连接失败：'+String(e)}}
refresh();setInterval(refresh,1000);
</script></html>"""


def create_app(client, config_path, *, enable_commands=False, max_state_age=0.25):
    path = Path(config_path).expanduser().resolve()
    config = load_profiles(path)
    lock = threading.Lock()
    app = Flask(__name__)

    @app.get("/")
    def index():
        with lock:
            return render_template_string(PAGE, config=config, names=JOINT_NAMES, enabled=enable_commands)

    @app.get("/api/state")
    def state():
        states = client.snapshot()
        return jsonify(ok=True, states=states, fresh={
            side: states[side] is not None and states[side]["age_s"] <= max_state_age
            for side in ("left", "right")})

    @app.post("/api/command")
    def command():
        if not enable_commands:
            return jsonify(ok=False, error="只读模式；需以 --enable-commands 启动"), 403
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify(ok=False, error="请求必须是JSON对象"), 400
        side = data.get("side")
        if side not in ("left", "right"):
            return jsonify(ok=False, error="请选择left或right，一次只控制一只手"), 400
        try:
            q = six(data.get("q"))
            actual = client.snapshot()[side]
            if actual is None or actual["age_s"] > max_state_age:
                return jsonify(ok=False, error="该侧没有新鲜的官方DDS反馈，拒绝发送"), 409
            with lock:
                speed = config["speed"]
            client.command(side, q, speed)
            return jsonify(ok=True, side=side, q=q, speed=speed)
        except (ValueError, TypeError) as exc:
            return jsonify(ok=False, error=str(exc)), 400
        except RuntimeError as exc:
            return jsonify(ok=False, error=str(exc)), 503

    @app.post("/api/save")
    def save():
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify(ok=False, error="请求必须是JSON对象"), 400
        side, pose = data.get("side"), data.get("pose")
        if side not in ("left", "right") or pose not in ("open", "grasp"):
            return jsonify(ok=False, error="side/pose无效"), 400
        try:
            q = six(data.get("q"))
        except (ValueError, TypeError) as exc:
            return jsonify(ok=False, error=str(exc)), 400
        with lock:
            updated = json.loads(json.dumps(config))
            updated[side][pose] = q
            # Keep temporary writes beside the project config, never in /tmp.
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
                    temp_path = Path(f.name)
                    json.dump(updated, f, indent=2)
                    f.write("\n")
                os.replace(temp_path, path)
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink()
            config.update(updated)
        return jsonify(ok=True, path=str(path), side=side, pose=pose, q=q,
                       note="已保存机器人配置；采集前须按SOP同步回PC")

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--enable-commands", action="store_true")
    parser.add_argument("--check-seconds", type=float, default=0,
                        help="只读检查两侧反馈后退出；不发送命令")
    args = parser.parse_args()
    load_profiles(args.config)
    client = BraincoDdsClient(args.network)
    try:
        if args.check_seconds > 0:
            deadline = time.monotonic() + args.check_seconds
            while time.monotonic() < deadline:
                result = client.snapshot()
                if all(v is not None and v["age_s"] <= 0.25 for v in result.values()):
                    print(json.dumps(result, indent=2))
                    return
                time.sleep(0.05)
            raise SystemExit("未收到两只手的新鲜反馈；检查官方服务、串口、DDS网卡")
        create_app(client, args.config, enable_commands=args.enable_commands).run(
            host=args.host, port=args.port, threaded=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
