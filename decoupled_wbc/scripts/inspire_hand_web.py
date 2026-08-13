#!/usr/bin/env python3
"""通过 DDS 控制 Inspire 灵巧手的本地网页服务。

网页仅发布 ``rt/inspire/cmd``；实际 Modbus 通信由同项目的
``inspire_modbus_hand.py --mode dds`` bridge 负责。DDS 索引约定为：
右手 0--5，左手 6--11。
"""

from __future__ import annotations

import argparse
import threading

from flask import Flask, jsonify, render_template_string, request
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_


HAND_DOF = 6
HAND_NAMES = ("小指", "无名指", "中指", "食指", "拇指弯曲", "拇指旋转")


class InspireDdsPublisher:
    """发布完整双手命令，防止控制一只手时覆写另一只手。"""

    def __init__(self) -> None:
        self._publisher = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
        self._publisher.Init()
        self._command = MotorCmds_([unitree_go_msg_dds__MotorCmd_() for _ in range(12)])
        self._lock = threading.Lock()
        self._values = {"left": [1.0] * HAND_DOF, "right": [1.0] * HAND_DOF}

    @staticmethod
    def _validate(values: object) -> list[float]:
        if not isinstance(values, list) or len(values) != HAND_DOF:
            raise ValueError(f"每只手需要 {HAND_DOF} 个关节值")
        try:
            return [max(0.0, min(1.0, float(value))) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError("关节值必须是 0 到 1 的数字") from exc

    def publish(self, updates: dict[str, object]) -> dict[str, list[float]]:
        with self._lock:
            for side, values in updates.items():
                if side not in self._values:
                    raise ValueError(f"未知手侧: {side}")
                self._values[side] = self._validate(values)

            for index, value in enumerate(self._values["right"]):
                self._command.cmds[index].q = value
            for index, value in enumerate(self._values["left"]):
                self._command.cmds[index + HAND_DOF].q = value
            self._publisher.Write(self._command)
            return {side: values.copy() for side, values in self._values.items()}

    def state(self) -> dict[str, list[float]]:
        with self._lock:
            return {side: values.copy() for side, values in self._values.items()}


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inspire 灵巧手 DDS 控制台</title><style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f2f5f8;margin:0;color:#17212b}.page{max-width:1080px;margin:32px auto;padding:24px;background:#fff;border-radius:16px;box-shadow:0 10px 32px #18283a18}.hint{color:#52606d}.hands{display:grid;grid-template-columns:repeat(auto-fit,minmax(410px,1fr));gap:20px}.card{border:1px solid #dbe3ea;border-radius:12px;padding:18px}.row{display:grid;grid-template-columns:92px 1fr 68px;gap:10px;align-items:center;margin:12px 0}input[type=range]{width:100%}input[type=number]{width:64px;padding:5px}button{border:0;border-radius:8px;padding:10px 14px;margin:4px;background:#1463a5;color:white;font-size:15px;cursor:pointer}.close{background:#b23b36}.muted{background:#596775}pre{background:#13202d;color:#e7f0f7;border-radius:10px;padding:14px;overflow:auto}.bar{margin:18px 0}@media(max-width:600px){.page{margin:0;border-radius:0}.hands{grid-template-columns:1fr}}
</style></head><body><main class="page"><h1>Inspire 灵巧手 DDS 控制台</h1><p class="hint">0 = 闭合，1 = 张开。点击“发送”才会下发；右手 DDS 索引 0–5，左手 6–11。</p>
<div class="bar"><button onclick="preset('both',1)">双手张开</button><button class="close" onclick="preset('both',0)">双手闭合</button><button class="muted" onclick="refreshState()">刷新网页状态</button></div><section class="hands" id="hands"></section><h3>发送结果</h3><pre id="log">就绪</pre></main><script>
const names={{ names|tojson }}, sides=['left','right'];
function el(id){return document.getElementById(id)} function clamp(v){v=Number(v);return Number.isFinite(v)?Math.max(0,Math.min(1,v)):0}
function build(){const root=el('hands');sides.forEach(side=>{const card=document.createElement('div');card.className='card';card.innerHTML=`<h2>${side==='left'?'左手':'右手'}</h2>`;names.forEach((name,i)=>{const row=document.createElement('div');row.className='row';row.innerHTML=`<label>${name}</label><input id="${side}-${i}" type="range" min="0" max="1" step="0.01" value="1"><input id="${side}-n${i}" type="number" min="0" max="1" step="0.01" value="1">`;const range=row.children[1],num=row.children[2];range.oninput=()=>num.value=range.value;num.oninput=()=>range.value=clamp(num.value);card.appendChild(row)});const actions=document.createElement('div');actions.innerHTML=`<button onclick="send('${side}')">发送此手</button><button class="muted" onclick="preset('${side}',1)">张开</button><button class="close" onclick="preset('${side}',0)">闭合</button>`;card.appendChild(actions);root.appendChild(card)})}
function values(side){return names.map((_,i)=>clamp(el(`${side}-${i}`).value))} function setValues(side,vals){vals.forEach((v,i)=>{el(`${side}-${i}`).value=v;el(`${side}-n${i}`).value=v})}
async function api(data){el('log').textContent='发送中…';try{const r=await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const v=await r.json();el('log').textContent=JSON.stringify(v,null,2);if(v.ok)Object.entries(v.values).forEach(([s,q])=>setValues(s,q))}catch(e){el('log').textContent=String(e)}}
function send(side){api({values:{[side]:values(side)}})} function preset(side,value){const target=side==='both'?sides:[side];target.forEach(s=>setValues(s,Array(6).fill(value)));api({values:Object.fromEntries(target.map(s=>[s,values(s)]))})} async function refreshState(){const r=await fetch('/api/state');const v=await r.json();Object.entries(v.values).forEach(([s,q])=>setValues(s,q));el('log').textContent=JSON.stringify(v,null,2)} build();refreshState();
</script></body></html>"""


def create_app(publisher: InspireDdsPublisher) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(PAGE, names=HAND_NAMES)

    @app.get("/api/state")
    def state():
        return jsonify(ok=True, values=publisher.state())

    @app.post("/api/command")
    def command():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("values"), dict):
            return jsonify(ok=False, error="请求体需要 values 对象"), 400
        try:
            values = publisher.publish(body["values"])
        except ValueError as exc:
            return jsonify(ok=False, error=str(exc)), 400
        return jsonify(ok=True, values=values)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 Inspire 灵巧手 DDS 网页控制台。")
    parser.add_argument("--network", default="eth0", help="DDS 网卡。")
    parser.add_argument("--domain-id", default=0, type=int, help="DDS 域 ID。")
    parser.add_argument("--host", default="127.0.0.1", help="网页监听地址。")
    parser.add_argument("--port", default=5000, type=int, help="网页监听端口。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ChannelFactoryInitialize(args.domain_id, args.network)
    app = create_app(InspireDdsPublisher())
    print(f"打开 http://{args.host}:{args.port}（DDS 网卡: {args.network}）")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
