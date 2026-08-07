#!/usr/bin/env python3
"""Read-only RH56 six-channel FORCE_ACT dashboard.

The public RH56 protocol exposes six scalar fingertip-force channels at
holding registers 1582--1593. It does not define a taxel/pixel matrix.
"""
from __future__ import annotations

import argparse
import statistics
import threading
import time
from collections import deque

from flask import Flask, jsonify, render_template_string
from pymodbus.client import ModbusTcpClient


FORCE_ACT = 1582
FINGERS = ("小指", "无名指", "中指", "食指", "拇指弯曲", "拇指旋转")


def _read(client, device_id):
    for keyword in ("device_id", "slave", "unit"):
        try:
            result = client.read_holding_registers(address=FORCE_ACT, count=6, **{keyword: device_id})
            break
        except TypeError:
            result = None
    if result is None or result.isError():
        raise RuntimeError("读取 FORCE_ACT(1582--1593) 失败")
    return [value if value < 32768 else value - 65536 for value in result.registers]


class Reader:
    def __init__(self, ip, port, device_id, hz):
        self.ip, self.port, self.device_id, self.period = ip, port, device_id, 1 / hz
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.latest = {"ok": False, "forces_g": [0] * 6, "error": "尚未读取"}
        self.times = deque(maxlen=300)

    def loop(self):
        client = None
        while not self.stop.is_set():
            start = time.perf_counter()
            try:
                if client is None:
                    client = ModbusTcpClient(self.ip, port=self.port, timeout=1.0)
                    if not client.connect():
                        client.close(); client = None
                        raise ConnectionError(f"无法连接 {self.ip}:{self.port}")
                values = _read(client, self.device_id)
                elapsed = (time.perf_counter() - start) * 1000
                with self.lock:
                    self.times.append(elapsed)
                    self.latest = {"ok": True, "forces_g": values, "read_ms": round(elapsed, 3), "at": time.monotonic(), "error": None}
            except Exception as exc:
                if client: client.close(); client = None
                with self.lock: self.latest = {"ok": False, "forces_g": [0] * 6, "error": str(exc)}
            self.stop.wait(max(0, self.period - (time.perf_counter() - start)))

    def snapshot(self):
        with self.lock: state, samples = dict(self.latest), list(self.times)
        at = state.pop("at", None)
        state["age_ms"] = None if at is None else round((time.monotonic() - at) * 1000, 1)
        state["performance"] = {"samples": len(samples), "mean_read_ms": round(statistics.fmean(samples), 3) if samples else None, "p95_read_ms": round(sorted(samples)[max(0, int(len(samples) * .95) - 1)], 3) if samples else None}
        return state


PAGE = """<!doctype html><html lang=zh-CN><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>RH56 触觉监视器</title><style>
:root{color-scheme:dark;--bg:#101719;--panel:#192326;--line:#334346;--ink:#edf4f2;--muted:#9cacaa;--accent:#f4ac3e}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,'Microsoft YaHei',sans-serif}main{max-width:1050px;margin:auto;padding:28px 20px}header{border-bottom:1px solid var(--line);padding-bottom:18px;display:flex;justify-content:space-between;gap:16px}h1{font-size:25px;margin:0 0 6px}.sub,.note{color:var(--muted);font-size:13px;line-height:1.6;max-width:700px}.badge{white-space:nowrap;border:1px solid var(--line);border-radius:999px;padding:8px 12px;height:min-content}.online{color:#9cdebb}.offline{color:#ffab98}.layout{display:grid;grid-template-columns:1.1fr .9fr;gap:18px;margin-top:18px}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}.metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.metric{border-left:2px solid var(--accent);padding-left:9px}.metric small{display:block;color:var(--muted)}.metric b{font-size:18px}.hand{height:465px;position:relative;width:320px;margin:auto}.palm,.wrist,.finger{position:absolute;background:#2b393b;border:1px solid #5a6b6d}.palm{left:85px;top:205px;width:153px;height:178px;border-radius:46% 46% 25px 25px}.wrist{left:122px;top:370px;width:80px;height:93px;border-radius:0 0 25px 25px}.finger{width:39px;border-radius:25px 25px 15px 15px}.f0{left:42px;top:112px;height:155px;transform:rotate(-19deg)}.f1{left:91px;top:47px;height:202px}.f2{left:141px;top:17px;height:231px}.f3{left:191px;top:67px;height:180px}.f4{left:227px;top:213px;height:117px;transform:rotate(47deg);transform-origin:bottom}.dot{position:absolute;width:32px;height:32px;border-radius:50%;border:1px solid #a1adac;transition:.1s}.d0{left:38px;top:111px}.d1{left:94px;top:47px}.d2{left:144px;top:17px}.d3{left:194px;top:67px}.d4{left:239px;top:205px}.d5{left:259px;top:246px}.sensors{display:grid;gap:9px}.sensor{display:grid;grid-template-columns:1fr 72px;gap:10px;align-items:center;padding:10px;border:1px solid var(--line);border-radius:8px}.bar{height:7px;background:#2b3739;border-radius:99px;overflow:hidden}.bar i{display:block;height:100%;width:0%;background:var(--accent);transition:.1s}.value{text-align:right;font-variant-numeric:tabular-nums}.error{min-height:20px;color:#ffab98;margin-top:12px;font-size:13px}@media(max-width:700px){header,.layout{display:block}.badge{display:inline-block;margin-top:12px}.layout{margin-top:16px}}</style><main><header><div><h1>RH56 左手 · 触觉监视器</h1><div class=sub>实时读取官方 <code>FORCE_ACT</code> 六路实际受力。每个发光点对应一个真实传感器；公开 RH56 协议没有逐像素点阵定义，因此页面不伪造点阵。</div></div><div id=badge class=badge>连接中</div></header><section class=layout><div class=card><div class=metrics><div class=metric><small>最新读取</small><b id=read>—</b></div><div class=metric><small>数据年龄</small><b id=age>—</b></div><div class=metric><small>平均读取</small><b id=mean>—</b></div><div class=metric><small>P95 读取</small><b id=p95>—</b></div></div><div class=hand><i class=palm></i><i class=wrist></i><i class='finger f0'></i><i class='finger f1'></i><i class='finger f2'></i><i class='finger f3'></i><i class='finger f4'></i><i class='dot d0'></i><i class='dot d1'></i><i class='dot d2'></i><i class='dot d3'></i><i class='dot d4'></i><i class='dot d5'></i></div></div><div class=card><h2 style='font-size:17px;margin-top:0'>六路触点</h2><div id=sensors class=sensors></div><div id=error class=error></div><p class=note>后台只向手发送连续 Modbus 读取请求；网页只读取缓存。默认 20 Hz，可用 <code>--poll-hz</code> 调整。</p></div></section></main><script>const names=['小指','无名指','中指','食指','拇指弯曲','拇指旋转'],box=document.querySelector('#sensors');box.innerHTML=names.map((x,i)=>`<div class=sensor><div>${x}<div class=bar><i id=b${i}></i></div></div><span id=v${i} class=value>—</span></div>`).join('');function color(v){let p=Math.max(0,Math.min(1,v/1000)),h=48-38*p;return[p*100,`hsl(${h} 90% ${48+10*p}%)`,`0 0 ${8+28*p}px hsl(${h} 90% 55%/${.15+.7*p})`]}async function tick(){try{let s=await(await fetch('/api/tactile',{cache:'no-store'})).json(),ok=s.ok;badge.textContent=ok?'● 左手在线':'● 左手离线';badge.className='badge '+(ok?'online':'offline');read.textContent=s.read_ms==null?'—':s.read_ms+' ms';age.textContent=s.age_ms==null?'—':s.age_ms+' ms';mean.textContent=s.performance.mean_read_ms==null?'—':s.performance.mean_read_ms+' ms';p95.textContent=s.performance.p95_read_ms==null?'—':s.performance.p95_read_ms+' ms';error.textContent=s.error||'';s.forces_g.forEach((v,i)=>{let [p,c,g]=color(v);document.querySelector('#v'+i).textContent=v+' g';let b=document.querySelector('#b'+i);b.style.width=p+'%';b.style.background=c;let d=document.querySelector('.d'+i);d.style.background=c;d.style.boxShadow=g})}catch(e){error.textContent=e}}tick();setInterval(tick,100)</script>"""


def benchmark(ip, port, device_id, samples):
    client = ModbusTcpClient(ip, port=port, timeout=1.0)
    if not client.connect(): raise SystemExit(f"无法连接 {ip}:{port}")
    try:
        times = []
        for _ in range(samples):
            started = time.perf_counter(); _read(client, device_id); times.append((time.perf_counter() - started) * 1000)
        times.sort(); print(f"samples={samples} mean_ms={statistics.fmean(times):.3f} p95_ms={times[int(samples*.95)-1]:.3f} max_ms={times[-1]:.3f}")
    finally: client.close()


def main():
    p = argparse.ArgumentParser(description="RH56 read-only tactile dashboard")
    p.add_argument("--hand", default="192.168.123.210"); p.add_argument("--hand-port", type=int, default=6000); p.add_argument("--device-id", type=int, default=1); p.add_argument("--poll-hz", type=float, default=20); p.add_argument("--host", default="0.0.0.0"); p.add_argument("--web-port", type=int, default=8081); p.add_argument("--benchmark", action="store_true"); p.add_argument("--samples", type=int, default=300)
    a = p.parse_args()
    if a.benchmark: return benchmark(a.hand, a.hand_port, a.device_id, a.samples)
    reader = Reader(a.hand, a.hand_port, a.device_id, a.poll_hz); threading.Thread(target=reader.loop, daemon=True).start()
    app = Flask(__name__); app.add_url_rule("/", "index", lambda: render_template_string(PAGE)); app.add_url_rule("/api/tactile", "tactile", lambda: jsonify(reader.snapshot()))
    print(f"Dashboard: http://{a.host}:{a.web_port} (read-only, {a.poll_hz:g} Hz)"); app.run(host=a.host, port=a.web_port, threaded=True)


if __name__ == "__main__": main()
