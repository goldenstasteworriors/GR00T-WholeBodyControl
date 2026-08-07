#!/usr/bin/env python3
"""Read-only visualizer for every public RH56DFTP tactile taxel."""
from __future__ import annotations

import argparse
import statistics
import threading
import time
from collections import deque

from flask import Flask, jsonify, render_template_string
from pymodbus.client import ModbusTcpClient

# (identifier, label, Modbus address, rows, columns).  Addresses and shapes
# are from RH56DFTP user manual PRJ-02-TS-U-010, section 2.6.20.
SENSORS = (
    ("little_end", "小指指端", 3000, 3, 3), ("little_tip", "小指指尖", 3018, 12, 8), ("little_pad", "小指指腹", 3210, 10, 8),
    ("ring_end", "无名指指端", 3370, 3, 3), ("ring_tip", "无名指指尖", 3388, 12, 8), ("ring_pad", "无名指指腹", 3580, 10, 8),
    ("middle_end", "中指端", 3740, 3, 3), ("middle_tip", "中指指尖", 3758, 12, 8), ("middle_pad", "中指指腹", 3950, 10, 8),
    ("index_end", "食指端", 4110, 3, 3), ("index_tip", "食指指尖", 4128, 12, 8), ("index_pad", "食指指腹", 4320, 10, 8),
    ("thumb_end", "拇指指端", 4480, 3, 3), ("thumb_tip", "拇指指尖", 4498, 12, 8), ("thumb_mid", "拇指指中", 4690, 3, 3), ("thumb_pad", "拇指指腹", 4708, 12, 8),
    ("palm", "掌心", 4900, 8, 14),
)


def read_words(client, address, count, device_id):
    for key in ("device_id", "slave", "unit"):
        try:
            response = client.read_holding_registers(address=address, count=count, **{key: device_id})
            break
        except TypeError:
            response = None
    if response is None or response.isError():
        raise RuntimeError(f"Modbus 读取失败：{address}")
    return response.registers


class TactileReader:
    def __init__(self, ip, port, device_id, hz):
        self.ip, self.port, self.device_id, self.period = ip, port, device_id, 1 / hz
        self.lock, self.stop, self.samples = threading.Lock(), threading.Event(), deque(maxlen=300)
        self.latest = {"ok": False, "taxels": {}, "error": "尚未读取"}

    def loop(self):
        client = None
        while not self.stop.is_set():
            start = time.perf_counter()
            try:
                if client is None:
                    client = ModbusTcpClient(self.ip, port=self.port, timeout=1.0)
                    if not client.connect():
                        client.close(); client = None; raise ConnectionError(f"无法连接 {self.ip}:{self.port}")
                data = {name: read_words(client, address, rows * cols, self.device_id) for name, _, address, rows, cols in SENSORS}
                elapsed = (time.perf_counter() - start) * 1000
                with self.lock:
                    self.samples.append(elapsed)
                    self.latest = {"ok": True, "taxels": data, "read_ms": round(elapsed, 3), "at": time.monotonic(), "error": None}
            except Exception as exc:
                if client: client.close(); client = None
                with self.lock: self.latest = {"ok": False, "taxels": {}, "error": str(exc)}
            self.stop.wait(max(0, self.period - (time.perf_counter() - start)))

    def snapshot(self):
        with self.lock: state, samples = dict(self.latest), list(self.samples)
        at = state.pop("at", None)
        state["age_ms"] = None if at is None else round((time.monotonic() - at) * 1000, 1)
        state["performance"] = {"samples": len(samples), "mean_ms": round(statistics.fmean(samples), 2) if samples else None, "p95_ms": round(sorted(samples)[max(0, int(.95 * len(samples)) - 1)], 2) if samples else None}
        return state


PAGE = """<!doctype html><html lang=zh-CN><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>RH56DFTP 全触点监视器</title><style>
:root{color-scheme:dark;--bg:#101719;--panel:#192326;--line:#324145;--ink:#edf4f2;--muted:#9badaa}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,'Microsoft YaHei',sans-serif}main{max-width:1240px;margin:auto;padding:24px}header{display:flex;justify-content:space-between;gap:16px;border-bottom:1px solid var(--line);padding-bottom:16px}h1{margin:0 0 7px;font-size:24px}.sub{color:var(--muted);font-size:13px;line-height:1.55}.badge{border:1px solid var(--line);border-radius:999px;padding:8px 12px;height:min-content;white-space:nowrap}.on{color:#9bdfbd}.off{color:#ffad98}.hand{margin-top:20px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}.fingers{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px;align-items:end}.finger,.thumb,.palm{border:1px solid var(--line);background:#151f21;border-radius:9px;padding:10px}.finger h2,.thumb h2,.palm h2{margin:0 0 9px;font-size:14px}.section{margin-top:8px}.section label{color:var(--muted);font-size:11px;display:block;margin-bottom:4px}.taxels{display:grid;gap:1px;background:#0d1314;padding:2px;border-radius:3px}.taxel{min-width:2px;min-height:3px;background:#263235}.bottom{display:grid;grid-template-columns:1fr 1.7fr;gap:12px;margin-top:12px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:15px}.metric{border-left:2px solid #f3ab3e;padding-left:8px}.metric small{display:block;color:var(--muted)}.metric b{font-variant-numeric:tabular-nums}.error{color:#ffad98;font-size:13px;min-height:18px;margin-top:12px}@media(max-width:760px){main{padding:14px}header{display:block}.badge{display:inline-block;margin-top:10px}.fingers{grid-template-columns:repeat(2,1fr)}.bottom{grid-template-columns:1fr}.taxel{min-height:5px}}</style><main><header><div><h1>RH56DFTP 左手 · 全触点监视器</h1><div class=sub>1062 个真实触觉点：四指各 185 点，拇指 210 点，掌心 112 点。颜色表示原始 16 位触觉值（0–4095）。仅读取 Modbus 数据，不会控制手。</div></div><div id=badge class=badge>连接中</div></header><section class=hand><div id=fingers class=fingers></div><div class=bottom><div id=thumb></div><div id=palm></div></div><div class=metrics><div class=metric><small>本次读取</small><b id=read>—</b></div><div class=metric><small>数据年龄</small><b id=age>—</b></div><div class=metric><small>P95 读取</small><b id=p95>—</b></div></div><div id=error class=error></div></section></main><script>
const specs=%SPECS%;const root={fingers,thumb,palm};function add(s){let wrap=document.createElement('div'),h=document.createElement('h2');h.textContent=s.group;wrap.className=s.group==='拇指'?'thumb':s.group==='掌心'?'palm':'finger';wrap.append(h);s.items.forEach(x=>{let sec=document.createElement('div'),lab=document.createElement('label'),grid=document.createElement('div');lab.textContent=x.label;grid.className='taxels';grid.style.gridTemplateColumns=`repeat(${x.cols},1fr)`;grid.id=x.id;for(let i=0;i<x.rows*x.cols;i++){let c=document.createElement('i');c.className='taxel';grid.append(c)}sec.className='section';sec.append(lab,grid);wrap.append(sec)});root[s.target].append(wrap)}specs.forEach(add);function color(v){let p=Math.max(0,Math.min(1,v/4095));return `hsl(${48-48*p} 95% ${10+48*p}%)`}async function tick(){try{let s=await(await fetch('/api/tactile',{cache:'no-store'})).json();badge.textContent=s.ok?'● 左手在线':'● 左手离线';badge.className='badge '+(s.ok?'on':'off');read.textContent=s.read_ms==null?'—':s.read_ms+' ms';age.textContent=s.age_ms==null?'—':s.age_ms+' ms';p95.textContent=s.performance.p95_ms==null?'—':s.performance.p95_ms+' ms';error.textContent=s.error||'';for(let x of specs){let d=s.taxels[x.id]||[];document.querySelectorAll('#'+x.id+' .taxel').forEach((c,i)=>c.style.background=color(d[i]||0))}}catch(e){error.textContent=e}}tick();setInterval(tick,250)</script>"""


def main():
    p = argparse.ArgumentParser(description="RH56DFTP read-only tactile taxel dashboard")
    p.add_argument("--hand", default="192.168.123.210"); p.add_argument("--hand-port", type=int, default=6000); p.add_argument("--device-id", type=int, default=1); p.add_argument("--poll-hz", type=float, default=4); p.add_argument("--host", default="0.0.0.0"); p.add_argument("--web-port", type=int, default=8081)
    a = p.parse_args(); reader = TactileReader(a.hand, a.hand_port, a.device_id, a.poll_hz); threading.Thread(target=reader.loop, daemon=True).start()
    groups = (("小指", "fingers", SENSORS[0:3]), ("无名指", "fingers", SENSORS[3:6]), ("中指", "fingers", SENSORS[6:9]), ("食指", "fingers", SENSORS[9:12]), ("拇指", "thumb", SENSORS[12:16]), ("掌心", "palm", SENSORS[16:17]))
    specs = [{"group": group, "target": target, "items": [{"id": n, "label": label, "rows": rows, "cols": cols} for n, label, _, rows, cols in items]} for group, target, items in groups]
    app = Flask(__name__); app.add_url_rule("/", "index", lambda: render_template_string(PAGE.replace("%SPECS%", __import__("json").dumps(specs, ensure_ascii=False)))); app.add_url_rule("/api/tactile", "tactile", lambda: jsonify(reader.snapshot()))
    print(f"RH56DFTP dashboard: http://{a.host}:{a.web_port} ({a.poll_hz:g} Hz, read-only)"); app.run(host=a.host, port=a.web_port, threaded=True)


if __name__ == "__main__": main()
