#!/usr/bin/env python3
"""Browser replay for .npz files created by record_rh56dftp_tactile.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, render_template_string

GROUPS = (
    ("小指", "fingers", (("little_end", "指端"), ("little_tip", "指尖"), ("little_pad", "指腹"))),
    ("无名指", "fingers", (("ring_end", "指端"), ("ring_tip", "指尖"), ("ring_pad", "指腹"))),
    ("中指", "fingers", (("middle_end", "指端"), ("middle_tip", "指尖"), ("middle_pad", "指腹"))),
    ("食指", "fingers", (("index_end", "指端"), ("index_tip", "指尖"), ("index_pad", "指腹"))),
    ("拇指", "thumb", (("thumb_end", "指端"), ("thumb_tip", "指尖"), ("thumb_mid", "指中"), ("thumb_pad", "指腹"))),
    ("掌心", "palm", (("palm", "掌心"),)),
)


PAGE = """<!doctype html><html lang=zh-CN><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>RH56DFTP 触觉回放</title><style>
:root{color-scheme:dark;--bg:#101719;--panel:#192326;--shell:#263638;--line:#546568;--ink:#edf4f2;--muted:#9badab;--accent:#f3ab3e}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,'Microsoft YaHei',sans-serif}main{max-width:1080px;margin:auto;padding:24px}header{border-bottom:1px solid var(--line);padding-bottom:16px;display:flex;justify-content:space-between;gap:16px}h1{font-size:24px;margin:0 0 6px}.sub{font-size:13px;color:var(--muted);line-height:1.55}.info{border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12px;height:min-content;white-space:nowrap}.controls{margin:18px 0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;display:grid;grid-template-columns:auto 1fr auto auto;gap:12px;align-items:center}button,select{font:inherit;color:var(--ink);background:#263336;border:1px solid #4a5a5d;border-radius:6px;padding:7px 11px}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}input[type=range]{width:100%;accent-color:var(--accent)}.hand{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}.anatomy{position:relative;margin:auto;width:650px;height:715px;background:#111a1c;border:1px solid #3b4a4d;border-radius:190px 190px 45px 45px;overflow:hidden}.part{position:absolute;background:var(--shell);border:1px solid var(--line);padding:7px;border-radius:25px 25px 12px 12px}.part h2{font-size:12px;text-align:center;margin:0 0 5px}.little{left:78px;top:142px;width:82px;height:313px}.ring{left:170px;top:70px;width:82px;height:390px}.middle{left:262px;top:28px;width:84px;height:433px}.index{left:356px;top:82px;width:82px;height:379px}.thumb{left:478px;top:348px;width:105px;height:278px;transform:rotate(-31deg);transform-origin:28px 220px;border-radius:30px}.palm{left:115px;top:468px;width:342px;height:205px;border-radius:28px 28px 34px 34px}.section{margin-top:5px}.section label{display:block;color:#c4d1ce;font-size:10px;margin:0 0 2px}.taxels{display:grid;gap:1px;padding:2px;background:#0c1213;border-radius:3px}.taxel{display:block;min-width:2px;min-height:4px;background:#172226;transition:background .12s}.palm .taxel{min-height:16px}.legend{display:flex;justify-content:space-between;align-items:center;margin-top:16px;font-size:12px;color:var(--muted)}.scale{width:180px;height:8px;border-radius:99px;background:linear-gradient(90deg,#172226,#b84017,#f4dc75)}@media(max-width:760px){main{padding:14px}header{display:block}.info{display:inline-block;margin-top:10px}.controls{grid-template-columns:auto 1fr}.anatomy{transform:scale(.72);transform-origin:top left;margin-bottom:-200px}}
</style><main><header><div><h1>RH56DFTP · 全触点回放</h1><div class=sub>每个色块都嵌在其真实手指或掌心位置：深色为无/弱接触，橙黄为更强接触。</div></div><div id=info class=info>载入中</div></header><section class=controls aria-label=回放控制><button id=play aria-label=播放或暂停>播放</button><input id=frame aria-label=回放帧 type=range min=0 value=0><output id=count>—</output><select id=speed aria-label=回放速度><option value=.5>0.5×</option><option value=1 selected>1×</option><option value=2>2×</option></select></section><section class=hand><div class=anatomy><div id=little></div><div id=ring></div><div id=middle></div><div id=index></div><div id=thumb></div><div id=palm></div></div><div class=legend><span>接触强度</span><i class=scale></i><span>0 — 4095</span></div></section></main><script>
const groups=%GROUPS%;let meta,frame=0,playing=false,timer;const frameInput=document.querySelector('#frame');function mount(g){let el=document.createElement('article'),h=document.createElement('h2');el.className='part '+g.location;h.textContent=g.name;el.append(h);for(const x of g.items){let sec=document.createElement('section'),label=document.createElement('label'),grid=document.createElement('div');sec.className='section';label.textContent=x.label;grid.className='taxels';grid.id=x.id;grid.style.gridTemplateColumns=`repeat(${x.shape[1]},1fr)`;for(let i=0;i<x.shape[0]*x.shape[1];i++){let dot=document.createElement('i');dot.className='taxel';grid.append(dot)}sec.append(label,grid);el.append(sec)}document.querySelector('#'+g.location).append(el)}function color(v){let p=Math.max(0,Math.min(1,v/4095));return `hsl(${190-145*p} ${35+55*p}% ${12+59*p}%)`}async function draw(i){let data=await (await fetch('/api/frame/'+i)).json();for(const g of groups)for(const x of g.items){let values=data[x.id],cells=document.querySelectorAll('#'+x.id+' .taxel');cells.forEach((c,n)=>c.style.background=color(values[n]))}frame=i;frameInput.value=i;count.textContent=`${i+1} / ${meta.frames} · ${(data.timestamp_s-meta.start_s).toFixed(2)} s`}function step(){if(!playing)return;draw((frame+1)%meta.frames).finally(()=>timer=setTimeout(step,1000/meta.hz/Number(speed.value)))}play.onclick=()=>{playing=!playing;play.textContent=playing?'暂停':'播放';if(playing)step();else clearTimeout(timer)};frameInput.oninput=e=>draw(Number(e.target.value));fetch('/api/meta').then(r=>r.json()).then(m=>{meta=m;info.textContent=`${m.frames} 帧 · ${m.hz.toFixed(2)} Hz · ${m.name}`;frameInput.max=m.frames-1;groups.forEach(mount);draw(0)})
</script>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="RH56DFTP tactile recording viewer")
    parser.add_argument("record", type=Path, help=".npz output from record_rh56dftp_tactile.py")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()
    if not args.record.is_file(): raise SystemExit(f"record not found: {args.record}")
    archive = np.load(args.record)
    frames = int(archive["timestamps_s"].shape[0])
    if frames == 0: raise SystemExit("record contains no frames")
    locations = ("little", "ring", "middle", "index", "thumb", "palm")
    groups = [{"name": name, "location": location, "items": [{"id": key, "label": label, "shape": list(archive[key].shape[1:])} for key, label in items]} for location, (name, _target, items) in zip(locations, GROUPS, strict=True)]
    app = Flask(__name__)
    @app.get("/")
    def index(): return render_template_string(PAGE.replace("%GROUPS%", __import__("json").dumps(groups, ensure_ascii=False)))
    @app.get("/api/meta")
    def api_meta(): return jsonify({"name": args.record.name, "frames": frames, "start_s": float(archive["timestamps_s"][0]), "hz": float(1 / np.median(np.diff(archive["timestamps_s"]))) if frames > 1 else 0.0})
    @app.get("/api/frame/<int:index>")
    def api_frame(index: int):
        if not 0 <= index < frames: abort(404)
        return jsonify({"timestamp_s": float(archive["timestamps_s"][index]), **{key: archive[key][index].reshape(-1).tolist() for _, _, items in GROUPS for key, _ in items}})
    print(f"Tactile replay: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__": main()
