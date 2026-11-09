"""Render held-out NaVILA predictions with their actual input images."""

import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = {r["id"]: r for r in json.loads((args.dataset / "test.json").read_text())}
    report = json.loads(args.predictions.read_text())
    rows = report["samples"]
    assert len({r["id"] for r in rows}) == len(rows)
    assert set(data) == {r["id"] for r in rows}, "Show the entire held-out set"
    args.output.mkdir(parents=True, exist_ok=True)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 19)
    canvas = Image.new("RGB", (1240, 180 + 440 * ((len(rows) + 1) // 2)), "#edf1f5")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "NaVILA 微调效果 · 全部留出测试样本", font=font, fill="#142332")
    draw.text((24, 60), "示教画面 + 模型离线预测；不是自主行走录像", font=small, fill="#405163")
    draw.text((24, 96), "终点停止 3/3；前进误停 2/5；距离为速度指令积分弱标签", font=small, fill="#405163")
    cards = []
    for i, row in enumerate(rows):
        item = data[row["id"]]
        assert row["expected_action"] == item["a"]
        predicted, expected = row["predicted_action"], item["a"]
        status = "一致" if predicted == expected else "提前停止" if predicted == "Stop." else "距离不同"
        color = {"一致": "#087f5b", "提前停止": "#ba3030", "距离不同": "#936000"}[status]
        task = "试剂瓶与试管架之间" if item["video_id"].startswith("c1_") else "烧杯前"
        x, y = 20 + (i % 2) * 610, 160 + (i // 2) * 440
        draw.rounded_rectangle((x, y, x + 590, y + 420), radius=12, fill="white")
        draw.text((x + 12, y + 8), f"{i + 1}. {task} · {status}", font=font, fill=color)
        latest = Image.open(args.dataset / item["frames"][-1]).convert("RGB")
        latest.thumbnail((560, 265))
        canvas.paste(latest, (x + (590 - latest.width) // 2, y + 50))
        draw.text((x + 12, y + 321), "示教：" + expected, font=small, fill="#25394c")
        draw.text((x + 12, y + 353), "预测：" + predicted, font=small, fill=color)
        draw.text((x + 12, y + 387), row["id"], font=small, fill="#657484")
        images = []
        for name in item["frames"]:
            im = Image.open(args.dataset / name).convert("RGB")
            im.thumbnail((640, 480))
            buffer = io.BytesIO()
            im.save(buffer, format="JPEG", quality=82)
            images.append("data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode())
        image_json = html.escape(json.dumps(images), quote=True)
        cards.append(f'''<article><h2>{i+1}. {task} <span style="color:{color}">{status}</span></h2>
<p>{html.escape(row['id'])}</p><img src="{images[-1]}" alt="实际模型输入历史图像">
<label>历史输入帧 <output>{len(images)}/{len(images)}</output>
<input type="range" min="0" max="{len(images)-1}" value="{len(images)-1}" data-images="{image_json}"
oninput="const a=JSON.parse(this.dataset.images);this.closest('article').querySelector('img').src=a[this.value];this.previousElementSibling.value=(Number(this.value)+1)+'/'+a.length"></label>
<p>示教：<b>{html.escape(expected)}</b></p><p style="color:{color}">预测：<b>{html.escape(predicted)}</b></p></article>''')
    canvas.save(args.output / "all_test_predictions.png")
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NaVILA 留出测试效果</title><style>body{font:17px system-ui;background:#edf1f5;color:#142332;margin:24px auto;max-width:1240;padding:0 16px}h1{font-size:29px}header{padding:12px 0 24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:20px}article{padding:18px;background:white;border-radius:12px}h2{font-size:20px}img{width:100%;aspect-ratio:4/3;object-fit:contain;background:#111}input{width:100%;margin:14px 0}p{line-height:1.6}header p{max-width:900px}label{display:block;margin-top:12px}</style>
<header><h1>NaVILA 两任务微调 · 全部留出测试</h1><p>这是示教图像上的离线预测，不是模型自主控制机器人。3 条独立测试轨迹，共 8 个决策；训练和选模均未使用这些样本。拖动滑块可查看该决策的所有历史输入帧。</p>
<p><b>终点停止 3/3；前进误停 2/5；完整指令一致 4/8。</b>前进距离为速度指令积分弱标签。提前停止仍明显，不能视为可靠的实机导航。</p></header><main>'''
    (args.output / "index.html").write_text(page + "\n".join(cards) + "</main></html>")
    print(args.output / "index.html")
    print(args.output / "all_test_predictions.png")


if __name__ == "__main__":
    main()
