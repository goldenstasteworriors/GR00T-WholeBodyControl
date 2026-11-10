"""
Web control service for Inspire RH56DFTP dexterous hands via Modbus TCP.

Hand IPs:
    Left hand:  192.168.123.210
    Right hand: 192.168.123.211
    Modbus TCP Port: 6000

Register map:
    1004  - clear error, write 1
    1474  - pos_set, 6 x INT16, range 0-2000
    1486  - angle_set, 6 x INT16, range 0-1000
    1498  - force_set, 6 x INT16, range 0-12000 grams
    1522  - speed_set, 6 x INT16, range 0-4000

Finger order:
    [little, ring, middle, index, thumb_bend, thumb_rotate]

Angle meaning used here:
    0    = fully closed / curled
    1000 = fully open / straight
"""

import time
import struct
import argparse
import threading

from flask import Flask, request, jsonify, render_template_string
from pymodbus.client import ModbusTcpClient


# -----------------------------
# Register addresses
# -----------------------------
REG_CLEAR_ERROR = 1004
REG_SAVE        = 1005
REG_POS_SET     = 1474
REG_ANGLE_SET   = 1486
REG_FORCE_SET   = 1498
REG_SPEED_SET   = 1522

REG_POS_ACT     = 1534
REG_ANGLE_ACT   = 1546
REG_FORCE_ACT   = 1582
REG_CURRENT_ACT = 1594
REG_ERROR       = 1606
REG_STATUS      = 1612
REG_TEMP        = 1618


FULL_OPEN_ANGLE = [1000, 1000, 1000, 1000, 1000, 1000]
FULL_CLOSE_ANGLE = [0, 0, 0, 0, 0, 1000]


# -----------------------------
# Modbus helpers
# -----------------------------
def pack_shorts(values):
    """
    Convert signed INT16 list to Modbus unsigned register list.
    """
    values = [int(v) for v in values]
    packed = struct.pack(">" + "h" * len(values), *values)
    return list(struct.unpack(">" + "H" * len(values), packed))


def unpack_shorts(registers):
    """
    Convert Modbus unsigned register list to signed INT16 list.
    """
    packed = struct.pack(">" + "H" * len(registers), *registers)
    return list(struct.unpack(">" + "h" * len(registers), packed))


def is_modbus_error(result):
    if result is None:
        return True

    checker = getattr(result, "isError", None)
    if callable(checker):
        return checker()

    return False


def mb_write_register(client, address, value, device_id=1):
    """
    Compatible with different pymodbus versions:
    - newer versions may use device_id
    - some versions use slave
    - older versions use unit
    """
    attempts = [
        lambda: client.write_register(address=address, value=value, device_id=device_id),
        lambda: client.write_register(address=address, value=value, slave=device_id),
        lambda: client.write_register(address=address, value=value, unit=device_id),
        lambda: client.write_register(address, value),
    ]

    last_error = None
    for fn in attempts:
        try:
            return fn()
        except TypeError as e:
            last_error = e

    raise last_error


def mb_write_registers(client, address, values, device_id=1):
    attempts = [
        lambda: client.write_registers(address=address, values=values, device_id=device_id),
        lambda: client.write_registers(address=address, values=values, slave=device_id),
        lambda: client.write_registers(address=address, values=values, unit=device_id),
        lambda: client.write_registers(address, values),
    ]

    last_error = None
    for fn in attempts:
        try:
            return fn()
        except TypeError as e:
            last_error = e

    raise last_error


def mb_read_holding_registers(client, address, count, device_id=1):
    attempts = [
        lambda: client.read_holding_registers(address=address, count=count, device_id=device_id),
        lambda: client.read_holding_registers(address=address, count=count, slave=device_id),
        lambda: client.read_holding_registers(address=address, count=count, unit=device_id),
        lambda: client.read_holding_registers(address, count=count),
    ]

    last_error = None
    for fn in attempts:
        try:
            return fn()
        except TypeError as e:
            last_error = e

    raise last_error


def clamp_values(values, min_value, max_value, length=6):
    if not isinstance(values, list):
        raise ValueError("values must be a list")

    if len(values) != length:
        raise ValueError(f"values must contain exactly {length} numbers")

    result = []
    for v in values:
        v = int(v)
        v = max(min_value, min(max_value, v))
        result.append(v)

    return result


def percent_to_angle_values(values):
    percents = clamp_values(values, 0, 100)
    return [v * 10 for v in percents]


def open_percent_to_angle_values(percent):
    percent = int(percent)
    percent = max(0, min(100, percent))
    angle = percent * 10
    values = [angle] * 6
    values[5] = FULL_OPEN_ANGLE[5]
    return values


# -----------------------------
# Hand class
# -----------------------------
class InspireHand:
    def __init__(self, side, ip, port=6000, device_id=1, timeout=2):
        self.side = side
        self.ip = ip
        self.port = port
        self.device_id = device_id
        self.timeout = timeout
        self.lock = threading.Lock()

    def _connect(self):
        client = ModbusTcpClient(
            host=self.ip,
            port=self.port,
            timeout=self.timeout,
        )

        if not client.connect():
            raise ConnectionError(f"{self.side} hand connect failed: {self.ip}:{self.port}")

        return client

    def _run(self, fn):
        """
        Open one short Modbus connection per operation.
        This is more stable for a web service and avoids stale connections.
        """
        with self.lock:
            client = self._connect()
            try:
                return fn(client)
            finally:
                client.close()

    def clear_errors(self, client):
        result = mb_write_register(
            client,
            REG_CLEAR_ERROR,
            1,
            device_id=self.device_id,
        )

        if is_modbus_error(result):
            raise RuntimeError(f"{self.side} hand clear error failed")

    def set_speed(self, client, values):
        values = clamp_values(values, 0, 4000)
        result = mb_write_registers(
            client,
            REG_SPEED_SET,
            pack_shorts(values),
            device_id=self.device_id,
        )

        if is_modbus_error(result):
            raise RuntimeError(f"{self.side} hand set speed failed")

    def set_force(self, client, values):
        values = clamp_values(values, 0, 12000)
        result = mb_write_registers(
            client,
            REG_FORCE_SET,
            pack_shorts(values),
            device_id=self.device_id,
        )

        if is_modbus_error(result):
            raise RuntimeError(f"{self.side} hand set force failed")

    def set_angle(self, values, speed=3000, force=12000, clear_error=True):
        """
        Set joint angles:
            0    = closed
            1000 = open
        """
        values = clamp_values(values, 0, 1000)
        speed_values = [int(speed)] * 6
        force_values = [int(force)] * 6

        def job(client):
            if clear_error:
                self.clear_errors(client)
                time.sleep(0.05)

            self.set_speed(client, speed_values)
            time.sleep(0.05)

            self.set_force(client, force_values)
            time.sleep(0.05)

            result = mb_write_registers(
                client,
                REG_ANGLE_SET,
                pack_shorts(values),
                device_id=self.device_id,
            )

            if is_modbus_error(result):
                raise RuntimeError(f"{self.side} hand set angle failed")

            return {
                "side": self.side,
                "ip": self.ip,
                "angle": values,
                "speed": speed_values,
                "force": force_values,
                "message": "angle command sent",
            }

        return self._run(job)

    def open(self, speed=3000, force=12000):
        return self.set_angle(FULL_OPEN_ANGLE, speed=speed, force=force)

    def close(self, speed=3000, force=12000):
        return self.set_angle(FULL_CLOSE_ANGLE, speed=speed, force=force)

    def set_open_percent(self, percent, speed=3000, force=12000):
        """
        percent:
            0   = fully closed
            100 = fully open
        """
        percent = int(percent)
        percent = max(0, min(100, percent))
        angle = int(percent * 10)
        return self.set_angle([angle] * 6, speed=speed, force=force)

    def read_registers(self, client, address, count):
        result = mb_read_holding_registers(
            client,
            address,
            count,
            device_id=self.device_id,
        )

        if is_modbus_error(result):
            return None

        return unpack_shorts(result.registers)

    def get_state(self):
        def job(client):
            return {
                "side": self.side,
                "ip": self.ip,
                "pos_act": self.read_registers(client, REG_POS_ACT, 6),
                "angle_act": self.read_registers(client, REG_ANGLE_ACT, 6),
                "force_act": self.read_registers(client, REG_FORCE_ACT, 6),
                "current": self.read_registers(client, REG_CURRENT_ACT, 6),
                "error": self.read_registers(client, REG_ERROR, 3),
                "status": self.read_registers(client, REG_STATUS, 3),
                "temp": self.read_registers(client, REG_TEMP, 3),
            }

        return self._run(job)


# -----------------------------
# Web app
# -----------------------------
app = Flask(__name__)

hands = {}


HTML_PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>Inspire 灵巧手控制台</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        body {
            margin: 0;
            font-family: Arial, "Microsoft YaHei", sans-serif;
            background: #f5f6f8;
            color: #222;
        }

        .container {
            max-width: 860px;
            margin: 40px auto;
            padding: 24px;
            background: white;
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        }

        h1 {
            margin-top: 0;
            font-size: 28px;
        }

        .desc {
            color: #666;
            line-height: 1.6;
        }

        .panel {
            margin-top: 24px;
            padding: 20px;
            border: 1px solid #e8e8e8;
            border-radius: 14px;
            background: #fafafa;
        }

        label {
            font-weight: bold;
        }

        select, input {
            font-size: 16px;
            padding: 8px 10px;
            border-radius: 8px;
            border: 1px solid #ccc;
        }

        .row {
            margin-top: 16px;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
        }

        button {
            border: none;
            border-radius: 10px;
            padding: 12px 18px;
            font-size: 16px;
            cursor: pointer;
            background: #222;
            color: white;
        }

        button:hover {
            opacity: 0.86;
        }

        button.secondary {
            background: #666;
        }

        button.danger {
            background: #b42318;
        }

        button.ok {
            background: #087443;
        }

        .slider {
            width: 260px;
        }

        .dof-grid {
            margin-top: 16px;
            display: grid;
            grid-template-columns: 1fr;
            gap: 12px;
        }

        .dof-row {
            display: grid;
            grid-template-columns: 120px minmax(180px, 1fr) 84px 60px;
            gap: 12px;
            align-items: center;
        }

        .dof-row input {
            width: 100%;
            box-sizing: border-box;
        }

        .number-input {
            width: 84px;
            box-sizing: border-box;
        }

        pre {
            min-height: 180px;
            max-height: 360px;
            overflow: auto;
            padding: 16px;
            border-radius: 12px;
            background: #111;
            color: #eee;
            font-size: 14px;
            line-height: 1.5;
        }

        .small {
            color: #777;
            font-size: 14px;
        }
    </style>
</head>

<body>
<div class="container">
    <h1>Inspire RH56DFTP 灵巧手控制台</h1>

    <p class="desc">
        可通过局域网浏览器控制左手、右手或双手。角度规则：0 = 闭合，1000 = 张开。
    </p>

    <div class="panel">
        <div class="row">
            <label for="target">控制对象：</label>
            <select id="target">
                <option value="left">左手</option>
                <option value="right">右手</option>
                <option value="both" selected>双手</option>
            </select>
        </div>

        <div class="row">
            <button class="ok" onclick="sendCommand('open')">张开</button>
            <button class="danger" onclick="sendCommand('close')">闭合</button>
            <button class="secondary" onclick="sendPercent(50)">半开 50%</button>
            <button class="secondary" onclick="getState()">读取状态</button>
        </div>

        <div class="row">
            <label for="percent">自定义开合：</label>
            <input id="percent" class="slider" type="range" min="0" max="100" value="100"
                   oninput="setPercentValue(this.value)">
            <input id="percentNumber" class="number-input" type="number" min="0" max="100" value="100"
                   oninput="setPercentValue(this.value)">
            <span id="percentText">100%</span>
            <button onclick="sendPercent()">发送比例</button>
        </div>

        <p class="small">
            0% = 完全闭合，100% = 完全张开。建议调试时先用较低速度和力度。
        </p>

        <div class="dof-grid" id="dofControls"></div>

        <div class="row">
            <button onclick="sendDofPercent()">发送六自由度</button>
            <button class="secondary" onclick="setAllDof(100)">全部 100%</button>
            <button class="secondary" onclick="setAllDof(0)">全部 0%</button>
        </div>
    </div>

    <h3>返回信息</h3>
    <pre id="log">等待操作...</pre>
</div>

<script>
function getTarget() {
    return document.getElementById("target").value;
}

function writeLog(obj) {
    const log = document.getElementById("log");
    log.textContent = JSON.stringify(obj, null, 2);
}

async function postJson(url, data) {
    writeLog({
        status: "sending",
        request: data
    });

    try {
        const resp = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await resp.json();
        writeLog(result);
    } catch (err) {
        writeLog({
            ok: false,
            error: String(err)
        });
    }
}

async function sendCommand(action) {
    await postJson("/api/command", {
        target: getTarget(),
        action: action
    });
}

async function sendPercent(percent = null) {
    if (percent === null) {
        percent = Number(document.getElementById("percent").value);
    }
    percent = clampPercent(percent);
    setPercentValue(percent);

    await postJson("/api/command", {
        target: getTarget(),
        action: "percent",
        percent: percent
    });
}

const dofNames = [
    "小指",
    "无名指",
    "中指",
    "食指",
    "拇指弯曲",
    "拇指旋转"
];

function clampPercent(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
        return 0;
    }
    return Math.max(0, Math.min(100, Math.round(number)));
}

function setPercentValue(value) {
    const percent = clampPercent(value);
    document.getElementById("percent").value = String(percent);
    document.getElementById("percentNumber").value = String(percent);
    document.getElementById("percentText").innerText = percent + "%";
    setAllDof(percent);
}

function initDofControls() {
    const container = document.getElementById("dofControls");
    container.innerHTML = "";

    dofNames.forEach((name, index) => {
        const row = document.createElement("div");
        row.className = "dof-row";

        const label = document.createElement("label");
        label.htmlFor = "dof" + index;
        label.textContent = name;

        const input = document.createElement("input");
        input.id = "dof" + index;
        input.type = "range";
        input.min = "0";
        input.max = "100";
        input.value = "100";
        input.oninput = () => {
            setDofValue(index, input.value);
        };

        const number = document.createElement("input");
        number.id = "dofNumber" + index;
        number.className = "number-input";
        number.type = "number";
        number.min = "0";
        number.max = "100";
        number.value = "100";
        number.oninput = () => {
            setDofValue(index, number.value);
        };

        const value = document.createElement("span");
        value.id = "dofText" + index;
        value.textContent = "100%";

        row.appendChild(label);
        row.appendChild(input);
        row.appendChild(number);
        row.appendChild(value);
        container.appendChild(row);
    });
}

function getDofValues() {
    return dofNames.map((_, index) => Number(document.getElementById("dof" + index).value));
}

function setDofValue(index, rawValue) {
    const percent = clampPercent(rawValue);
    document.getElementById("dof" + index).value = String(percent);
    document.getElementById("dofNumber" + index).value = String(percent);
    document.getElementById("dofText" + index).innerText = percent + "%";
}

function setAllDof(percent) {
    dofNames.forEach((_, index) => {
        setDofValue(index, percent);
    });
}

async function sendDofPercent() {
    await postJson("/api/command", {
        target: getTarget(),
        action: "dof_percent",
        values: getDofValues()
    });
}

async function getState() {
    await postJson("/api/state", {
        target: getTarget()
    });
}

initDofControls();
</script>
</body>
</html>
"""


def get_target_sides(target):
    target = str(target).lower().strip()

    if target in ["left", "l"]:
        return ["left"]

    if target in ["right", "r"]:
        return ["right"]

    if target in ["both", "all", "双手"]:
        return ["left", "right"]

    raise ValueError("target must be left, right or both")


def execute_for_target(target, fn):
    sides = get_target_sides(target)
    results = {}

    for side in sides:
        if side not in hands:
            results[side] = {
                "ok": False,
                "error": f"{side} hand is not configured"
            }
            continue

        try:
            results[side] = {
                "ok": True,
                "data": fn(hands[side])
            }
        except Exception as e:
            results[side] = {
                "ok": False,
                "error": str(e)
            }

    all_ok = all(item["ok"] for item in results.values())

    return {
        "ok": all_ok,
        "target": target,
        "results": results,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE)


@app.route("/api/command", methods=["POST"])
def api_command():
    data = request.get_json(force=True) or {}

    target = data.get("target", "both")
    action = str(data.get("action", "open")).lower().strip()

    speed = int(data.get("speed", 3000))
    force = int(data.get("force", 12000))

    speed = max(0, min(4000, speed))
    force = max(0, min(12000, force))

    if action == "open":
        result = execute_for_target(
            target,
            lambda hand: hand.open(speed=speed, force=force)
        )

    elif action == "close":
        result = execute_for_target(
            target,
            lambda hand: hand.close(speed=speed, force=force)
        )

    elif action == "percent":
        percent = int(data.get("percent", 100))
        angle_values = open_percent_to_angle_values(percent)
        result = execute_for_target(
            target,
            lambda hand: hand.set_angle(angle_values, speed=speed, force=force)
        )

    elif action in ["dof_percent", "percent_values"]:
        angle_values = percent_to_angle_values(data.get("values"))
        result = execute_for_target(
            target,
            lambda hand: hand.set_angle(angle_values, speed=speed, force=force)
        )

    elif action == "angle":
        values = data.get("values")
        values = clamp_values(values, 0, 1000)
        result = execute_for_target(
            target,
            lambda hand: hand.set_angle(values, speed=speed, force=force)
        )

    else:
        return jsonify({
            "ok": False,
            "error": "action must be open, close, percent, dof_percent or angle"
        }), 400

    return jsonify(result)


@app.route("/api/state", methods=["POST", "GET"])
def api_state():
    if request.method == "POST":
        data = request.get_json(force=True) or {}
        target = data.get("target", "both")
    else:
        target = request.args.get("target", "both")

    result = execute_for_target(
        target,
        lambda hand: hand.get_state()
    )

    return jsonify(result)


@app.route("/api/ping", methods=["GET"])
def api_ping():
    return jsonify({
        "ok": True,
        "hands": {
            side: {
                "ip": hand.ip,
                "port": hand.port,
                "device_id": hand.device_id
            }
            for side, hand in hands.items()
        }
    })


def main():
    parser = argparse.ArgumentParser(description="Inspire RH56DFTP Web Control Service")

    parser.add_argument("--left", default="192.168.123.210", help="left hand IP")
    parser.add_argument("--right", default="192.168.123.211", help="right hand IP")
    parser.add_argument("--hand-port", type=int, default=6000, help="Modbus TCP port")
    parser.add_argument("--device-id", type=int, default=1, help="Modbus device id")

    parser.add_argument("--host", default="0.0.0.0", help="web service bind host")
    parser.add_argument("--web-port", type=int, default=8080, help="web service port")

    args = parser.parse_args()

    hands["left"] = InspireHand(
        side="left",
        ip=args.left,
        port=args.hand_port,
        device_id=args.device_id,
    )

    hands["right"] = InspireHand(
        side="right",
        ip=args.right,
        port=args.hand_port,
        device_id=args.device_id,
    )

    print("======================================")
    print("Inspire hand web service started")
    print(f"Left hand:  {args.left}:{args.hand_port}")
    print(f"Right hand: {args.right}:{args.hand_port}")
    print(f"Web:        http://{args.host}:{args.web_port}")
    print("LAN usage:  http://服务器局域网IP:8080")
    print("======================================")

    app.run(
        host=args.host,
        port=args.web_port,
        threaded=True,
    )


if __name__ == "__main__":
    main()
