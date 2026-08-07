#!/usr/bin/env python3
"""Record all public RH56DFTP tactile taxels without sending hand commands."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from pymodbus.client import ModbusTcpClient

# (dataset name, Modbus address, rows, columns), per PRJ-02-TS-U-010 §2.6.20.
TACTILE_SPECS = (
    ("little_end", 3000, 3, 3), ("little_tip", 3018, 12, 8), ("little_pad", 3210, 10, 8),
    ("ring_end", 3370, 3, 3), ("ring_tip", 3388, 12, 8), ("ring_pad", 3580, 10, 8),
    ("middle_end", 3740, 3, 3), ("middle_tip", 3758, 12, 8), ("middle_pad", 3950, 10, 8),
    ("index_end", 4110, 3, 3), ("index_tip", 4128, 12, 8), ("index_pad", 4320, 10, 8),
    ("thumb_end", 4480, 3, 3), ("thumb_tip", 4498, 12, 8), ("thumb_mid", 4690, 3, 3), ("thumb_pad", 4708, 12, 8),
    ("palm", 4900, 8, 14),
)
FORCE_ACT_ADDRESS = 1582


def read_holding(client: ModbusTcpClient, address: int, count: int, device_id: int) -> list[int]:
    for key in ("device_id", "slave", "unit"):
        try:
            reply = client.read_holding_registers(address=address, count=count, **{key: device_id})
            break
        except TypeError:
            reply = None
    if reply is None or reply.isError():
        raise RuntimeError(f"Modbus read failed at address {address}")
    return reply.registers


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only RH56DFTP tactile recorder")
    parser.add_argument("--hand", default="192.168.123.210")
    parser.add_argument("--hand-port", type=int, default=6000)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--duration", type=float, required=True, help="recording duration in seconds")
    parser.add_argument("--hz", type=float, default=5.0, help="sampling rate, 1-10 Hz (default: 5)")
    parser.add_argument("--output-dir", type=Path, default=Path("output/tactile_records"))
    args = parser.parse_args()
    if args.duration <= 0 or not 1 <= args.hz <= 10:
        raise SystemExit("--duration must be positive and --hz must be in [1, 10]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = ModbusTcpClient(args.hand, port=args.hand_port, timeout=1.0)
    if not client.connect():
        raise SystemExit(f"cannot connect to {args.hand}:{args.hand_port}")

    frames: dict[str, list[np.ndarray]] = {name: [] for name, *_ in TACTILE_SPECS}
    force_frames: list[np.ndarray] = []
    timestamps: list[float] = []
    read_ms: list[float] = []
    period, deadline = 1.0 / args.hz, time.monotonic() + args.duration
    print(f"Recording {args.duration:g}s at {args.hz:g} Hz from {args.hand}:{args.hand_port} (read-only)")
    try:
        while time.monotonic() < deadline:
            started = time.perf_counter()
            frame = {name: np.asarray(read_holding(client, address, rows * cols, args.device_id), dtype=np.uint16).reshape(rows, cols) for name, address, rows, cols in TACTILE_SPECS}
            force = np.asarray(read_holding(client, FORCE_ACT_ADDRESS, 6, args.device_id), dtype=np.uint16).view(np.int16)
            timestamps.append(time.time())
            for name, values in frame.items(): frames[name].append(values)
            force_frames.append(force)
            elapsed = time.perf_counter() - started
            read_ms.append(elapsed * 1000)
            time.sleep(max(0.0, period - elapsed))
    finally:
        client.close()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_dir / f"rh56dftp_tactile_{stamp}.npz"
    arrays = {name: np.stack(samples) for name, samples in frames.items()}
    np.savez_compressed(output, timestamps_s=np.asarray(timestamps), read_ms=np.asarray(read_ms), force_act_g=np.stack(force_frames), **arrays)
    summary = {name: {"max": int(values.max()), "nonzero_taxels": int(np.count_nonzero(values))} for name, values in arrays.items()}
    metadata = {"npz": output.name, "frames": len(timestamps), "requested_hz": args.hz, "mean_read_ms": float(np.mean(read_ms)), "p95_read_ms": float(np.percentile(read_ms, 95)), "tactile_shapes": {name: list(values.shape[1:]) for name, values in arrays.items()}, "summary": summary}
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved {output} ({len(timestamps)} frames); metadata {metadata_path}")
    print(f"mean={metadata['mean_read_ms']:.2f}ms p95={metadata['p95_read_ms']:.2f}ms")


if __name__ == "__main__":
    main()
