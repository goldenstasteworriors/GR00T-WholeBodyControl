"""Display NaVILA predictions without importing or starting a robot controller."""
import argparse
from pathlib import Path
import signal
import time

import cv2

from deploy.navila.bridge import Camera, encode_image, infer, log, parse_action, sample_history

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", default="Find the beaker and stop in front of it.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=54321)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Image path on ThinkPad; default: saved G1 snapshot")
    source.add_argument("--live", action="store_true", help="Read the existing camera stream only")
    parser.add_argument("--camera-host", default="192.168.123.164")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--camera-key", default="ego_view")
    parser.add_argument("--steps", type=int, default=1, help="Prediction count; 0 means until Ctrl+C")
    parser.add_argument("--interval", type=float, default=0.5, help="Pause between predictions")
    args = parser.parse_args()
    camera = None
    history = []
    image_path = args.image or ROOT / ".runtime/navila/ego-view.jpg"
    print("PREVIEW ONLY: no controller, no ROS commands, no action execution.", flush=True)
    print(f"Instruction: {args.instruction}", flush=True)
    print(f"Source: {'live ego camera' if args.live else str(image_path)}", flush=True)
    try:
        if args.live:
            camera = Camera(args.camera_host, args.camera_port, args.camera_key)
            while camera.sample is None:
                if camera.error:
                    raise RuntimeError(camera.error)
                time.sleep(0.05)
        else:
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                raise ValueError(f"Cannot read image on ThinkPad: {image_path}")
            encoded = encode_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            shape = list(bgr.shape)
        step = 0
        while args.steps == 0 or step < args.steps:
            stamp = None
            if camera:
                encoded, shape, stamp = camera.snapshot()
            history.append(encoded)
            response, latency = infer(args.host, args.port, sample_history(history), args.instruction)
            # Always show the raw response, including text outside the action grammar.
            action, amount, parse_error = None, None, None
            try:
                action, amount = parse_action(response)
            except ValueError as exc:
                parse_error = str(exc)
            step += 1
            log("preview", step=step, response=response, parsed_action=action, amount=amount,
                parse_error=parse_error, round_trip_s=latency, execute=False,
                source="live" if camera else "image", camera_shape=shape,
                camera_source_timestamp=stamp)
            if args.steps == 0 or step < args.steps:
                time.sleep(args.interval)
    finally:
        if camera:
            camera.close()


if __name__ == "__main__":
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        main()
    except KeyboardInterrupt:
        print("Preview stopped. No robot command was sent.", flush=True)
