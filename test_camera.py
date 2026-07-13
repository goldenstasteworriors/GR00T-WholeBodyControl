import time
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

print("starting RGB stream")
pipeline.start(config)

ok = 0
errors = 0
t0 = time.time()
last = t0

try:
    while time.time() - t0 < 300:
        try:
            frames = pipeline.wait_for_frames(2000)
            if frames.get_color_frame():
                ok += 1
        except Exception as e:
            errors += 1
            print("wait_error", errors, "at", round(time.time() - t0, 2), e)

        now = time.time()
        if now - last >= 10:
            print("progress", round(now - t0, 1), "ok", ok, "errors", errors)
            last = now

    print("done", "ok", ok, "errors", errors, "fps", ok / max(time.time() - t0, 1))
finally:
    pipeline.stop()