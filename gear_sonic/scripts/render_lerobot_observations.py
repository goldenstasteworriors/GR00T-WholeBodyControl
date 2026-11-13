"""Render recorded G1 body observations beside RGB and measured BrainCo hand bars.

This is offline kinematic playback: no physics controller, DDS, or ZMQ publisher
is created. Body mapping and root-pose assignment reuse replay_lerobot_mujoco.
Root translation is fixed because this dataset does not record it. No substitute
hand geometry is animated as a BrainCo hand.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np

from gear_sonic.scripts import replay_lerobot_mujoco as replay


WIDTH, HEIGHT = 1280, 720
VIEW_WIDTH, VIEW_HEIGHT = 640, 480
HAND_NAMES = ("thumb", "thumb_aux", "index", "middle", "ring", "pinky")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument(
        "--body-model", type=Path,
        default=replay.REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml",
    )
    parser.add_argument("--root-position", type=float, nargs=3, default=(0.0, 0.0, 0.8))
    parser.add_argument("--azimuth", type=float, default=300.0)
    parser.add_argument("--elevation", type=float, default=-15.0)
    parser.add_argument("--distance", type=float, default=2.2)
    return parser


def load_render_model(path: Path) -> mujoco.MjModel:
    """Add a rendering scene in memory without changing the canonical asset."""
    path = path.expanduser().resolve()
    tree = ET.parse(path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        for key in ("meshdir", "texturedir"):
            if compiler.get(key):
                compiler.set(key, str((path.parent / compiler.get(key)).resolve()))
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="1 -2 3", dir="-0.3 0.4 -1", directional="true")
    ET.SubElement(world, "light", pos="-2 1 2", dir="1 -0.2 -1", diffuse="0.35 0.35 0.35")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="replay_grid", type="2d", builtin="checker",
                  rgb1="0.22 0.25 0.30", rgb2="0.28 0.31 0.36", width="256", height="256")
    ET.SubElement(asset, "material", name="replay_floor", texture="replay_grid",
                  texrepeat="10 10", texuniform="true")
    # MJCF can contain multiple worldbody sections; avoid coplanar floor geoms.
    for section in root.findall("worldbody"):
        for plane in section.findall("geom[@type='plane']"):
            section.remove(plane)
    ET.SubElement(world, "geom", type="plane", size="4 4 0.05",
                  material="replay_floor", contype="0", conaffinity="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.vis.global_.offwidth = VIEW_WIDTH
    model.vis.global_.offheight = VIEW_HEIGHT
    model.vis.headlight.ambient[:] = (0.35, 0.35, 0.35)
    # The canonical 29-DOF model has fixed rubber hands, not Revo2 geometry.
    for geom in range(model.ngeom):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[geom]))
            if mesh_name in ("left_rubber_hand", "right_rubber_hand"):
                model.geom_rgba[geom, 3] = 0.0
    return model


def text(image: np.ndarray, value: str, x: int, y: int, scale: float = 0.5,
         color: tuple[int, int, int] = (225, 230, 238)) -> None:
    cv2.putText(image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def draw_hand_bars(canvas: np.ndarray, names: list[str], values: np.ndarray) -> None:
    text(canvas, "BrainCo measured observation: 0 = open, 1 = closed", 16, 550, 0.48)
    lookup = dict(zip(names, values))
    for side_index, side in enumerate(("left", "right")):
        origin = 16 + side_index * 316
        color = (75, 210, 220) if side == "left" else (245, 165, 75)
        text(canvas, side.upper(), origin, 571, 0.46, color)
        for index, motor in enumerate(HAND_NAMES):
            value = float(lookup[f"{side}_hand_{motor}"])
            y = 581 + index * 18
            text(canvas, motor, origin, y + 11, 0.36)
            cv2.rectangle(canvas, (origin + 85, y), (origin + 247, y + 10), (48, 56, 68), -1)
            length = int(162 * np.clip(value, 0, 1))
            if length:
                cv2.rectangle(canvas, (origin + 85, y), (origin + 85 + length, y + 10), color, -1)
            text(canvas, f"{value:.3f}", origin + 254, y + 10, 0.36, color)


def render(args: argparse.Namespace) -> None:
    if not args.ffmpeg:
        raise ValueError("Provide an installed ffmpeg binary using --ffmpeg")
    base_argv = ["--dataset-path", str(args.dataset_path), "--episode-index", str(args.episode_index),
                 "--mode", "joint", "--headless", "--start-frame", str(args.start_frame)]
    if args.max_frames is not None:
        base_argv += ["--max-frames", str(args.max_frames)]
    base_args = replay._build_parser().parse_args(base_argv)
    episode = replay._load_episode(base_args)
    names = replay._joint_names_for_column(base_args, episode)
    hand_keys = {f"{side}_hand_{name}" for side in ("left", "right") for name in HAND_NAMES}
    if not hand_keys.issubset(names):
        raise ValueError("This renderer expects the twelve named BrainCo Revo2 motor observations")

    model = load_render_model(args.body_model)
    data = mujoco.MjData(model)
    body_names = [name for name in names if "_hand_" not in name]
    if len(body_names) != 29:
        raise ValueError(f"Expected 29 named body joints, got {len(body_names)}")
    body_indices = [names.index(name) for name in body_names]
    mapping = replay._build_qpos_mapping(model, body_names)

    camera_pattern = episode.info["video_path"]
    source_video = episode.dataset_path / camera_pattern.format(
        episode_chunk=args.episode_index // episode.info.get("chunks_size", 1000),
        episode_index=args.episode_index, video_key="observation.images.ego_view",
    )
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot decode recorded camera video: {source_video}")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".ffmpeg.log")
    cmd = [str(args.ffmpeg), "-hide_banner", "-loglevel", "warning", "-n",
           "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}",
           "-framerate", f"{episode.fps:g}", "-i", "pipe:0", "-an", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "21", "-threads", "2", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(output)]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = args.azimuth, args.elevation, args.distance
    cam.lookat[:] = (args.root_position[0], args.root_position[1], 0.72)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[1] = 1  # canonical mesh visuals, not duplicate collision geoms
    option.geomgroup[0] = 1  # floor and the canonical geoms not assigned to group 1
    max_joint_error = 0.0
    current_camera_index = 0
    first_source_frame = int(episode.frame_table.iloc[0]["frame_index"])
    if first_source_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first_source_frame)
        current_camera_index = first_source_frame
    started = time.monotonic()
    first_preview = output.with_suffix(".preview.jpg")
    with log_path.open("wb") as log, mujoco.Renderer(model, height=VIEW_HEIGHT, width=VIEW_WIDTH) as renderer:
        encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
        try:
            for offset, (_, row) in enumerate(episode.frame_table.iterrows()):
                values = np.asarray(row["observation.state"], dtype=np.float64)
                if values.shape != (len(names),) or not np.all(np.isfinite(values)):
                    raise ValueError(f"Invalid observation at selected frame {offset}")
                for body_index, qpos_index, _ in mapping:
                    data.qpos[qpos_index] = values[body_indices[body_index]]
                replay._set_free_root_pose(model, data, args.root_position, row["observation.root_orientation"])
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                max_joint_error = max(max_joint_error, max(
                    abs(float(data.qpos[address]) - float(values[body_indices[index]]))
                    for index, address, _ in mapping
                ))
                source_frame = int(row["frame_index"])
                if source_frame != current_camera_index:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
                ok, bgr = capture.read()
                if not ok:
                    raise ValueError(f"Camera frame {source_frame} is missing")
                current_camera_index = source_frame + 1
                renderer.update_scene(data, camera=cam, scene_option=option)
                body_rgb = renderer.render()
                camera_rgb = cv2.cvtColor(cv2.resize(bgr, (VIEW_WIDTH, VIEW_HEIGHT)), cv2.COLOR_BGR2RGB)
                canvas = np.full((HEIGHT, WIDTH, 3), (18, 23, 31), dtype=np.uint8)
                canvas[48:528, :640] = body_rgb
                canvas[48:528, 640:] = camera_rgb
                text(canvas, "MuJoCo: observation.state / 29 body joints", 16, 30, 0.58)
                text(canvas, "Recorded ego_view / same source frame", 656, 30, 0.58)
                cv2.line(canvas, (640, 48), (640, HEIGHT), (65, 75, 90), 1)
                draw_hand_bars(canvas, names, values)
                text(canvas, f"{episode.dataset_path.name} | episode {args.episode_index}", 660, 552, 0.43)
                text(canvas, f"Frame {source_frame:05d}    recorded t={float(row['timestamp']):.2f}s", 660, 579)
                text(canvas, "Kinematic playback; no policy or physics control", 660, 607, 0.46)
                text(canvas, "Root translation FIXED (not recorded in dataset)", 660, 631, 0.46)
                text(canvas, "Revo2 hand geometry not rendered; measured bars only", 660, 655, 0.43)
                age = np.asarray(row["observation.brainco_state_age_s"]) * 1000
                text(canvas, f"Hand state age: L {age[0]:.1f} ms / R {age[1]:.1f} ms", 660, 680, 0.46)
                text(canvas, "Camera is frame-paired, without time-shift correction", 660, 704, 0.43)
                if offset == 0:
                    cv2.imwrite(str(first_preview), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                encoder.stdin.write(canvas.tobytes())
                if offset % 250 == 0:
                    print(f"Rendered {offset + 1}/{len(episode.frame_table)} frames, "
                          f"elapsed={time.monotonic() - started:.1f}s", flush=True)
        finally:
            capture.release()
            encoder.stdin.close()
            returncode = encoder.wait()
        if returncode:
            raise RuntimeError(f"ffmpeg exited {returncode}; see {log_path}")
    manifest = {
        "dataset": str(episode.dataset_path), "episode": args.episode_index,
        "source_column": "observation.state", "frames": len(episode.frame_table),
        "fps": episode.fps, "output": str(output), "body_model": str(args.body_model.resolve()),
        "body_joint_names": body_names, "max_joint_roundtrip_error_rad": max_joint_error,
        "root_orientation_recorded": True, "root_translation_recorded": False,
        "fixed_root_position": list(args.root_position), "hand_geometry_rendered": False,
        "hand_bars_source": "observation.state with metadata joint names",
        "camera_pairing": "source frame_index, unshifted", "physics_or_policy_run": False,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output}; maximum mapped-joint error={max_joint_error:g}", flush=True)


if __name__ == "__main__":
    render(build_parser().parse_args())
