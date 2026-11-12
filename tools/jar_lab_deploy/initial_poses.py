"""Recorded achieved arm/Dex1 poses; never use action labels or recorded legs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from contract import JOINT_NAMES, TASKS


class PoseCatalog:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads(self.path.read_text())
        if self.meta["joint_names"] != JOINT_NAMES or self.meta["source_indices"] != list(range(15,31)):
            raise ValueError("Initial poses must use the observation-only JAR 16D order")
        data = self.path.with_name(self.meta["array_file"])
        if hashlib.sha256(data.read_bytes()).hexdigest() != self.meta["sha256"]:
            raise ValueError("Initial-pose catalog checksum mismatch")
        self.arrays = np.load(data, allow_pickle=False)
        self.tasks = [task for task in TASKS if any(e["task"] == task for e in self.meta["episodes"])]
        self.entries = {(e["task"], e["episode"]): e for e in self.meta["episodes"]}

    def pose(self, selection):
        skill, episode, frame = selection["task"], selection["episode"], selection["frame"]
        entry = self.entries[(skill, episode)]
        if type(frame) is not int or not 0 <= frame < entry["frames"]:
            raise ValueError(f"Frame must be 0..{entry['frames']-1}")
        pose = self.arrays[entry["key"]][frame].astype(float)
        if pose.shape != (16,) or not np.isfinite(pose).all():
            raise ValueError("Invalid achieved pose")
        return pose

    def menu(self):
        lines = ["p: select initial pose; input TASK_NUMBER EPISODE FRAME, then Enter (0-based episode/frame)."]
        for i, skill in enumerate(self.tasks):
            episodes = [f"{e['episode']}(0..{e['frames']-1})" for e in self.meta["episodes"] if e["task"] == skill]
            lines.append(f"{i}: {skill}: " + ", ".join(episodes))
        return "\n".join(lines)

    def parse(self, value):
        parts = value.split()
        if len(parts) != 3 or any(not p.isdecimal() for p in parts):
            raise ValueError("Enter three nonnegative integers: task_number episode frame")
        sid, episode, frame = map(int, parts)
        selection = dict(task=self.tasks[sid], episode=episode, frame=frame)
        self.pose(selection)
        return selection

    def load_selection(self, path, default_task="TAKE_TUBE"):
        path = Path(path)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["catalog_sha256"] != self.meta["sha256"]:
                raise ValueError("Saved initial pose belongs to a different catalog")
            selection = saved["selection"]
        else:
            entry = next(e for e in self.meta["episodes"] if e["task"] == default_task)
            selection = dict(task=default_task, episode=entry["episode"], frame=0)
        self.pose(selection)
        return selection

    def save_selection(self, path, selection):
        self.pose(selection)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(json.dumps(dict(catalog_sha256=self.meta["sha256"], selection=selection), indent=2) + "\n")
        temporary.replace(path)
