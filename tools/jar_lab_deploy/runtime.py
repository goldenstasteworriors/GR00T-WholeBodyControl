"""Portable inference for ModelScope-exported future-observation JAR weights."""
from pathlib import Path
import gc
import json
import sys
import time

import numpy as np
import torch


from contract import JOINT_NAMES, CAMERAS


class Predictor:
    def __init__(self, bundle, task, checkpoint="best", device="cuda:0"):
        self.bundle = Path(bundle).resolve()
        sys.path.insert(0, str(self.bundle / "code"))
        from jar.olmo2 import load_olmo2_bundle
        from jar.uniform_action_tokenizer import OLMo2UniformTokenizer
        from jar.data import build_prefix
        from jar.uniform_inference import generate_uniform_chunk
        self.build_prefix, self.generate = build_prefix, generate_uniform_chunk
        catalog = json.loads((self.bundle / "export_manifest.json").read_text())
        operation = next((r for r in catalog["operations"] if r["task"] == task), None)
        if operation is None:
            raise ValueError(f"Operation has no completed export: {task}")
        candidates = operation["checkpoints"]
        selected = next((w for w in candidates if w.get(checkpoint)), None) if checkpoint in ("best", "latest") else next((w for w in candidates if str(w["step"]) == checkpoint.removeprefix("step")), None)
        if selected is None:
            raise ValueError(f"Unknown checkpoint: {checkpoint}")
        weights = (self.bundle / selected["path"]).resolve()
        if not weights.is_relative_to(self.bundle) or weights.stat().st_size != selected["size"]:
            raise ValueError("Invalid checkpoint path/size")
        self.contract = json.loads((self.bundle / task / "observation_representation.json").read_text())
        if (self.contract["joint_names"] != JOINT_NAMES or self.contract["source_indices"] != list(range(15, 31))
                or self.contract["fps"] != 30 or self.contract["horizon"] != 32 or self.contract["bins"] != 512):
            raise ValueError("Checkpoint does not use the verified G1 Dex1 observation contract")
        instructions = {r["instruction"] for r in self.contract["rows"]}
        if len(instructions) != 1:
            raise ValueError("Operation must have one recorded English instruction")
        self.instruction = instructions.pop()
        self.text = f"Robot: Unitree G1 with Dex1 grippers. Control rate: 30 Hz. {self.instruction}"
        self.device = torch.device(device)
        torch.set_num_threads(4)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.matmul.allow_tf32 = True
        self.model, self.config = load_olmo2_bundle(weights, device=device)
        if self.config.state_dim != 16 or self.config.action_dim != 16:
            raise ValueError("Expected 16-dimensional upper-body observations")
        self.config.text_tokenizer_path = str(self.bundle / "tokenizer")
        self.tokenizer = OLMo2UniformTokenizer(self.config.text_tokenizer_path, bins=512, action_dim=16)
        self.low, self.high = (np.asarray(self.contract["normalization"][k], dtype=np.float32) for k in ("low", "high"))
        self.task, self.weights = task, str(weights)

    def normalize(self, state):
        span = self.high - self.low
        values = 2 * (state - self.low) / np.maximum(span, 1e-6) - 1
        values[span < 1e-6] = 0
        return np.clip(values, -1, 1).astype(np.float32)

    @torch.inference_mode()
    def predict(self, state, images, frame_times, steps=1):
        state, images, ages = np.asarray(state, dtype=np.float32), np.asarray(images), np.asarray(frame_times, dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("state must be finite observed radians in canonical 16D order")
        if images.ndim != 5 or images.shape[1:] != (3, 3, 240, 240) or not 1 <= len(images) <= 4 or images.dtype != np.uint8:
            raise ValueError("images must be uint8 [history, ego/left/right, RGB,240,240]")
        if ages.shape != (len(images), 3) or not np.isfinite(ages).all() or np.any(ages > 1e-7):
            raise ValueError("camera ages must be finite, nonpositive [history,3]")
        if steps not in (1, 8, 32):
            raise ValueError("Use an explicitly validated 1, 8 or 32-step autoregressive prefix")
        views = images.reshape(-1, 3, 240, 240)
        prefix = self.build_prefix(self.tokenizer, self.config, self.text, len(views))
        observations = dict(state=torch.from_numpy(self.normalize(state))[None].to(self.device),
            images=torch.from_numpy(np.ascontiguousarray(views))[None].to(self.device),
            image_positions=prefix["image_positions"][None].to(self.device),
            state_positions=prefix["state_positions"][None].to(self.device),
            camera_ids=torch.arange(3, device=self.device).repeat(len(images))[None],
            frame_times=torch.from_numpy(ages.reshape(-1).copy())[None].to(self.device))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.monotonic()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            normalized, ids = self.generate(self.model, prefix["input_ids"][None].to(self.device),
                                             observations, self.tokenizer, horizon=steps)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        positions = (normalized + 1) * (self.high - self.low) / 2 + self.low
        if not np.isfinite(positions).all():
            raise RuntimeError("Nonfinite predicted observations")
        return dict(positions=positions.astype(np.float32), token_ids=ids[0].cpu().numpy(),
                    inference_seconds=time.monotonic() - started, task=self.task,
                    semantics="future observed position reference; not an actuator command")

    def close(self):
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
