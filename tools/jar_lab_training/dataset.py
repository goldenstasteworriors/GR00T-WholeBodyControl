"""Causal observation prefixes and strictly future observation labels."""
from collections import OrderedDict
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from jar.data import build_prefix, history_indices
from jar.mixed_cache import decode_rgb
from jar.uniform_action_tokenizer import OLMo2UniformTokenizer


def normalize(values, statistics):
    low, high = (np.asarray(statistics[k], dtype=np.float32) for k in ("low", "high"))
    span = high - low
    result = 2 * (np.asarray(values, dtype=np.float32) - low) / np.maximum(span, 1e-6) - 1
    result[..., span < 1e-6] = 0
    return np.clip(result, -1, 1).astype(np.float32)


def denormalize(values, statistics):
    low, high = (np.asarray(statistics[k], dtype=np.float32) for k in ("low", "high"))
    return (np.asarray(values) + 1) * (high - low) / 2 + low


class ObservationDataset(Dataset):
    def __init__(self, cache, task, config, split):
        self.cache = Path(cache)
        if not (self.cache / "complete.json").is_file():
            raise ValueError("Dataset preparation incomplete")
        self.manifest = json.loads((self.cache / task / "manifest.json").read_text())
        self.config = config
        self.tokenizer = OLMo2UniformTokenizer(config.text_tokenizer_path, bins=512, action_dim=16)
        self.rows = [r for r in self.manifest["rows"] if r["episode"] in self.manifest[split]]
        assert not set(self.manifest["train"]) & set(self.manifest["validation"])
        self.numeric = []
        for row in self.rows:
            with np.load(self.cache / "episodes" / str(row["episode"]) / "observations.npz") as data:
                self.numeric.append({k: data[k] for k in data.files})
        self.offsets = np.concatenate(([0], np.cumsum([r["length"] - 1 for r in self.rows])))
        self.files, self.frames, self.prefixes = OrderedDict(), OrderedDict(), {}

    def __len__(self):
        return int(self.offsets[-1])

    def locate(self, index):
        row = int(np.searchsorted(self.offsets, index, side="right") - 1)
        return row, int(index - self.offsets[row])

    def sample_at(self, row, cut):
        record, ep = self.rows[row], self.numeric[row]
        n, horizon = record["length"], self.manifest["horizon"]
        if not 0 <= cut < n - 1:
            raise ValueError("Every cut must have a strictly future observation")
        history = history_indices(cut, self.config.history_frames, self.config.history_stride)
        if row not in self.files:
            folder = self.cache / "episodes" / str(record["episode"])
            self.files[row] = (os.open(folder / "rgb.lz4", os.O_RDONLY), np.load(folder / "rgb-offsets.npy"))
            if len(self.files) > 16:
                _, old = self.files.popitem(last=False)
                os.close(old[0])
        self.files.move_to_end(row)
        fd, offsets = self.files[row]
        pictures = []
        for index in history:
            key = row, int(index)
            if key not in self.frames:
                a, b = map(int, offsets[index:index + 2])
                self.frames[key] = decode_rgb(os.pread(fd, b - a, a), self.config.image_size)
                if len(self.frames) > 64:
                    self.frames.popitem(last=False)
            self.frames.move_to_end(key)
            pictures.append(self.frames[key])
        images = torch.from_numpy(np.stack(pictures).transpose(0, 1, 4, 2, 3).reshape(-1, 3, 240, 240))
        text = f"Robot: Unitree G1 with Dex1 grippers. Control rate: 30 Hz. {record['instruction']}"
        key = text, len(images)
        if key not in self.prefixes:
            self.prefixes[key] = build_prefix(self.tokenizer, self.config, text, len(images))
        prefix = self.prefixes[key]
        indices = np.minimum(cut + 1 + np.arange(horizon), n - 1)
        valid = min(horizon, n - cut - 1)
        future = normalize(ep["state"][indices], self.manifest["normalization"])
        target = torch.from_numpy(self.tokenizer.encode_actions(future))
        labels = target.clone()
        labels[valid * 16:] = -100
        age = ep["camera_time"][history].reshape(-1) - ep["timestamp"][cut]
        if np.any(age > 1e-7):
            raise ValueError("Future image leakage")
        return dict(prefix, input_ids=torch.cat((prefix["input_ids"], target)),
                    labels=torch.cat((torch.full_like(prefix["input_ids"], -100), labels)), images=images,
                    state=torch.from_numpy(normalize(ep["state"][cut], self.manifest["normalization"])),
                    camera_ids=torch.arange(3).repeat(len(history)), frame_times=torch.tensor(np.minimum(age, 0), dtype=torch.float32),
                    metadata=dict(episode=record["episode"], cut=cut, valid_steps=valid, target_tokens=valid * 16,
                                  first_target_frame=cut + 1, last_history_frame=int(history[-1])))

    def __getitem__(self, index):
        return self.sample_at(*self.locate(index))


class GlobalBatchSampler(Sampler):
    """Use every cut exactly once, merging a tail too small for all ranks."""
    def __init__(self, size, batch, rank, world, epoch=0):
        if size < world or batch < world:
            raise ValueError("Insufficient samples per distributed batch")
        self.size, self.batch, self.rank, self.world, self.epoch = size, batch, rank, world, epoch

    def boundaries(self):
        starts = list(range(0, self.size, self.batch))
        if len(starts) > 1 and self.size - starts[-1] < self.world:
            starts.pop()
        return starts + [self.size]

    def __len__(self):
        return len(self.boundaries()) - 1

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(6198 + self.epoch)).tolist()
        bounds = self.boundaries()
        for a, b in zip(bounds[:-1], bounds[1:]):
            yield order[a:b][self.rank::self.world]


def validation_indices(dataset, fractions=(0., .25, .5, .75)):
    return sorted({int(dataset.offsets[row]) + min(int(f * (r["length"] - 1)), r["length"] - 2)
                   for row, r in enumerate(dataset.rows) for f in fractions})
