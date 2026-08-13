"""RH56DFTP tactile layout and wire-format helpers.

The bridge updates one Modbus batch at a time.  Consumers receive a complete
latest-value snapshot together with per-region timestamps, so a failed or
deferred read never erases previously acquired taxels.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import msgpack
import msgpack_numpy as mnp
import numpy as np


TACTILE_TOPIC = b"inspire_tactile"
TACTILE_PROTOCOL_VERSION = 1
MODBUS_METRIC_NAMES = (
    "target_full_refresh_hz",
    "actual_full_refresh_hz",
    "tactile_success_hz",
    "tactile_error_hz",
    "tactile_io_mean_ms",
    "tactile_io_p95_ms",
    "state_cycle_hz",
    "state_io_p95_ms",
    "lock_wait_p95_ms",
    "modbus_busy_ratio",
    "state_deadline_miss_ratio",
    "estimated_safe_max_full_refresh_hz",
)


@dataclass(frozen=True)
class TactileRegion:
    name: str
    address: int
    rows: int
    cols: int

    @property
    def size(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class TactileBatch:
    name: str
    address: int
    count: int
    region_names: tuple[str, ...]


TACTILE_REGIONS = (
    TactileRegion("little_end", 3000, 3, 3),
    TactileRegion("little_tip", 3018, 12, 8),
    TactileRegion("little_pad", 3210, 10, 8),
    TactileRegion("ring_end", 3370, 3, 3),
    TactileRegion("ring_tip", 3388, 12, 8),
    TactileRegion("ring_pad", 3580, 10, 8),
    TactileRegion("middle_end", 3740, 3, 3),
    TactileRegion("middle_tip", 3758, 12, 8),
    TactileRegion("middle_pad", 3950, 10, 8),
    TactileRegion("index_end", 4110, 3, 3),
    TactileRegion("index_tip", 4128, 12, 8),
    TactileRegion("index_pad", 4320, 10, 8),
    TactileRegion("thumb_end", 4480, 3, 3),
    TactileRegion("thumb_tip", 4498, 12, 8),
    TactileRegion("thumb_mid", 4690, 3, 3),
    TactileRegion("thumb_pad", 4708, 12, 8),
    TactileRegion("palm", 4900, 8, 14),
)

TACTILE_REGION_BY_NAME = {region.name: region for region in TACTILE_REGIONS}
TACTILE_REGION_NAMES = tuple(region.name for region in TACTILE_REGIONS)
TACTILE_REGION_INDEX_BY_NAME = {
    name: index for index, name in enumerate(TACTILE_REGION_NAMES)
}
TACTILE_REGION_COUNT = len(TACTILE_REGIONS)
TACTILE_TAXEL_COUNT = sum(region.size for region in TACTILE_REGIONS)
TACTILE_FORCE_COUNT = 6

# Each read stays below Modbus function-03's common 125-register limit.  Gaps
# between public matrices are included in the response but discarded locally.
TACTILE_BATCHES = (
    TactileBatch("little_end_tip", 3000, 114, ("little_end", "little_tip")),
    TactileBatch("little_pad", 3210, 80, ("little_pad",)),
    TactileBatch("ring_end_tip", 3370, 114, ("ring_end", "ring_tip")),
    TactileBatch("ring_pad", 3580, 80, ("ring_pad",)),
    TactileBatch("middle_end_tip", 3740, 114, ("middle_end", "middle_tip")),
    TactileBatch("middle_pad", 3950, 80, ("middle_pad",)),
    TactileBatch("index_end_tip", 4110, 114, ("index_end", "index_tip")),
    TactileBatch("index_pad", 4320, 80, ("index_pad",)),
    TactileBatch("thumb_end_tip", 4480, 114, ("thumb_end", "thumb_tip")),
    TactileBatch("thumb_mid_pad", 4690, 114, ("thumb_mid", "thumb_pad")),
    TactileBatch("palm", 4900, 112, ("palm",)),
)
TACTILE_BATCH_COUNT_WITH_FORCE = len(TACTILE_BATCHES) + 1


def unpack_batch(batch: TactileBatch, registers: list[int] | np.ndarray) -> dict[str, np.ndarray]:
    """Extract public matrices from one contiguous Modbus response."""
    raw = np.asarray(registers, dtype=np.uint16)
    if raw.shape != (batch.count,):
        raise ValueError(f"{batch.name}: expected {batch.count} registers, got {raw.shape}")

    result = {}
    for region_name in batch.region_names:
        region = TACTILE_REGION_BY_NAME[region_name]
        start = region.address - batch.address
        result[region_name] = raw[start : start + region.size].copy()
    return result


def flatten_regions(region_values: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten all regions in the stable manual order used by the dataset."""
    return np.concatenate(
        [np.asarray(region_values[region.name], dtype=np.uint16).reshape(-1) for region in TACTILE_REGIONS]
    )


def empty_snapshot() -> dict:
    """Create an invalid snapshot that is safe to write before the first read."""
    return {
        "version": TACTILE_PROTOCOL_VERSION,
        "sequence": 0,
        "publish_time_s": 0.0,
        "values": np.zeros(TACTILE_TAXEL_COUNT, dtype=np.uint16),
        "valid": np.zeros(TACTILE_REGION_COUNT, dtype=np.bool_),
        "updated_time_s": np.zeros(TACTILE_REGION_COUNT, dtype=np.float64),
        "update_sequence": np.zeros(TACTILE_REGION_COUNT, dtype=np.int64),
        "force_act_g": np.zeros(TACTILE_FORCE_COUNT, dtype=np.int16),
        "force_valid": False,
        "force_updated_time_s": 0.0,
        "force_update_sequence": 0,
        "metrics": np.zeros(len(MODBUS_METRIC_NAMES), dtype=np.float32),
    }


def snapshot_frame_fields(snapshot: dict, *, now_s: float | None = None) -> dict[str, np.ndarray]:
    """Convert a bridge snapshot into fixed-shape LeRobot frame fields."""
    now_s = time.time() if now_s is None else now_s
    updated_s = np.asarray(snapshot["updated_time_s"], dtype=np.float64)
    valid = np.asarray(snapshot["valid"], dtype=np.bool_)
    age_ms = np.where(valid, np.maximum(0.0, now_s - updated_s) * 1000.0, -1.0)
    force_valid = bool(snapshot["force_valid"])
    force_age_ms = (
        max(0.0, now_s - float(snapshot["force_updated_time_s"])) * 1000.0
        if force_valid
        else -1.0
    )
    return {
        "observation.tactile.left_values": np.asarray(snapshot["values"], dtype=np.uint16),
        "observation.tactile.left_valid": valid,
        "observation.tactile.left_age_ms": age_ms.astype(np.float32),
        "observation.tactile.left_update_sequence": np.asarray(
            snapshot["update_sequence"], dtype=np.int64
        ),
        "observation.tactile.left_force_act_g": np.asarray(
            snapshot["force_act_g"], dtype=np.int16
        ),
        "observation.tactile.left_force_valid": np.array([force_valid], dtype=np.bool_),
        "observation.tactile.left_force_age_ms": np.array([force_age_ms], dtype=np.float32),
        "observation.tactile.modbus_metrics": np.asarray(snapshot["metrics"], dtype=np.float32),
    }


def pack_snapshot(snapshot: dict) -> bytes:
    mnp.patch()
    return TACTILE_TOPIC + b" " + msgpack.packb(snapshot, default=mnp.encode, use_bin_type=True)


def unpack_snapshot(message: bytes) -> dict:
    prefix = TACTILE_TOPIC + b" "
    if not message.startswith(prefix):
        raise ValueError("not an Inspire tactile message")
    mnp.patch()
    snapshot = msgpack.unpackb(message[len(prefix) :], object_hook=mnp.decode, raw=False)
    if int(snapshot.get("version", -1)) != TACTILE_PROTOCOL_VERSION:
        raise ValueError(f"unsupported tactile protocol version: {snapshot.get('version')}")
    return snapshot
