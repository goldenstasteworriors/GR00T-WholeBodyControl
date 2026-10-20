"""Measurements on the PC clock; never subtract PICO/robot clocks from it."""

import numpy as np


def pipeline_metrics(trace, *, received_ns, goal_read_ns, send_started_ns, send_finished_ns):
    metrics = {}
    def span(name, end, start):
        if end is not None and start is not None and end >= start:
            metrics[name] = (end - start) / 1e6
    span("pico_sdk_read_ms", trace.get("pico_sdk_read_finished_ns"), trace.get("pico_sdk_read_started_ns"))
    span("teleop_total_ms", trace.get("teleop_action_finished_ns"), trace.get("teleop_loop_started_ns"))
    span("ros_delivery_ms", received_ns, trace.get("teleop_publish_ns"))
    span("control_goal_wait_ms", goal_read_ns, received_ns)
    span("dds_write_call_ms", send_finished_ns, send_started_ns)
    span("pc_sdk_read_to_dds_ms", send_finished_ns, trace.get("pico_sdk_read_finished_ns"))
    span("pc_frame_first_seen_to_dds_ms", send_finished_ns, trace.get("pico_frame_first_seen_ns"))
    for name in ("pico_source_interval_ms", "pico_observed_interval_ms",
                 "sdk_producer_source_interval_ms", "sdk_producer_observed_interval_ms"):
        if trace.get(name) is not None and trace[name] >= 0:
            metrics[name] = trace[name]
    return metrics


def summarize(values):
    array = np.asarray(values, dtype=float)
    return {"count": len(array), "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)), "max": float(array.max())}
