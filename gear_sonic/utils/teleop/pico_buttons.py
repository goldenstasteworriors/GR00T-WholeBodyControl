"""Shared PICO button combinations used by official and decoupled teleoperation."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class PicoButtonState:
    """One coherent snapshot of the buttons used by the official manager."""

    a: bool = False
    b: bool = False
    x: bool = False
    y: bool = False
    left_grip: float = 0.0
    left_axis_click: bool = False


@dataclass(frozen=True)
class PicoButtonEvents:
    """Rising-edge events produced from one PICO button snapshot."""

    state: PicoButtonState
    ax_pressed: bool = False
    by_pressed: bool = False
    start_stop_pressed: bool = False
    left_axis_click_pressed: bool = False
    toggle_data_collection: bool = False
    toggle_data_abort: bool = False

    @property
    def any_pressed(self) -> bool:
        return any(
            (
                self.ax_pressed,
                self.by_pressed,
                self.start_stop_pressed,
                self.left_axis_click_pressed,
                self.toggle_data_collection,
                self.toggle_data_abort,
            )
        )


@dataclass(frozen=True)
class PicoButtonDiagnostics:
    """Read-only counters for locating failures without changing button behavior."""

    state: PicoButtonState
    samples: int
    read_errors: int
    last_error: str
    ax_latched: int
    by_latched: int
    start_stop_latched: int
    data_collection_latched: int
    data_abort_latched: int


class PicoButtonEdgeDetector:
    """Apply the official manager's combinations and rising-edge semantics."""

    def __init__(self) -> None:
        self._previous = PicoButtonState()

    def update(self, state: PicoButtonState) -> PicoButtonEvents:
        previous = self._previous

        ax_pressed = state.a and state.x
        previous_ax_pressed = previous.a and previous.x
        by_pressed = state.b and state.y
        previous_by_pressed = previous.b and previous.y
        start_stop_pressed = state.a and state.b and state.x and state.y
        previous_start_stop_pressed = previous.a and previous.b and previous.x and previous.y
        toggle_data_collection = state.a and state.left_grip > 0.5
        previous_toggle_data_collection = previous.a and previous.left_grip > 0.5
        toggle_data_abort = state.b and state.left_grip > 0.5
        previous_toggle_data_abort = previous.b and previous.left_grip > 0.5

        events = PicoButtonEvents(
            state=state,
            ax_pressed=ax_pressed and not previous_ax_pressed,
            by_pressed=by_pressed and not previous_by_pressed,
            start_stop_pressed=start_stop_pressed and not previous_start_stop_pressed,
            left_axis_click_pressed=state.left_axis_click and not previous.left_axis_click,
            toggle_data_collection=(
                toggle_data_collection and not previous_toggle_data_collection
            ),
            toggle_data_abort=toggle_data_abort and not previous_toggle_data_abort,
        )
        self._previous = state
        return events


class PicoButtonEventSampler:
    """Poll one initialized XRT client quickly and latch official button edges."""

    def __init__(
        self,
        read_state: Callable[[], PicoButtonState],
        poll_hz: float = 200.0,
        max_pending_events: int = 64,
    ) -> None:
        if poll_hz <= 0:
            raise ValueError("poll_hz must be positive")
        if max_pending_events <= 0:
            raise ValueError("max_pending_events must be positive")

        self._read_state = read_state
        self._period = 1.0 / poll_hz
        self._detector = PicoButtonEdgeDetector()
        self._events: deque[PicoButtonEvents] = deque(maxlen=max_pending_events)
        self._latest_state = PicoButtonState()
        self._samples = 0
        self._read_errors = 0
        self._last_error = ""
        self._ax_latched = 0
        self._by_latched = 0
        self._start_stop_latched = 0
        self._data_collection_latched = 0
        self._data_abort_latched = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="pico-official-button-sampler",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def latest_state(self) -> PicoButtonState:
        with self._lock:
            return self._latest_state

    def consume_events(self) -> tuple[PicoButtonEvents, ...]:
        with self._lock:
            events = tuple(self._events)
            self._events.clear()
        return events

    def clear_events(self) -> None:
        with self._lock:
            self._events.clear()

    def diagnostics(self) -> PicoButtonDiagnostics:
        with self._lock:
            return PicoButtonDiagnostics(
                state=self._latest_state,
                samples=self._samples,
                read_errors=self._read_errors,
                last_error=self._last_error,
                ax_latched=self._ax_latched,
                by_latched=self._by_latched,
                start_stop_latched=self._start_stop_latched,
                data_collection_latched=self._data_collection_latched,
                data_abort_latched=self._data_abort_latched,
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                state = self._read_state()
            except Exception as exc:
                with self._lock:
                    self._read_errors += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
                self._stop.wait(0.05)
                continue

            events = self._detector.update(state)
            with self._lock:
                self._latest_state = state
                self._samples += 1
                if events.any_pressed:
                    self._events.append(events)
                self._ax_latched += int(events.ax_pressed)
                self._by_latched += int(events.by_pressed)
                self._start_stop_latched += int(events.start_stop_pressed)
                self._data_collection_latched += int(events.toggle_data_collection)
                self._data_abort_latched += int(events.toggle_data_abort)
            self._stop.wait(self._period)
