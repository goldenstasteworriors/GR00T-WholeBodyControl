"""Manual collection transitions: damped -> initializing -> ready -> running."""

import queue

from decoupled_wbc.control.policy.wbc_policy_factory import get_startup_upper_body_policy


class CollectionMode:
    def __init__(self, env, policy, robot_model, config, shutdown):
        self.env = env
        self.policy = policy
        self.robot_model = robot_model
        self.config = config
        self.shutdown = shutdown
        self.state = "initializing"
        self.initialization_started = None
        self.initialization_epoch = 0
        self.last_start_event = 0
        self.keyboard_events = queue.SimpleQueue()

    def handle_keyboard_button(self, key):
        if key == "]":
            self.keyboard_events.put("advance")
            return True
        if key == "o":
            self.shutdown.request_estop("keyboard:o")
            return True
        # Navigation/height keys only apply after a separate standing request.
        return self.state != "running" and key in {"w", "s", "a", "q", "e", "z", "1", "2", "n", "m"}

    def enter_damping(self, reason):
        self.policy.lower_body_policy.use_policy_action = False
        self.env.enter_damping()
        self.state = "damped"
        print(f"[CollectionMode] damped: {reason}; program remains running", flush=True)

    def advance(self, now):
        with self.shutdown.transition_lock:
            if self.shutdown.estop_requested.is_set():
                return
            self._advance(now)

    def _advance(self, now):
        if self.state == "damped":
            lower = self.policy.lower_body_policy
            lower.use_policy_action = False
            lower._waist_initial_time = None
            lower.cmd = lower.config["cmd_init"].copy()
            lower.height_cmd = lower.config["height_cmd"]
            lower.freq_cmd = lower.config["freq_cmd"]
            lower.roll_cmd, lower.pitch_cmd, lower.yaw_cmd = lower.config["rpy_cmd"]
            lower.action[:] = 0.0
            lower.obs_history.clear()
            self.policy.upper_body_policy = get_startup_upper_body_policy(
                self.robot_model, self.config, now
            )
            self.env.restart_initialization()
            self.initialization_started = now
            self.initialization_epoch += 1
            self.state = "initializing"
            print("[CollectionMode] initializing; press again only after ready", flush=True)
        elif self.state == "ready":
            self.state = "running"
            self.policy.lower_body_policy.cmd[:] = 0.0
            self.policy.lower_body_policy.use_policy_action = True
            print("[CollectionMode] running: standing policy enabled", flush=True)
        elif self.state == "running":
            self.enter_damping("start/stop combination")
        else:
            print("[CollectionMode] still initializing; request consumed without enabling standing", flush=True)

    def update(self, goal, now):
        if self.initialization_started is None:
            self.initialization_started = now
        event_id = int(goal.get("start_stop_event_id", 0))
        fresh = event_id > self.last_start_event
        if fresh:
            self.last_start_event = event_id
        keyboard = []
        while not self.keyboard_events.empty():
            keyboard.append(self.keyboard_events.get())
        if self.shutdown.estop_requested.is_set() or goal.get("emergency_stop", False):
            reason = self.shutdown.estop_reason or "teleop_emergency_stop"
            self.shutdown.estop_requested.clear()
            self.enter_damping(reason)
            return
        # Only distinct rising-edge IDs can progress the two-step restart.
        if fresh or keyboard or goal.get("toggle_policy_action", False):
            self.advance(now)
        if self.state == "initializing":
            if self.env.safety_monitor.startup_complete and now - self.initialization_started >= 2.0:
                self.state = "ready"
                print("[CollectionMode] ready; a new combination enables standing", flush=True)
        self.policy.lower_body_policy.use_policy_action = self.state == "running"

    def status(self):
        return {
            "control_mode": self.state,
            "initialization_epoch": self.initialization_epoch,
            "start_stop_event_id": self.last_start_event,
        }
