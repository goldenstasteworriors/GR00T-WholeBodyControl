"""Three-Enter operator lifecycle around the original SONICMJ control loop.

The keyboard/damping path never waits for model HTTP. Recorded legs are never
commanded. Initialization uses both arms and native Dex1 achieved positions.
"""
import json
import queue
from pathlib import Path
import subprocess
import sys
import threading
import time
import numpy as np


class OperatorState:
    def __init__(self):
        self.phase, self.epoch, self.recovering = "idle", 0, False

    def key(self, key):
        before = self.phase
        if key == "d":
            if self.phase != "damped":
                self.epoch += 1
            self.phase = "damped"
        elif key == "r" and self.phase in ("damped", "fault", "paused"):
            self.phase, self.recovering = "standing", True
            self.epoch += 1
        elif key == "enter":
            self.phase = {"idle": "standing", "standing_ready": "arm_initializing",
                          "ready": "running"}.get(self.phase, self.phase)
            if self.phase != before:
                self.epoch += 1
        return before != self.phase

    def tick(self, *, stand_ready=False, pose_ready=False):
        if self.phase == "standing" and stand_ready:
            self.phase = "arm_initializing" if self.recovering else "standing_ready"
        elif self.phase == "arm_initializing" and pose_ready:
            self.phase, self.recovering = "ready", False


def validate_pose(robot, pose):
    body = np.asarray(robot.get_body_actuated_joint_indices())
    arms = pose[:14]
    lo, hi = np.asarray(robot.lower_joint_limits)[body[15:29]], np.asarray(robot.upper_joint_limits)[body[15:29]]
    if np.any(arms < lo) or np.any(arms > hi) or np.any(pose[14:16] < 0) or np.any(pose[14:16] > 5.4):
        raise ValueError("Selected achieved pose exceeds the robot's joint limits; select another frame")


def install_operator(loop, *, catalog, selection_path, task, sonic_root, policy_endpoint,
                     log_dir, execute, chunks, policy_ready, on_close=lambda: None,
                     camera_host="127.0.0.1", camera_port=5555, gravity_options=None):
    """Install only in this launcher process; leave SONICMJ source untouched."""
    from wbc_bridge import install_jar_adapter
    install_jar_adapter(loop, gravity_options=gravity_options)
    base_env, base_shutdown = loop.G1Env, loop.DampingShutdown
    holder = {}
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    class SessionEnv(base_env):
        def __init__(self, *args, **kwargs):
            # Do not release the vendor's standing mode while merely opening
            # the program. The first Enter owns the explicit takeover.
            kwargs["config"] = dict(kwargs["config"], diagnostic_read_only=True)
            super().__init__(*args, **kwargs)
            self.operator_phase = "idle"
            sender = self.body().body_command_sender
            self.arm_motors = np.asarray(sender.config["JOINT2MOTOR"])[15:29]
            self.arm_kp = sender.robot_kp[self.arm_motors].copy()
            self.arm_kd = sender.robot_kd[self.arm_motors].copy()

        def _queue_action(self, action):
            if self.operator_phase in ("idle", "taking_control"):
                return
            arms_enabled = self.operator_phase not in ("standing", "standing_ready")
            sender = self.body().body_command_sender
            sender.robot_kp[self.arm_motors] = self.arm_kp if arms_enabled else 0.
            sender.robot_kd[self.arm_motors] = self.arm_kd if arms_enabled else 0.
            super()._queue_action({**action, "jar_control_hands": arms_enabled, "jar_control_arms": arms_enabled})

    class SessionMode:
        def __init__(self, env, policy, robot_model, config, shutdown):
            self.env, self.policy, self.robot, self.shutdown = env, policy, robot_model, shutdown
            self.machine = OperatorState()
            self.events = queue.SimpleQueue()
            self.selection = catalog.load_selection(selection_path, default_task=task)
            self.target = catalog.pose(self.selection)
            validate_pose(self.robot, self.target)
            self.client, self.retiring = None, []
            self.menu_input = None
            self.pose_start = self.pose_origin = self.settled_since = None
            self.stand_started = None
            self.takeover_thread = None
            self.takeover_result = queue.SimpleQueue()
            self.takeover_cancel = threading.Event()
            self.owns_control = False
            self.sequence = 0
            self.last_notice = None
            holder["mode"] = self
            # Feed measured arms to the lower-body policy while arms are free.
            buffer = policy.jar_buffer
            class HoldUpper:
                def get_action(self, time=None):
                    return {"target_upper_body_pose": buffer.observation[buffer.upper], "navigate_cmd": np.zeros(3)}
            self.hold_upper = HoldUpper()
            policy.upper_body_policy = self.hold_upper
            self.notice()

        @property
        def state(self):
            return self.machine.phase

        def notice(self):
            text = (self.state, json.dumps(self.selection), self.menu_input is not None)
            if text != self.last_notice:
                print(f"[JAROperator] {self.state}; initial={self.selection}; Enter=advance, p=pose, d=damping, r=recover", flush=True)
                self.last_notice = text

        def handle_keyboard_button(self, key):
            # The original KeyboardEStop handles d first and sends damping
            # immediately, independent of the 50 Hz loop and HTTP inference.
            if key == "d":
                self.shutdown.request_estop("keyboard:d")
            else:
                self.events.put(key)
            return True  # never pass navigation/legacy restart keys downstream

        def stop_client(self):
            if self.client is not None:
                if self.client.poll() is None:
                    self.client.terminate()
                    self.retiring.append((self.client, time.monotonic()))
                self.client = None
            self.policy.jar_buffer.stop()

        def enter_damping(self, reason):
            self.takeover_cancel.set()
            while not self.events.empty():
                self.events.get()
            self.stop_client()
            self.machine.key("d")
            self.menu_input = None
            self.policy.lower_body_policy.use_policy_action = False
            self.env.enter_damping()
            self.env.operator_phase = "damped"
            self.notice()

        def takeover(self, epoch):
            """Bounded vendor RPC on its own thread, with cancellation checks."""
            try:
                if execute and not self.owns_control:
                    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
                    client = MotionSwitcherClient()
                    client.SetTimeout(1.)
                    client.Init()
                    for _ in range(10):
                        if self.takeover_cancel.is_set():
                            return
                        status, result = client.CheckMode()
                        if status != 0:
                            raise RuntimeError(f"CheckMode failed: {status}")
                        if not result.get("name"):
                            self.owns_control = True
                            break
                        if self.takeover_cancel.is_set():
                            return
                        status = client.ReleaseMode()
                        if self.takeover_cancel.wait(.1):
                            return
                    else:
                        raise RuntimeError("Vendor mode did not release within the bounded takeover")
                self.takeover_result.put((epoch, None))
            except Exception as error:
                self.takeover_result.put((epoch, str(error)))
            finally:
                if self.takeover_cancel.is_set():
                    self.env.send_damping()  # also cover an RPC that returned after d

        def begin_stand(self, now):
            self.stop_client()
            self.policy.upper_body_policy = self.hold_upper
            lower = self.policy.lower_body_policy
            lower.use_policy_action = False
            lower._waist_initial_time = None
            lower.action[:] = 0.
            lower.obs_history.clear()
            lower.cmd = lower.config["cmd_init"].copy()
            lower.cmd[:] = 0.
            lower.height_cmd = lower.config["height_cmd"]
            lower.freq_cmd = lower.config["freq_cmd"]
            lower.roll_cmd, lower.pitch_cmd, lower.yaw_cmd = lower.config["rpy_cmd"]
            self.stand_started = None
            self.env.config["diagnostic_read_only"] = not execute
            self.env.operator_phase = "taking_control"
            self.takeover_cancel.clear()
            self.takeover_thread = threading.Thread(target=self.takeover, args=(self.machine.epoch,), daemon=True)
            self.takeover_thread.start()

        def begin_pose(self, now):
            self.policy.jar_buffer.stop()
            self.pose_origin = self.policy.jar_buffer.state[np.arange(15,31)].copy()
            self.pose_start, self.settled_since = now, None
            speeds = np.full(16, .5)
            speeds[14:16] = .8
            self.pose_duration = max(1., float(np.max(np.abs(self.target-self.pose_origin)/speeds)))
            self.sequence = 0

        def pose_tick(self, now):
            fraction = min(1., max(0., (now-self.pose_start)/self.pose_duration))
            target = self.pose_origin + fraction * (self.target-self.pose_origin)
            self.policy.set_goal(dict(jar_target=target, jar_binding_id=f"init:{self.machine.epoch}:{self.pose_start}",
                jar_sequence=self.sequence, jar_sent_monotonic=time.monotonic(), jar_task=task, jar_reset=True))
            self.sequence += 1
            if self.policy.jar_buffer.last_error:
                raise ValueError(self.policy.jar_buffer.last_error)
            measured = self.policy.jar_buffer.state[np.arange(15,31)]
            tolerance = np.full(16, .04)
            tolerance[14:16] = .1
            arrived = fraction == 1. and np.all(np.abs(measured-self.target) <= tolerance)
            self.settled_since = (now if self.settled_since is None else self.settled_since) if arrived else None
            if self.state == "arm_initializing" and now-self.pose_start > self.pose_duration+20:
                raise RuntimeError("Initial pose did not converge; use r to retry after checking the robot")
            return self.settled_since is not None and now-self.settled_since >= .3

        def start_client(self):
            if self.retiring:
                raise ValueError("Previous inference client is still stopping; press Enter again after it exits")
            if not policy_ready():
                raise ValueError("Policy service is not ready; initialization remains active")
            command = [sys.executable, str(Path(__file__).with_name("operator_client.py")),
                "--sonic-root", str(sonic_root), "--task", task, "--policy-endpoint", policy_endpoint,
                "--camera-endpoint", f"tcp://{camera_host}:{camera_port}",
                "--operator-epoch", str(self.machine.epoch), "--chunks", str(chunks)]
            if execute:
                command.append("--execute")
            with (log_dir/f"inference_{self.machine.epoch}.log").open("w") as log:
                self.client = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)

        def choose(self):
            selection = catalog.parse(self.menu_input)
            target = catalog.pose(selection)
            validate_pose(self.robot, target)
            catalog.save_selection(selection_path, selection)
            self.selection, self.target = selection, target
            self.menu_input = None
            if self.state == "ready":
                self.machine.epoch += 1
                self.machine.phase = "arm_initializing"
                self.begin_pose(time.monotonic())

        def key(self, key, now):
            if key == "p" and self.state in ("idle", "standing_ready", "ready", "damped"):
                self.menu_input = ""
                print(catalog.menu(), flush=True)
                return
            if self.menu_input is not None:
                if key == "enter":
                    self.choose()
                elif key in ("esc", "escape"):
                    self.menu_input = None
                elif key == "backspace":
                    self.menu_input = self.menu_input[:-1]
                elif key == "space" or key.isdecimal():
                    self.menu_input += " " if key == "space" else key
                if self.menu_input is not None:
                    print(f"\r[Pose selection] {self.menu_input}   ", end="", flush=True)
                return
            if key == "r" and self.takeover_thread is not None and self.takeover_thread.is_alive():
                raise ValueError("Takeover RPC is still exiting; wait before r")
            before = self.state
            if self.machine.key(key):
                if self.state == "standing":
                    self.begin_stand(now)
                elif self.state == "arm_initializing":
                    self.begin_pose(now)
                elif self.state == "running":
                    try:
                        self.start_client()
                        self.policy.jar_buffer.stop()
                    except Exception:
                        self.machine.phase = before
                        raise

        def update(self, goal, now):
            link = self.env.body().body_command_sender.onboard_link
            if link is not None and link.disconnected:
                self.enter_damping("connection lost")
                goal.clear()
                return
            if self.shutdown.estop_requested.is_set() or goal.get("emergency_stop"):
                self.enter_damping("operator emergency stop")
                self.shutdown.estop_requested.clear()
                goal.clear()
                return
            for child, started in list(self.retiring):
                if child.poll() is not None:
                    self.retiring.remove((child, started))
                elif now-started > 5:
                    child.kill()
            while not self.events.empty():
                try:
                    self.key(self.events.get(), now)
                except (ValueError, KeyError, IndexError, OSError) as error:
                    print(f"[JAROperator] {error}", flush=True)
            if not self.takeover_result.empty():
                epoch, error = self.takeover_result.get()
                if epoch == self.machine.epoch and self.state == "standing":
                    if error:
                        self.enter_damping(error)
                    else:
                        self.env.restart_initialization()
                        self.stand_started = now
            if self.state == "standing" and self.stand_started is not None:
                self.machine.tick(stand_ready=self.env.safety_monitor.startup_complete and now-self.stand_started >= 2.)
                if self.state == "arm_initializing":
                    self.begin_pose(now)
            try:
                if self.state in ("arm_initializing", "ready"):
                    ready = self.pose_tick(now)
                    self.machine.tick(pose_ready=ready)
            except (ValueError, RuntimeError) as error:
                self.policy.jar_buffer.stop()
                self.machine.phase = "fault"
                print(f"[JAROperator] {error}", flush=True)
            if self.client is not None and self.client.poll() is not None:
                result = self.client.returncode
                self.client = None
                self.policy.jar_buffer.stop()
                self.machine.phase = "paused" if result == 0 else "fault"
                print(f"[JAROperator] inference exited {result}; r restores the selected pose", flush=True)
            standing = self.state not in ("idle", "damped") and self.stand_started is not None
            self.policy.lower_body_policy.use_policy_action = standing
            self.policy.lower_body_policy.cmd[:] = 0.
            self.env.operator_phase = self.state if standing or self.state in ("idle", "damped") else "taking_control"
            # An old client must never re-arm after d/r, even if its delayed
            # packet arrives after a new inference epoch has started.
            if self.state != "running" or goal.get("jar_operator_epoch") != self.machine.epoch:
                goal.clear()
            self.notice()

        def status(self):
            return dict(control_mode=self.state, jar_operator_epoch=self.machine.epoch,
                        selected_initial_pose=self.selection, initialization_epoch=self.machine.epoch,
                        connection_latched=False, initialization_only=False)

        def close(self):
            self.takeover_cancel.set()
            self.stop_client()
            for child, _ in self.retiring:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=2)

    class SessionShutdown(base_shutdown):
        def finish(self):
            self.env.enter_damping()
            if "mode" in holder:
                holder["mode"].close()
            try:
                return super().finish()
            finally:
                on_close()

    loop.G1Env, loop.CollectionMode, loop.DampingShutdown = SessionEnv, SessionMode, SessionShutdown
    return holder
