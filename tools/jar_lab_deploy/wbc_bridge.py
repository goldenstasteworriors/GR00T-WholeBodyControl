"""Arm-only adapter for the SONICMJ wired-PICO decoupled controller.

No DDS or ROS imports here. Leg/waist commands remain in the existing WBC.
The sender and controller must run on the same host (monotonic timestamps).
"""
import time
import numpy as np

from contract import JOINT_NAMES, TASKS
ARM_INDICES = list(range(15,31))


class ArmTargetBuffer:
    def __init__(self, robot_model, *, max_arm_speed=.5, command_timeout=.25):
        self.robot = robot_model
        self.body = np.asarray(robot_model.get_body_actuated_joint_indices(), dtype=int)
        if len(self.body) != 29:
            raise ValueError("Expected 29 G1 body actuators")
        if [robot_model.joint_names[i] for i in self.body[15:29]] != JOINT_NAMES[:14]:
            raise ValueError("Robot arm order differs from the JAR observation contract")
        self.arms = self.body[15:29]
        self.upper = np.asarray(robot_model.get_joint_group_indices("upper_body"), dtype=int)
        if set(self.upper) & set(self.body[:15]):
            raise ValueError("Configure waist_location=lower_body / enable_waist=False")
        self.speed, self.timeout = float(max_arm_speed), float(command_timeout)
        if not 0 < self.speed <= 2 or not 0 < self.timeout <= 1:
            raise ValueError("Invalid controller limits")
        self.binding = None
        self.sequence = -1
        self.observation = None
        self.last_time = None
        self.target = None
        self.native = None
        self.last_error = None
        self.lower_limits = np.asarray(robot_model.lower_joint_limits)[self.arms]
        self.upper_limits = np.asarray(robot_model.upper_joint_limits)[self.arms]

    def observe(self, observation):
        q = np.asarray(observation["q"], dtype=float)
        left = np.asarray(observation["left_hand_dex1_q"], dtype=float).reshape(-1)
        right = np.asarray(observation["right_hand_dex1_q"], dtype=float).reshape(-1)
        if q.shape != (self.robot.num_dofs,) or left.shape != (1,) or right.shape != (1,):
            raise ValueError("Expected full model state and native one-axis Dex1 feedback")
        state = np.r_[q[self.body], left, right]
        if not np.isfinite(state).all():
            raise ValueError("Invalid measured state")
        self.observation = q.copy()
        self.state = state
        if self.target is None:
            self.previous = state[ARM_INDICES].copy()

    def stop(self):
        self.binding, self.target = None, None
        self.sequence = -1

    def receive(self, goal, now=None):
        now = time.monotonic() if now is None else now
        if self.observation is None:
            raise RuntimeError("Measured state required before a command")
        if goal.get("jar_stop"):
            if goal.get("jar_binding_id") == self.binding:
                self.stop()
            return
        if "jar_target" not in goal:
            return
        binding = goal["jar_binding_id"]
        sequence = goal["jar_sequence"]
        sent = float(goal["jar_sent_monotonic"])
        if not isinstance(binding, str) or not binding or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("Invalid binding/sequence")
        if not np.isfinite(sent) or not 0 <= now - sent <= self.timeout:
            raise ValueError("Stale command; sender must share the controller clock")
        if binding == self.binding and sequence <= self.sequence:
            raise ValueError("Repeated or reordered command")
        if goal["jar_task"] not in TASKS:
            raise ValueError("Unknown JAR operation")
        active = np.ones(16, dtype=bool)
        target = np.asarray(goal["jar_target"], dtype=float)
        if target.shape != (16,) or not np.isfinite(target).all():
            raise ValueError("Expected finite 16D manipulation command")
        arm_targets = target[:14]
        arm_active = active[:14]
        if np.any(((arm_targets < self.lower_limits) | (arm_targets > self.upper_limits)) & arm_active):
            raise ValueError("Arm target outside model joint limits")
        if np.any(((target[14:16] < 0) | (target[14:16] > 5.4)) & active[14:16]):
            raise ValueError("Dex1 native angle outside [0,5.4] radians")
        if binding != self.binding:
            if not goal.get("jar_reset", False):
                raise ValueError("A new target binding requires jar_reset")
            self.previous = self.state[ARM_INDICES].copy()
            self.last_time = now
        self.binding, self.sequence = binding, sequence
        self.target, self.active, self.sent = target.copy(), active, sent
        self.last_error = None

    def sample(self, now=None):
        now = time.monotonic() if now is None else now
        if self.observation is None:
            raise RuntimeError("No measured state")
        measured = self.state[ARM_INDICES]
        dt = 0 if self.last_time is None else np.clip(now - self.last_time, 0, .05)
        self.last_time = now
        if self.target is None or now - self.sent > self.timeout:
            # Clear target and binding, requiring explicit re-arming after expiry.
            self.stop()
            command = measured.copy()
        else:
            command = np.where(self.active, self.target, measured)
            arms = np.arange(14)
            command[arms] = np.clip(command[arms], self.previous[arms] - self.speed * dt,
                                    self.previous[arms] + self.speed * dt)
            command[14:16] = np.clip(command[14:16], measured[14:16] - .18,
                                      measured[14:16] + .18)
            command = np.where(self.active, command, measured)
        self.previous = command.copy()
        q = self.observation.copy()
        q[self.arms] = command[:14]
        self.native = command[14:16].copy()
        return {"target_upper_body_pose": q[self.upper], "navigate_cmd": np.zeros(3)}


def install_jar_adapter(loop_module, *, gravity_options=None):
    """Select JAR adapter in a dedicated control process, not the source tree."""
    original_factory, original_env = loop_module.get_wbc_policy, loop_module.G1Env
    holder = {}

    class JAREnvironment(original_env):
        def __init__(self, *args, **kwargs):
            from gravity import build_arm_gravity
            # Validate dynamics before initializing any robot transport.
            self.jar_gravity = build_arm_gravity(kwargs["robot_model"], kwargs["config"], gravity_options)
            super().__init__(*args, **kwargs)

        def enter_damping(self):
            with self._action_lock:
                if self.jar_gravity is not None:
                    self.jar_gravity.reset()
                return super().enter_damping()

        def restart_initialization(self):
            with self._action_lock:
                if self.jar_gravity is not None:
                    self.jar_gravity.reset()
                return super().restart_initialization()

        def observe(self):
            observation = super().observe()
            buffer = holder.get("buffer")
            observation["jar_binding_id"] = None if buffer is None else buffer.binding
            observation["jar_last_sequence"] = -1 if buffer is None else buffer.sequence
            observation["jar_error"] = None if buffer is None else buffer.last_error
            return observation

        def _queue_action(self, action):
            if self.last_obs is not None:
                action = self.safety_monitor.handle_violations(self.last_obs, action)["action"]
            if self.config.get("diagnostic_read_only", False):
                return
            body_q = self.robot_model.get_body_actuated_joints(action["q"])
            body_tau = np.zeros_like(body_q)
            if self.jar_gravity is not None:
                body_tau = self.jar_gravity.sample(self.last_obs, enabled=action.get("jar_control_arms", True))
            self.body().queue_action({"body_q": body_q, "body_dq": np.zeros_like(body_q),
                                      "body_tau": body_tau})
            link = self.body().body_command_sender.onboard_link
            if link is not None and link.disconnected:
                return
            if not action.get("jar_control_hands", True):
                return
            # This process owns the sole Dex1 writer. Never convert to binary grasp modes.
            targets = action.get("jar_dex1_targets")
            if targets is None:
                targets = [float(self.last_obs[f"{side}_hand_dex1_q"][0]) for side in ("left", "right")]
            for side, value in zip(("left", "right"), targets):
                getattr(self.hands(), side).command_position(float(value))

    def factory(robot_type, robot_model, wbc_config, *args, **kwargs):
        if wbc_config.get("HAND_TYPE") != "dex1" or not wbc_config.get("with_hands") or wbc_config.get("enable_waist"):
            raise ValueError("Lab deployment requires Dex1, with_hands=True, enable_waist=False")
        policy = original_factory(robot_type, robot_model, wbc_config, *args, **kwargs)
        buffer = ArmTargetBuffer(robot_model, max_arm_speed=wbc_config["upper_body_max_joint_speed"])
        holder["buffer"] = buffer
        original_observe, original_action = policy.set_observation, policy.get_action

        class Upper:
            def set_goal(self, goal):
                buffer.receive(goal)

            def get_action(self, time=None):
                return buffer.sample(time)

        jar_upper = Upper()

        def observe(observation):
            buffer.observe(observation)
            original_observe(observation)

        def goal(message):
            try:
                if "target_upper_body_pose" in message:
                    raise ValueError("Do not run the PICO arm publisher alongside JAR")
                buffer.receive(message)
            except (ValueError, KeyError, TypeError, RuntimeError) as error:
                # An invalid upper-body command must not take down balance.
                buffer.stop()
                buffer.last_error = str(error)
                policy.upper_body_policy = jar_upper
                return
            if "jar_target" in message:
                policy.upper_body_policy = jar_upper
            policy.lower_body_policy.set_use_teleop_policy_cmd(True)
            policy.lower_body_policy.set_goal({k: message[k] for k in ("set_policy_action", "toggle_policy_action") if k in message})

        def action(time=None):
            if policy.upper_body_policy is not jar_upper:
                # CollectionMode restores its own startup policy after damping.
                buffer.stop()
            result = original_action(time)
            result["jar_dex1_targets"] = (buffer.native.copy() if policy.upper_body_policy is jar_upper
                                           else buffer.state[[29, 30]].copy())
            return result

        policy.set_observation, policy.set_goal, policy.get_action = observe, goal, action
        policy.jar_buffer, policy.jar_upper_policy = buffer, jar_upper
        return policy

    loop_module.get_wbc_policy, loop_module.G1Env = factory, JAREnvironment
