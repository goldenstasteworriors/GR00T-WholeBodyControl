from copy import deepcopy
import time

import tyro

from decoupled_wbc.control.envs.g1.g1_env import G1Env
from decoupled_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    DEFAULT_BASE_HEIGHT,
    DEFAULT_NAV_CMD,
    DEFAULT_WRIST_POSE,
    JOINT_SAFETY_STATUS_TOPIC,
    LOWER_BODY_POLICY_STATUS_TOPIC,
    ROBOT_CONFIG_TOPIC,
    SIM_RESET_TOPIC,
    STATE_TOPIC_NAME,
)
from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
from decoupled_wbc.control.policy.unitree_loco_keyboard_controller import (
    UnitreeLocoKeyboardController,
)
from decoupled_wbc.control.robot_model.instantiation.g1 import (
    instantiate_g1_robot_model,
)
from decoupled_wbc.control.utils.keyboard_dispatcher import (
    KeyboardDispatcher,
    KeyboardEStop,
    KeyboardListenerPublisher,
    ROSKeyboardDispatcher,
)
from decoupled_wbc.control.utils.ros_utils import (
    ROSManager,
    ROSMsgPublisher,
    ROSMsgSubscriber,
    ROSServiceServer,
)
from decoupled_wbc.control.utils.telemetry import Telemetry

CONTROL_NODE_NAME = "ControlPolicy"


class SimResetKeyboardPublisher:
    def __init__(self, publisher: ROSMsgPublisher):
        self.publisher = publisher

    def handle_keyboard_button(self, key):
        if key in {"r", "backspace"}:
            print(f"Requesting MuJoCo reset from keyboard: {key}")
            self.publisher.publish({"reason": f"keyboard:{key}", "timestamp": time.time()})


def main(config: ControlLoopConfig):
    ros_manager = ROSManager(node_name=CONTROL_NODE_NAME)
    node = ros_manager.node

    # start the robot config server
    ROSServiceServer(ROBOT_CONFIG_TOPIC, config.to_dict())

    wbc_config = config.load_wbc_yaml()

    data_exp_pub = ROSMsgPublisher(STATE_TOPIC_NAME)
    lower_body_policy_status_pub = ROSMsgPublisher(LOWER_BODY_POLICY_STATUS_TOPIC)
    joint_safety_status_pub = ROSMsgPublisher(JOINT_SAFETY_STATUS_TOPIC)
    sim_reset_pub = ROSMsgPublisher(SIM_RESET_TOPIC)

    # Initialize telemetry
    telemetry = Telemetry(window_size=100)

    waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
    robot_model = instantiate_g1_robot_model(
        waist_location=waist_location,
        high_elbow_pose=config.high_elbow_pose,
        with_hands=config.with_hands,
    )

    env = G1Env(
        env_name=config.env_name,
        robot_model=robot_model,
        config=wbc_config,
        wbc_version=config.wbc_version,
    )
    if env.sim and not config.sim_sync_mode:
        env.start_simulator()

    wbc_policy = get_wbc_policy("g1", robot_model, wbc_config)

    keyboard_listener_pub = KeyboardListenerPublisher()
    keyboard_estop = KeyboardEStop()
    sim_reset_keyboard_pub = SimResetKeyboardPublisher(sim_reset_pub)
    keyboard_loco_controller = None
    if config.keyboard_lower_body_control:
        keyboard_loco_controller = UnitreeLocoKeyboardController(
            max_linear_velocity=config.unitree_loco_max_linear_velocity,
            max_angular_velocity=config.unitree_loco_max_angular_velocity,
        )
    if config.keyboard_dispatcher_type == "raw":
        dispatcher = KeyboardDispatcher()
    elif config.keyboard_dispatcher_type == "ros":
        dispatcher = ROSKeyboardDispatcher()
    else:
        raise ValueError(
            f"Invalid keyboard dispatcher: {config.keyboard_dispatcher_type}, please use 'raw' or 'ros'"
        )
    dispatcher.register(env)
    dispatcher.register(wbc_policy)
    dispatcher.register(keyboard_listener_pub)
    dispatcher.register(sim_reset_keyboard_pub)
    if keyboard_loco_controller is not None:
        dispatcher.register(keyboard_loco_controller)
    dispatcher.register(keyboard_estop)
    dispatcher.start()

    rate = node.create_rate(config.control_frequency)

    upper_body_policy_subscriber = ROSMsgSubscriber(CONTROL_GOAL_TOPIC)

    last_teleop_cmd = None
    last_data_collection_event_id = 0
    last_data_abort_event_id = 0
    try:
        while ros_manager.ok():
            t_start = time.monotonic()
            with telemetry.timer("total_loop"):
                # Step simulator if in sync mode
                with telemetry.timer("step_simulator"):
                    if env.sim and config.sim_sync_mode:
                        env.step_simulator()

                # Measure observation time
                with telemetry.timer("observe"):
                    obs = env.observe()
                    wbc_policy.set_observation(obs)

                # Measure policy setup time
                with telemetry.timer("policy_setup"):
                    upper_body_cmd = upper_body_policy_subscriber.get_msg()

                    t_now = time.monotonic()

                    wbc_goal = {}
                    if upper_body_cmd:
                        wbc_goal = upper_body_cmd.copy()
                        last_teleop_cmd = upper_body_cmd.copy()
                        if config.ik_indicator:
                            env.set_ik_indicator(upper_body_cmd)
                    if keyboard_loco_controller is not None:
                        wbc_goal.update(
                            keyboard_loco_controller.get_control_goal(
                                env.lower_body_active()
                            )
                        )
                        last_teleop_cmd = wbc_goal.copy()
                    if wbc_goal.get("emergency_stop", False):
                        print("Emergency stop requested from teleop")
                        wbc_goal["navigate_cmd"] = DEFAULT_NAV_CMD
                        wbc_goal["set_policy_action"] = False
                    env.handle_control_goal(wbc_goal)
                    # Send goal to policy
                    if wbc_goal:
                        wbc_goal["interpolation_garbage_collection_time"] = t_now - 2 * (
                            1 / config.control_frequency
                        )
                        wbc_policy.set_goal(wbc_goal)

                # Measure policy action calculation time
                with telemetry.timer("policy_action"):
                    wbc_action = wbc_policy.get_action(time=t_now)

                # Measure action queue time
                with telemetry.timer("queue_action"):
                    env.queue_action(wbc_action)

                # Publish status information for InteractiveModeController
                with telemetry.timer("publish_status"):
                    # Get policy status - check if the lower body policy has use_policy_action enabled
                    policy_use_action = False
                    if config.lower_body_controller == "unitree_loco":
                        policy_use_action = env.lower_body_active()
                    try:
                        # Access the lower body policy through the decoupled whole body policy
                        if (
                            config.lower_body_controller != "unitree_loco"
                            and hasattr(wbc_policy, "lower_body_policy")
                        ):
                            policy_use_action = getattr(
                                wbc_policy.lower_body_policy, "use_policy_action", False
                            )
                    except (AttributeError, TypeError):
                        policy_use_action = False

                    policy_status_msg = {"use_policy_action": policy_use_action, "timestamp": t_now}
                    lower_body_policy_status_pub.publish(policy_status_msg)

                    # Get joint safety status from G1Env (which already runs the safety monitor)
                    joint_safety_ok = env.get_joint_safety_status()

                    joint_safety_status_msg = {
                        "joint_safety_ok": joint_safety_ok,
                        "timestamp": t_now,
                    }
                    joint_safety_status_pub.publish(joint_safety_status_msg)

                # PICO recording events carry a persistent unique identifier
                # so they survive the teleop/control loop timing boundary.
                data_collection_event_id = int(wbc_goal.get("data_collection_event_id", 0))
                data_abort_event_id = int(wbc_goal.get("data_abort_event_id", 0))

                new_data_collection_event = (
                    data_collection_event_id != 0
                    and data_collection_event_id != last_data_collection_event_id
                )
                new_data_abort_event = (
                    data_abort_event_id != 0
                    and data_abort_event_id != last_data_abort_event_id
                )

                # Start or stop data collection. Preserve the one-frame bool
                # fallback for non-PICO and older teleop publishers.
                if new_data_collection_event or (
                    data_collection_event_id == 0
                    and wbc_goal.get("toggle_data_collection", False)
                ):
                    print("Recording toggle received from PICO", flush=True)
                    dispatcher.handle_key("c")
                if new_data_collection_event:
                    last_data_collection_event_id = data_collection_event_id

                # Abort the current episode
                if new_data_abort_event or (
                    data_abort_event_id == 0 and wbc_goal.get("toggle_data_abort", False)
                ):
                    print("Recording discard received from PICO", flush=True)
                    dispatcher.handle_key("x")
                if new_data_abort_event:
                    last_data_abort_event_id = data_abort_event_id

                if env.use_sim and wbc_goal.get("reset_env_and_policy", False):
                    print("Resetting sim environment and policy")
                    sim_reset_pub.publish({"reason": "teleop_reset_env_and_policy", "timestamp": time.time()})
                    # Reset teleop policy & sim env
                    dispatcher.handle_key("k")

                    # Clear upper body commands
                    upper_body_policy_subscriber._msg = None
                    upper_body_cmd = {
                        "target_upper_body_pose": obs["q"][
                            robot_model.get_joint_group_indices("upper_body")
                        ],
                        "wrist_pose": DEFAULT_WRIST_POSE,
                        "base_height_command": DEFAULT_BASE_HEIGHT,
                        "navigate_cmd": DEFAULT_NAV_CMD,
                    }
                    last_teleop_cmd = upper_body_cmd.copy()

                    time.sleep(0.5)

                msg = deepcopy(obs)
                for key in obs.keys():
                    if key.endswith("_image"):
                        del msg[key]

                # exporting data
                if last_teleop_cmd:
                    msg.update(
                        {
                            "action": wbc_action["q"],
                            "action.eef": last_teleop_cmd.get("wrist_pose", DEFAULT_WRIST_POSE),
                            "base_height_command": last_teleop_cmd.get(
                                "base_height_command", DEFAULT_BASE_HEIGHT
                            ),
                            "navigate_command": last_teleop_cmd.get(
                                "navigate_cmd", DEFAULT_NAV_CMD
                            ),
                            "timestamps": {
                                "main_loop": time.time(),
                                "proprio": time.time(),
                            },
                        }
                    )
                data_exp_pub.publish(msg)
                end_time = time.monotonic()

            if env.sim and (not env.sim.sim_thread or not env.sim.sim_thread.is_alive()):
                raise RuntimeError("Simulator thread is not alive")

            rate.sleep()

            # Log timing information every 100 iterations (roughly every 2 seconds at 50Hz)
            if config.verbose_timing:
                # When verbose timing is enabled, always show timing
                telemetry.log_timing_info(context="G1 Control Loop", threshold=0.0)
    except ros_manager.exceptions() as e:
        print(f"ROSManager interrupted by user: {e}")
    finally:
        print("Cleaning up...")
        # the order of the following is important
        dispatcher.stop()
        ros_manager.shutdown()
        env.close()


if __name__ == "__main__":
    config = tyro.cli(ControlLoopConfig)
    main(config)
