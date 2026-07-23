import time

import rclpy
import tyro

from decoupled_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    LOWER_BODY_POLICY_STATUS_TOPIC,
    STATE_TOPIC_NAME,
)
from decoupled_wbc.control.main.teleop.configs.configs import TeleopConfig
from decoupled_wbc.control.policy.lerobot_replay_policy import LerobotReplayPolicy
from decoupled_wbc.control.policy.teleop_policy import TeleopPolicy
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK
from decoupled_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher, ROSMsgSubscriber
from decoupled_wbc.control.utils.telemetry import Telemetry

TELEOP_NODE_NAME = "TeleopPolicy"


def main(config: TeleopConfig):
    ros_manager = ROSManager(node_name=TELEOP_NODE_NAME)
    node = ros_manager.node

    if config.robot == "g1":
        waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
        robot_model = instantiate_g1_robot_model(
            waist_location=waist_location,
            high_elbow_pose=config.high_elbow_pose,
            with_hands=config.with_hands,
        )
        if config.with_hands:
            left_hand_ik_solver, right_hand_ik_solver = instantiate_g1_hand_ik_solver()
            hand_control_device = config.hand_control_device
        else:
            left_hand_ik_solver = None
            right_hand_ik_solver = None
            hand_control_device = None
    else:
        raise ValueError(f"Unsupported robot name: {config.robot}")

    if config.lerobot_replay_path:
        teleop_policy = LerobotReplayPolicy(
            robot_model=robot_model, parquet_path=config.lerobot_replay_path
        )
    else:
        print("running teleop policy, waiting teleop policy to be initialized...")
        retargeting_ik = TeleopRetargetingIK(
            robot_model=robot_model,
            left_hand_ik_solver=left_hand_ik_solver,
            right_hand_ik_solver=right_hand_ik_solver,
            enable_visualization=config.enable_visualization,
            body_active_joint_groups=["upper_body"],
        )
        teleop_policy = TeleopPolicy(
            robot_model=robot_model,
            retargeting_ik=retargeting_ik,
            body_control_device=config.body_control_device,
            hand_control_device=hand_control_device,
            body_streamer_ip=config.body_streamer_ip,  # vive tracker, leap motion does not require
            body_streamer_keyword=config.body_streamer_keyword,
            enable_real_device=config.enable_real_device,
            replay_data_path=config.teleop_replay_path,
        )

    # Create a publisher for the navigation commands
    control_publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
    robot_state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
    lower_body_policy_status_subscriber = ROSMsgSubscriber(LOWER_BODY_POLICY_STATUS_TOPIC)

    # Create rate controller
    rate = node.create_rate(config.teleop_frequency)
    iteration = 0
    time_to_get_to_initial_pose = 2  # seconds

    telemetry = Telemetry(window_size=100)
    pending_policy_action = None

    try:
        while rclpy.ok():
            with telemetry.timer("total_loop"):
                t_start = time.monotonic()
                robot_state = robot_state_subscriber.get_msg()
                lower_body_policy_status = lower_body_policy_status_subscriber.get_msg()
                if lower_body_policy_status is not None:
                    lower_body_policy_active = bool(
                        lower_body_policy_status.get("use_policy_action", False)
                    )
                    if hasattr(teleop_policy, "set_lower_body_policy_active"):
                        teleop_policy.set_lower_body_policy_active(lower_body_policy_active)
                    if (
                        pending_policy_action is not None
                        and lower_body_policy_active == pending_policy_action
                    ):
                        print(
                            "Lower-body policy request acknowledged: "
                            f"{'enabled' if lower_body_policy_active else 'disabled'}"
                        )
                        pending_policy_action = None
                if robot_state is not None and hasattr(teleop_policy, "set_robot_state"):
                    teleop_policy.set_robot_state(robot_state)
                # Get the current teleop action
                with telemetry.timer("get_action"):
                    data = teleop_policy.get_action()

                requested_policy_action = data.get("set_policy_action")
                if requested_policy_action is not None:
                    pending_policy_action = bool(requested_policy_action)
                    print(
                        "Lower-body policy request queued: "
                        f"{'enable' if pending_policy_action else 'disable'}"
                    )
                if pending_policy_action is not None:
                    # This is a state transition, not a best-effort one-frame event.
                    # Repeat it until the control loop reports the requested state.
                    data["set_policy_action"] = pending_policy_action
                    if not pending_policy_action:
                        data["emergency_stop"] = True

                # Add timing information to the message
                t_now = time.monotonic()
                data["timestamp"] = t_now

                # Set target completion time - longer for initial pose, then match control frequency.
                if iteration == 0:
                    data["target_time"] = t_now + time_to_get_to_initial_pose
                else:
                    data["target_time"] = t_now + (1 / config.teleop_frequency)

                # Publish the teleop command
                with telemetry.timer("publish_teleop_command"):
                    control_publisher.publish(data)

                # For the initial pose, wait the full duration before continuing
                if iteration == 0:
                    print(f"Moving to initial pose for {time_to_get_to_initial_pose} seconds")
                    time.sleep(time_to_get_to_initial_pose)
                iteration += 1
            end_time = time.monotonic()
            if config.verbose_timing and (end_time - t_start) > (1 / config.teleop_frequency):
                telemetry.log_timing_info(context="Teleop Policy Loop Missed", threshold=0.001)
            rate.sleep()

    except ros_manager.exceptions() as e:
        print(f"ROSManager interrupted by user: {e}")

    finally:
        print("Cleaning up...")
        ros_manager.shutdown()


if __name__ == "__main__":
    config = tyro.cli(TeleopConfig)
    main(config)
