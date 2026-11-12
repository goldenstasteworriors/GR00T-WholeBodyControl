"""JAR observation inference on the controller host; ROS goal output is opt-in."""
import argparse
import signal
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from contract import TASKS
from rpc import Client
from sensors import Cameras


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonic-root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--operator-epoch", type=int, required=True)
    parser.add_argument("--camera-endpoint", default="tcp://192.168.123.164:5555")
    parser.add_argument("--policy-endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--chunks", type=int, default=1000000)
    args = parser.parse_args()
    if args.chunks < 1:
        parser.error("chunks must be positive")
    sys.path.insert(0, str(args.sonic_root))
    from decoupled_wbc.control.utils.ros_utils import ROSManager, ROSMsgSubscriber, ROSMsgPublisher
    from decoupled_wbc.control.main.constants import STATE_TOPIC_NAME, CONTROL_GOAL_TOPIC
    from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
    robot = instantiate_g1_robot_model(waist_location="lower_body")
    indices = np.asarray(robot.get_body_actuated_joint_indices())[15:29]
    ros = ROSManager(node_name="jar_observation_client")
    states = ROSMsgSubscriber(STATE_TOPIC_NAME)
    publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC) if args.execute else None
    cameras = Cameras(args.camera_endpoint)
    stopped, lock = threading.Event(), threading.Lock()
    latest, prediction = {}, {}
    binding = str(uuid.uuid4())
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopped.set())

    def infer():
        client = Client(args.policy_endpoint)
        try:
            for _ in range(args.chunks):
                if stopped.is_set():
                    break
                with lock:
                    packet = latest.copy()
                if not packet:
                    stopped.wait(.02)
                    continue
                images, ages = cameras.history()
                captured = time.monotonic()
                if captured - packet["received"] > .15:
                    raise TimeoutError("Controller state stream stale")
                result = client.predict(packet["state"], images, ages, steps=1)
                if result["task"] != args.task:
                    raise ValueError("Model service task does not match the selected operation")
                with lock:
                    current = latest.copy()
                    if np.max(np.abs(current["state"][:14] - packet["state"][:14])) > .1:
                        raise RuntimeError("Measured arms moved too far during inference; reset the session")
                    prediction.update(target=result["positions"][0], captured=captured)
            stopped.set()
        except Exception as error:
            print(f"JAR inference stopped: {error}", flush=True)
            prediction["error"] = str(error)
            stopped.set()
        finally:
            client.close()

    worker = None
    sequence, received, startup, armed = 0, 0., time.monotonic(), False
    try:
        while not stopped.is_set() and ros.ok():
            now = time.monotonic()
            message = states.get_msg()
            if message is not None:
                if message.get("control_mode") != "running" or message.get("jar_operator_epoch") != args.operator_epoch:
                    if armed:
                        raise RuntimeError("Controller phase/epoch changed; this inference session is obsolete")
                    if now-startup > 10:
                        raise TimeoutError("Controller did not arm the requested inference epoch")
                    stopped.wait(.01)
                    continue
                armed = True
                state = np.r_[np.asarray(message["q"])[indices], message["left_hand_dex1_q"], message["right_hand_dex1_q"]]
                if state.shape != (16,) or not np.isfinite(state).all():
                    raise ValueError("Invalid measured JAR observation")
                received = now
                with lock:
                    latest.update(state=state, received=now)
                if worker is None:
                    try:
                        cameras.history()
                    except TimeoutError:
                        if now-startup > 10:
                            raise TimeoutError("Waiting for three fresh camera streams")
                        stopped.wait(.01)
                        continue
                    worker = threading.Thread(target=infer, name="jar-inference")
                    worker.start()
            if now - (received or startup) > (0.15 if received else 10.):
                raise TimeoutError("No fresh controller observations")
            with lock:
                target = prediction.copy()
            if publisher is not None and target.get("target") is not None and now - target["captured"] <= .75:
                publisher.publish(dict(jar_target=target["target"], jar_task=args.task, jar_binding_id=binding,
                                       jar_sequence=sequence, jar_sent_monotonic=now, jar_reset=True,
                                       jar_operator_epoch=args.operator_epoch))
                sequence += 1
            stopped.wait(1/30)
        if prediction.get("error"):
            raise RuntimeError(prediction["error"])
    finally:
        stopped.set()
        if publisher is not None:
            publisher.publish(dict(jar_stop=True, jar_binding_id=binding, jar_operator_epoch=args.operator_epoch))
        if worker is not None:
            worker.join(timeout=17)
        cameras.close()


if __name__ == "__main__":
    main()
