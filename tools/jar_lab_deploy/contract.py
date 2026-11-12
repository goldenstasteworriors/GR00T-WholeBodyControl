"""Observation contract shared by GPU inference and the CPU robot controller."""
JOINT_NAMES = [f"{side}_{joint}_joint" for side in ("left", "right")
               for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                             "wrist_roll", "wrist_pitch", "wrist_yaw")]
JOINT_NAMES += ["left_hand_dex1_motor", "right_hand_dex1_motor"]
BODY_JOINT_NAMES = [f"{side}_{joint}_joint" for side in ("left", "right")
                    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
BODY_JOINT_NAMES += ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"] + JOINT_NAMES[:14]
CAMERAS = ["ego_view", "left_wrist", "right_wrist"]
TASKS = ["TAKE_TUBE", "TAKE_DROPPER_AND_ASPIRATE", "DISPENSE_TO_HELD_TUBE",
         "RETURN_DROPPER", "RETURN_TUBE_TO_RACK", "PLACE_TUBE_IN_BATH"]
