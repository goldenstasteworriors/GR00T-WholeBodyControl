# Raised symmetric startup arms

The NaVILA controller launcher sets `NAVILA_STARTUP_ARM_POSE` to
`deploy/navila/startup_arm_pose.json`. The startup policy loads the named arm
joints after its legacy elbow initialization. Other controller launchers keep
their existing defaults unless they set this environment variable.

The source is the user-specified dataset:
`/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/outputs/thinkpad_imports/whole_pipeline_20261029_161210`.
Of 97 episodes, the 7 indices marked discarded in `meta/info.json` are excluded.
For each of the remaining 90 episodes, the final 30 frames (one second at 30 Hz)
of the right-arm `observation.state` are reduced by joint-wise median. The final
reference is the joint-wise median of those 90 episode medians. Joint ordering
comes from `meta/info.json`; only the 7 right-arm joints are used. The source
dataset is unchanged. `startup_arm_pose_analysis.json` records the analysis.

| Joint | Right, degrees | Left, degrees |
| --- | ---: | ---: |
| Shoulder pitch | -21.81 | -21.81 |
| Shoulder roll | -15.63 | +15.63 |
| Shoulder yaw | -0.62 | +0.62 |
| Elbow | -42.57 | -42.57 |
| Wrist roll | +19.41 | -19.41 |
| Wrist pitch | +26.08 | +26.08 |
| Wrist yaw | -6.10 | +6.10 |

The left arm is mirrored from the right using the actual G1 URDF joint axes:
pitch/elbow keep their signs; roll/yaw reverse signs. The dataset's left arm is
not used. The JSON stores full-precision radians; the table is rounded.

Validation used the existing ThinkPad collection environment and actual G1
robot model without a hardware connection or actuator writes. All 14 joint
values are within model limits. Both wrist-yaw-link origins rise by 0.180402 m
relative to the previous zero arm pose with the same root and lower-body pose.
Left/right wrist positions are mirrored within 2 mm. This is model geometry,
not a measurement of clearance above a particular table.

The original startup ramp remains in use (about two seconds without T-pose).
Navigation goals contain velocities only, so the interpolation policy retains
the raised arm target. Waist and legs retain their existing control logic.
Hands remain disabled as before.

This configuration loads when the controller starts or reinitializes using a
process that already has the environment variable. An already running old
controller is not updated by editing these files. To load the new profile,
exit that controller with `x` in the ThinkPad menu, rerun `navila_start.sh`, then
use the usual three Enter presses: initialize, stand, navigate. The deployment
agent did not restart the real controller or execute the new posture.
