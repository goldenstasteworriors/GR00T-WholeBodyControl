"""Robot-local Balance fallback using the existing 86 x 6 WBC observation contract."""

from collections import deque

import numpy as np

from decoupled_wbc.control.utils.gear_wbc_utils import get_gravity_orientation, load_config


class OnboardBalanceHold:
    def __init__(self, config_path, model_path):
        import onnxruntime as ort

        self.config, _ = load_config(config_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.history = deque(maxlen=self.config["obs_history_len"])

    def start(self, context):
        self.action = np.asarray(context["action"], dtype=np.float32).copy()
        self.height = float(np.asarray(context["height"]).reshape(-1)[0])
        self.rpy = np.asarray(context["rpy"], dtype=np.float32).copy()
        self.history.clear()
        for row in np.asarray(context["history"], dtype=np.float32).reshape(-1, 86):
            self.history.append(row.copy())

    def step(self, q, dq, quaternion, gyroscope):
        cfg = self.config
        defaults = np.zeros(29, dtype=np.float32)
        defaults[:15] = cfg["default_angles"]
        obs = np.zeros(86, dtype=np.float32)
        # Navigation is permanently zero after the disconnect latch.
        obs[3] = self.height
        obs[4:7] = self.rpy
        obs[7:10] = np.asarray(gyroscope) * cfg["ang_vel_scale"]
        obs[10:13] = get_gravity_orientation(np.asarray(quaternion))
        obs[13:42] = (np.asarray(q) - defaults) * cfg["dof_pos_scale"]
        obs[42:71] = np.asarray(dq) * cfg["dof_vel_scale"]
        obs[71:86] = self.action
        self.history.append(obs)
        inputs = np.concatenate(self.history).reshape(1, cfg["num_obs"])
        self.action = self.session.run(None, {self.input_name: inputs})[0].reshape(15)
        return self.action * cfg["action_scale"] + cfg["default_angles"]

