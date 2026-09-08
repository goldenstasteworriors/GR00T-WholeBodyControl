"""Record on ThinkPad without launching controllers or changing host clocks."""
from dataclasses import dataclass
from pathlib import Path
import tyro
from gear_sonic.scripts.run_decoupled_vla_data_exporter import (
    DecoupledVLADataExporterConfig, main as export_main, get_g1_robot_model,
    get_features_sonic_inspire6, load_inspire_hand_profiles,
)


@dataclass
class ThinkPadConfig(DecoupledVLADataExporterConfig):
    inspire_hand_pose_config: str = str(
        Path(__file__).resolve().parents[1] / "config/data_collection/thinkpad_grab_beaker_pose.json"
    )
    receive_local: bool = True
    robot_config_timeout: float = 30.0
    audio_cues: bool = False
    preflight: bool = False
    """Check imports, robot model and profiles; never opens camera or ROS."""


def main():
    config = tyro.cli(ThinkPadConfig)
    if not config.receive_local:
        raise ValueError("This entry point only supports receive-local")
    if config.preflight:
        import pyrealsense2
        model = get_g1_robot_model()
        features = get_features_sonic_inspire6(model)
        load_inspire_hand_profiles(config.inspire_hand_pose_config)
        print(f"Preflight OK: model joints={model.num_joints}; features={len(features)}; no hardware opened")
        return
    export_main(config)


if __name__ == "__main__":
    main()
