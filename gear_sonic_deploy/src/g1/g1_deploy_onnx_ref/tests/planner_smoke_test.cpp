#include <array>
#include <exception>
#include <iostream>
#include <string>

#include "../include/localmotion_kplanner_tensorrt.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <planner.onnx> <planner_version>" << std::endl;
        return 2;
    }

    PlannerConfig config;
    config.model_path = argv[1];

    try {
        config.version = std::stoi(argv[2]);
    } catch (const std::exception& error) {
        std::cerr << "Invalid planner version: " << error.what() << std::endl;
        return 2;
    }

    if (config.version < 0 || config.version > 2) {
        std::cerr << "Planner version must be 0, 1, or 2" << std::endl;
        return 2;
    }

    try {
        // This test only runs planner inference. It does not initialize Unitree DDS
        // or create any robot state/control publisher.
        LocalMotionPlannerTensorRT planner(false, 0, config);
        const std::array<double, 4> base_quaternion{1.0, 0.0, 0.0, 0.0};
        const std::array<double, 29> joint_positions{};

        if (!planner.Initialize(base_quaternion, joint_positions)) {
            std::cerr << "Planner initialization failed" << std::endl;
            return 1;
        }

        if (!planner.planner_state_.initialized || !planner.motion_available_ ||
            planner.planner_motion_50hz_.timesteps <= 0) {
            std::cerr << "Planner produced no usable motion" << std::endl;
            return 1;
        }

        std::cout << "Planner smoke test passed: "
                  << planner.planner_motion_50hz_.timesteps
                  << " frames at 50 Hz" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Planner smoke test failed: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
