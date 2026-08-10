#include <array>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "../include/localmotion_kplanner_tensorrt.hpp"

TEST(PlannerTensorRT, LoadsAndRunsOneInference) {
    const char* model_path = std::getenv("SONIC_PLANNER_TEST_MODEL");
    const char* version_text = std::getenv("SONIC_PLANNER_TEST_VERSION");
    ASSERT_NE(model_path, nullptr) << "SONIC_PLANNER_TEST_MODEL is required";
    ASSERT_NE(version_text, nullptr) << "SONIC_PLANNER_TEST_VERSION is required";

    PlannerConfig config;
    config.model_path = model_path;

    try {
        config.version = std::stoi(version_text);
    } catch (const std::exception& error) {
        FAIL() << "Invalid SONIC_PLANNER_TEST_VERSION: " << error.what();
    }
    ASSERT_GE(config.version, 0);
    ASSERT_LE(config.version, 2);

    std::unique_ptr<LocalMotionPlannerTensorRT> planner;
    try {
        // This test only runs planner inference. It does not initialize Unitree
        // DDS or create any robot state/control publisher.
        planner = std::make_unique<LocalMotionPlannerTensorRT>(false, 0, config);
    } catch (const std::exception& error) {
        FAIL() << "TensorRT planner construction failed: " << error.what();
    }

    const std::array<double, 4> base_quaternion{1.0, 0.0, 0.0, 0.0};
    const std::array<double, 29> joint_positions{};
    ASSERT_TRUE(planner->Initialize(base_quaternion, joint_positions));
    EXPECT_TRUE(planner->planner_state_.initialized);
    EXPECT_TRUE(planner->motion_available_);
    EXPECT_GT(planner->planner_motion_50hz_.timesteps, 0);
}
