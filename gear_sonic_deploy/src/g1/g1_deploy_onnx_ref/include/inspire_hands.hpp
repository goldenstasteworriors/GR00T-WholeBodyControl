/**
 * @file inspire_hands.hpp
 * @brief Driver adapter for Unitree RH56DFTP Inspire hands.
 *
 * The full-body controller still produces the legacy 7-DOF-per-hand buffers.
 * This adapter maps those buffers to the two states needed for RH56DFTP:
 * open / grasp are task-specific, selected by SONIC_HAND_TASK.
 *
 * DDS order is:
 * [little, ring, middle, index, thumb_bend, thumb_rotate] for right hand,
 * then the same six joints for left hand.
 */

#ifndef INSPIRE_HANDS_HPP
#define INSPIRE_HANDS_HPP

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "utils.hpp"

static constexpr int INSPIRE_MOTOR_MAX = 6;
static constexpr int INSPIRE_LEGACY_MOTOR_MAX = 7;

class InspireHands
{
public:
    InspireHands() = default;

    void initialize(const std::string &networkInterface)
    {
        if (!networkInterface.empty())
        {
            unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface.c_str());
        }

        cmd_.cmds().resize(INSPIRE_MOTOR_MAX * 2);
        publisher_.reset(new unitree::robot::ChannelPublisher<unitree_go::msg::dds_::MotorCmds_>("rt/inspire/cmd"));
        subscriber_.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::MotorStates_>("rt/inspire/state"));
        publisher_->InitChannel();
        subscriber_->InitChannel([this](const void *message) { this->onState(message); }, 1);

        const char *hand_task_env = std::getenv("SONIC_HAND_TASK");
        hand_task_ = hand_task_env == nullptr ? "pick_up_pipette" : std::string(hand_task_env);
        loadTaskConfig();
        left_q_ = taskOpenQ();
        right_q_ = taskOpenQ();
        writeOnce();
    }

    void SetMaxCloseRatio(double ratio)
    {
        max_close_ratio_ = std::max(0.2, std::min(1.0, ratio));
    }

    double GetMaxCloseRatio() const { return max_close_ratio_; }

    void writeOnce()
    {
        if (!publisher_) { return; }
        for (int i = 0; i < INSPIRE_MOTOR_MAX; ++i)
        {
            cmd_.cmds()[i].q(static_cast<float>(right_q_[i]));
            cmd_.cmds()[i + INSPIRE_MOTOR_MAX].q(static_cast<float>(left_q_[i]));
        }
        publisher_->Write(cmd_);
    }

    std::shared_ptr<const unitree_go::msg::dds_::MotorStates_> getState(bool /*is_left*/) const
    {
        return state_buffer_.GetDataWithTime().data;
    }

    bool hasState(bool /*is_left*/) const
    {
        return state_buffer_.GetDataWithTime().HasData();
    }

    void setAllJointsCommand(bool is_left, const std::array<double, INSPIRE_LEGACY_MOTOR_MAX> &q)
    {
        const bool grasp = legacyCommandIsGrasp(q);
        setInspireCommand(is_left, grasp ? taskGraspQ() : taskOpenQ());
    }

    void open(bool is_left, double /*kp*/ = 1.5, double /*kd*/ = 0.1)
    {
        setInspireCommand(is_left, taskOpenQ());
    }

    void close(bool is_left, double /*kp*/ = 1.5, double /*kd*/ = 0.1)
    {
        setInspireCommand(is_left, taskGraspQ());
    }

    void hold(bool /*is_left*/, double /*kp*/ = 1.5, double /*kd*/ = 0.1) {}
    void stop(bool is_left) { open(is_left); }

private:
    void loadTaskConfig()
    {
        task_open_q_ = OPEN_Q;
        task_grasp_q_ = PICK_UP_PIPETTE_GRASP_Q;

        const char *config_env = std::getenv("SONIC_HAND_TASK_CONFIG");
        std::vector<std::string> candidate_paths;
        if (config_env != nullptr && std::string(config_env).size() > 0)
        {
            candidate_paths.emplace_back(config_env);
        }
        candidate_paths.emplace_back("../gear_sonic/config/data_collection/inspire_hand_tasks.json");
        candidate_paths.emplace_back("gear_sonic/config/data_collection/inspire_hand_tasks.json");

        for (const auto &path : candidate_paths)
        {
            std::ifstream file(path);
            if (!file.is_open()) { continue; }
            try
            {
                nlohmann::json config;
                file >> config;
                if (!config.contains(hand_task_))
                {
                    std::cerr << "[InspireHands] Hand task '" << hand_task_
                              << "' not found in " << path << ". Using default." << std::endl;
                    return;
                }
                const auto &task = config.at(hand_task_);
                task_open_q_ = parsePose(task.at("open"), hand_task_ + ".open");
                task_grasp_q_ = parsePose(task.at("pressed"), hand_task_ + ".pressed");
                std::cout << "[InspireHands] Loaded hand task '" << hand_task_
                          << "' from " << path << std::endl;
                return;
            }
            catch (const std::exception &e)
            {
                std::cerr << "[InspireHands] Failed to parse " << path << ": "
                          << e.what() << ". Using default." << std::endl;
                return;
            }
        }

        std::cerr << "[InspireHands] Hand task config not found. Using default." << std::endl;
    }

    static std::array<double, INSPIRE_MOTOR_MAX> parsePose(
        const nlohmann::json &values,
        const std::string &label)
    {
        if (!values.is_array() || values.size() != INSPIRE_MOTOR_MAX)
        {
            throw std::runtime_error(label + " must be an array of 6 numbers");
        }
        std::array<double, INSPIRE_MOTOR_MAX> pose {};
        for (int i = 0; i < INSPIRE_MOTOR_MAX; ++i)
        {
            pose[i] = std::max(0.0, std::min(1.0, values.at(i).get<double>()));
        }
        return pose;
    }

    void onState(const void *message)
    {
        const auto *incoming = static_cast<const unitree_go::msg::dds_::MotorStates_ *>(message);
        state_buffer_.SetData(*incoming);
    }

    static bool legacyCommandIsGrasp(const std::array<double, INSPIRE_LEGACY_MOTOR_MAX> &q)
    {
        for (double value : q)
        {
            if (std::abs(value) > 0.05) { return true; }
        }
        return false;
    }

    const std::array<double, INSPIRE_MOTOR_MAX>& taskOpenQ() const
    {
        return task_open_q_;
    }

    const std::array<double, INSPIRE_MOTOR_MAX>& taskGraspQ() const
    {
        return task_grasp_q_;
    }

    void setInspireCommand(bool is_left, const std::array<double, INSPIRE_MOTOR_MAX> &q)
    {
        if (is_left)
        {
            left_q_ = q;
        }
        else
        {
            right_q_ = q;
        }
    }

    unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::MotorCmds_> publisher_;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::MotorStates_> subscriber_;
    unitree_go::msg::dds_::MotorCmds_ cmd_;
    DataBuffer<unitree_go::msg::dds_::MotorStates_> state_buffer_;

    std::array<double, INSPIRE_MOTOR_MAX> left_q_ {};
    std::array<double, INSPIRE_MOTOR_MAX> right_q_ {};
    std::array<double, INSPIRE_MOTOR_MAX> task_open_q_ {};
    std::array<double, INSPIRE_MOTOR_MAX> task_grasp_q_ {};
    double max_close_ratio_ = 1.0;
    std::string hand_task_ = "pick_up_pipette";

    static constexpr std::array<double, INSPIRE_MOTOR_MAX> OPEN_Q = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    static constexpr std::array<double, INSPIRE_MOTOR_MAX> PICK_UP_PIPETTE_GRASP_Q = {0.0, 0.0, 0.0, 0.0, 1.0, 1.0};
};

#endif // INSPIRE_HANDS_HPP
