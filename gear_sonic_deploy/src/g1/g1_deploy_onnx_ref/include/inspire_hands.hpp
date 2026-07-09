/**
 * @file inspire_hands.hpp
 * @brief Driver adapter for Unitree RH56DFTP Inspire hands.
 *
 * The full-body controller still produces the legacy 7-DOF-per-hand buffers.
 * This adapter maps those buffers to the two states needed for RH56DFTP:
 * open  = [1, 1, 1, 1, 1, 0.2]
 * grasp = [0.15, 0.15, 0.15, 0.15, 1, 0.2]
 *
 * DDS order is:
 * [little, ring, middle, index, thumb_bend, thumb_rotate] for right hand,
 * then the same six joints for left hand.
 */

#ifndef INSPIRE_HANDS_HPP
#define INSPIRE_HANDS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>

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

        left_q_ = OPEN_Q;
        right_q_ = OPEN_Q;
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
        setInspireCommand(is_left, grasp ? GRASP_Q : OPEN_Q);
    }

    void open(bool is_left, double /*kp*/ = 1.5, double /*kd*/ = 0.1)
    {
        setInspireCommand(is_left, OPEN_Q);
    }

    void close(bool is_left, double /*kp*/ = 1.5, double /*kd*/ = 0.1)
    {
        setInspireCommand(is_left, GRASP_Q);
    }

    void hold(bool /*is_left*/, double /*kp*/ = 1.5, double /*kd*/ = 0.1) {}
    void stop(bool is_left) { open(is_left); }

private:
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
    double max_close_ratio_ = 1.0;

    static constexpr std::array<double, INSPIRE_MOTOR_MAX> OPEN_Q = {1.0, 1.0, 1.0, 1.0, 1.0, 0.2};
    static constexpr std::array<double, INSPIRE_MOTOR_MAX> GRASP_Q = {0.15, 0.15, 0.15, 0.15, 1.0, 0.2};
};

#endif // INSPIRE_HANDS_HPP
