#pragma once

// Unitree brainco_hand_service d71996b: Revo2, six normalized motors per side.
// Official tau_est is status.currents / 1000 (A), NOT a torque estimate.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <yaml-cpp/yaml.h>
#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

class BraincoHands {
 public:
  using Six = std::array<double, 6>;
  struct Snapshot {
    Six q{}, speed{}, current{}, sent{};
    double age = std::numeric_limits<double>::infinity();
    bool command_sent = false;
  };

  void initialize(const std::string& config_path) {
    const auto cfg = YAML::LoadFile(config_path);
    const std::array<std::string, 6> names = {
        "thumb", "thumb_aux", "index", "middle", "ring", "pinky"};
    if (cfg["hand_type"].as<std::string>() != "brainco_revo2" ||
        !cfg["joint_names"].IsSequence() || cfg["joint_names"].size() != 6)
      throw std::runtime_error("Expected BrainCo Revo2 config and six official joint names");
    for (size_t i = 0; i < 6; ++i)
      if (cfg["joint_names"][i].as<std::string>() != names[i])
        throw std::runtime_error("BrainCo joint_names must use official motor order");
    speed_ = cfg["speed"].as<double>();
    requireUnit(speed_);
    if (speed_ == 0) throw std::runtime_error("BrainCo command speed must be > 0");
    for (int side = 0; side < 2; ++side) {
      const std::string name = side == 0 ? "left" : "right";
      open_[side] = readSix(cfg[name]["open"]);
      grasp_[side] = readSix(cfg[name]["grasp"]);
      target_[side] = open_[side];
      publishers_[side] = std::make_shared<unitree::robot::ChannelPublisher<unitree_go::msg::dds_::MotorCmds_>>(
          "rt/brainco/" + name + "/cmd");
      subscribers_[side] = std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::MotorStates_>>(
          "rt/brainco/" + name + "/state");
      publishers_[side]->InitChannel();
      subscribers_[side]->InitChannel([this, side](const void* p) {
        receive(side, *static_cast<const unitree_go::msg::dds_::MotorStates_*>(p));
      }, 1);
    }
    // No command is sent on initialization: wait for an explicit control goal.
  }

  void SetMaxCloseRatio(double ratio) {
    requireUnit(ratio);
    std::lock_guard<std::mutex> lock(mutex_);
    ratio_ = ratio;
  }
  double GetMaxCloseRatio() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ratio_;
  }
  void setAllJointsCommand(bool left, const std::array<double, 7>& legacy) {
    bool close = false;
    for (double q : legacy) {
      if (!finite(q)) throw std::runtime_error("Non-finite legacy hand command");
      close |= std::abs(q) > 0.05;
    }
    const int side = left ? 0 : 1;
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < 6; ++i)
      target_[side][i] = open_[side][i] +
          (close ? ratio_ * (grasp_[side][i] - open_[side][i]) : 0.0);
    active_[side] = true;
  }
  void writeOnce() {
    const auto now = Clock::now();
    if (now - last_write_ < std::chrono::milliseconds(10)) return;
    last_write_ = now;
    for (int side = 0; side < 2; ++side) {
      Six target;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!active_[side]) continue;
        target = target_[side];
      }
      unitree_go::msg::dds_::MotorCmds_ cmd;
      cmd.cmds().resize(6);
      for (int i = 0; i < 6; ++i) {
        cmd.cmds()[i].q(static_cast<float>(target[i]));
        cmd.cmds()[i].dq(static_cast<float>(speed_));
      }
      if (publishers_[side]->Write(cmd)) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int i = 0; i < 6; ++i)
          state_[side].sent[i] = cmd.cmds()[i].q();
        state_[side].command_sent = true;
      }
    }
  }
  Snapshot snapshot(bool left) const {
    const int side = left ? 0 : 1;
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = state_[side];
    if (received_[side])
      result.age = std::chrono::duration<double>(Clock::now() - seen_[side]).count();
    return result;
  }

 private:
  using Clock = std::chrono::steady_clock;
  // Bitwise check also works when the body controller uses -ffast-math.
  static bool finite(double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
  }
  static void requireUnit(double value) {
    if (!finite(value) || value < 0 || value > 1)
      throw std::runtime_error("BrainCo q/dq configuration must be finite in [0,1]");
  }
  static Six readSix(const YAML::Node& node) {
    if (!node.IsSequence() || node.size() != 6)
      throw std::runtime_error("BrainCo open/grasp must contain six values");
    Six result;
    for (int i = 0; i < 6; ++i) { result[i] = node[i].as<double>(); requireUnit(result[i]); }
    return result;
  }
  void receive(int side, const unitree_go::msg::dds_::MotorStates_& msg) {
    if (msg.states().size() != 6) return;
    Snapshot incoming;
    for (int i = 0; i < 6; ++i) {
      incoming.q[i] = msg.states()[i].q();
      incoming.speed[i] = msg.states()[i].dq();
      incoming.current[i] = msg.states()[i].tau_est();
      if (!finite(incoming.q[i]) || incoming.q[i] < 0 || incoming.q[i] > 1 ||
          !finite(incoming.speed[i]) || !finite(incoming.current[i])) return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    state_[side].q = incoming.q;
    state_[side].speed = incoming.speed;
    state_[side].current = incoming.current;
    seen_[side] = Clock::now();
    received_[side] = true;
  }
  mutable std::mutex mutex_;
  std::array<Snapshot, 2> state_{};
  std::array<Six, 2> open_{}, grasp_{}, target_{};
  std::array<bool, 2> active_{}, received_{};
  std::array<Clock::time_point, 2> seen_{};
  Clock::time_point last_write_{};
  double speed_ = 1.0, ratio_ = 1.0;
  std::array<unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::MotorCmds_>, 2> publishers_;
  std::array<unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::MotorStates_>, 2> subscribers_;
};
