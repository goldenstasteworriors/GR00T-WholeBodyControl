// BrainCo Revo2 only: official DDS hand topics, no body-control publisher.
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

using Clock = std::chrono::steady_clock;
using Cmd = unitree_go::msg::dds_::MotorCmds_;
using State = unitree_go::msg::dds_::MotorStates_;
using Six = std::array<float, 6>;
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }
struct Feedback { Six q{}; Clock::time_point stamp{}; unsigned count = 0; };

int main(int argc, char** argv) {
    try {
        std::string network = "eth0";
        bool move = false;
        int domain = 0;
        double ramp = 3., hold = 1., wait = 8.;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "Usage: run_both_hands.sh [--network eth0] [--move]\n"
                          << "Default: read-only feedback check. --move: open, close, open once.\n"
                          << "Optional: --domain 0 --ramp-seconds 3 --hold-seconds 1 --wait-seconds 8\n";
                return 0;
            }
            if (arg == "--move") { move = true; continue; }
            if (i + 1 == argc) throw std::runtime_error("Missing value: " + arg);
            std::string value = argv[++i];
            if (arg == "--network") network = value;
            else if (arg == "--domain") domain = std::stoi(value);
            else if (arg == "--ramp-seconds") ramp = std::stod(value);
            else if (arg == "--hold-seconds") hold = std::stod(value);
            else if (arg == "--wait-seconds") wait = std::stod(value);
            else throw std::runtime_error("Unknown option: " + arg);
        }
        if (domain < 0 || domain > 232 || !std::isfinite(ramp) || ramp <= 0 ||
            !std::isfinite(hold) || hold < 0 || !std::isfinite(wait) || wait <= 0)
            throw std::runtime_error("Invalid timing/domain argument");
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        unitree::robot::ChannelFactory::Instance()->Init(domain, network);
        std::mutex mutex;
        std::array<Feedback, 2> feedback;
        std::array<std::unique_ptr<unitree::robot::ChannelSubscriber<State>>, 2> subs;
        std::array<std::unique_ptr<unitree::robot::ChannelPublisher<Cmd>>, 2> pubs;
        const std::array<std::string, 2> sides{"left", "right"};
        for (int side = 0; side < 2; ++side) {
            subs[side] = std::make_unique<unitree::robot::ChannelSubscriber<State>>("rt/brainco/" + sides[side] + "/state");
            subs[side]->InitChannel([&, side](const void* p) {
                const auto& state = *static_cast<const State*>(p);
                if (state.states().size() != 6) return;
                Six q;
                for (int j = 0; j < 6; ++j) {
                    q[j] = state.states()[j].q();
                    if (!std::isfinite(q[j]) || q[j] < 0 || q[j] > 1) return;
                }
                std::lock_guard<std::mutex> lock(mutex);
                feedback[side].q = q;
                feedback[side].stamp = Clock::now();
                ++feedback[side].count;
            });
        }
        auto snapshot = [&] { std::lock_guard<std::mutex> lock(mutex); return feedback; };
        auto fresh = [](const Feedback& f) { return f.count && Clock::now() - f.stamp < std::chrono::milliseconds(500); };
        auto print = [&](const std::array<Feedback, 2>& data) {
            for (int side = 0; side < 2; ++side) {
                std::cout << sides[side] << " received=" << data[side].count << " fresh=" << fresh(data[side]) << " q=";
                for (float q : data[side].q) std::cout << q << ' ';
                std::cout << '\n';
            }
            std::cout << std::flush;
        };
        auto deadline = Clock::now() + std::chrono::duration<double>(wait);
        while (!interrupted && Clock::now() < deadline) {
            auto data = snapshot();
            if (data[0].count >= 3 && data[1].count >= 3 && fresh(data[0]) && fresh(data[1])) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        auto initial = snapshot();
        print(initial);
        if (interrupted) return 130;
        if (initial[0].count < 3 || initial[1].count < 3 || !fresh(initial[0]) || !fresh(initial[1]))
            throw std::runtime_error("Both hands need fresh feedback; no commands sent");
        if (!move) { std::cout << "Read-only check passed; no commands sent\n"; return 0; }
        // Publishers are created only after --move and both feedback checks.
        for (int side = 0; side < 2; ++side) {
            pubs[side] = std::make_unique<unitree::robot::ChannelPublisher<Cmd>>("rt/brainco/" + sides[side] + "/cmd");
            pubs[side]->InitChannel();
        }
        std::array<Six, 2> start{initial[0].q, initial[1].q};
        for (float target : {0.f, 1.f, 0.f}) {
            std::cout << "Both hands target=" << target << " (0=open, 1=closed)\n" << std::flush;
            const auto begin = Clock::now();
            auto tick = begin;
            for (;;) {
                if (interrupted) { std::cout << "Interrupted: stop publishing; hands may hold last target\n"; return 130; }
                auto data = snapshot();
                if (!fresh(data[0]) || !fresh(data[1])) throw std::runtime_error("Feedback stale: stopped publishing; hands may hold last target");
                double elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
                double alpha = std::min(1., elapsed / ramp);
                for (int side = 0; side < 2; ++side) {
                    Cmd cmd;
                    cmd.cmds().resize(6);
                    for (int j = 0; j < 6; ++j) {
                        cmd.cmds()[j].q() = start[side][j] + alpha * (target - start[side][j]);
                        cmd.cmds()[j].dq() = 1.f; // Official recommended normalized speed; q is ramped.
                    }
                    if (!pubs[side]->Write(cmd)) throw std::runtime_error("DDS write failed; stopped publishing");
                }
                if (elapsed >= ramp + hold) break;
                tick += std::chrono::milliseconds(20);
                std::this_thread::sleep_until(tick);
            }
            auto actual = snapshot();
            print(actual);
            for (const auto& f : actual) for (float q : f.q)
                if (std::abs(q - target) > .15f)
                    throw std::runtime_error("A finger did not reach the target; stopped before the next phase");
            start = {actual[0].q, actual[1].q};
        }
        std::cout << "Completed one open-close-open cycle on both hands\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
}
