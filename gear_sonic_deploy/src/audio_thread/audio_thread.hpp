#pragma once

#include <memory>
#include <string>
#include <stop_token>
#include <thread>
#include <unitree/robot/g1/audio/g1_audio_client.hpp>

struct AudioCommand {
  bool streaming_data_absent = false;
  bool motor_error = false;
  bool low_state_late = false;
  std::string tts_message;  // One-shot TTS message (spoken once when non-empty)
  bool high_temperature = false;             // Continuous warning while true
  std::string high_temperature_message;      // Spoken every poll while high_temperature is true
};

class AudioThread {
 public:
  AudioThread();

  void SetCommand(const AudioCommand& command);

 private:
  void loop(std::stop_token st);

  unitree::robot::g1::AudioClient client_;
  std::jthread thread_;

  std::mutex command_mutex_;
  AudioCommand command_;
  AudioCommand command_last_;

  // Cooldowns reduce repeated motor/LowState/high-temperature speech.
  // Streaming-data warnings retain the original rising-edge behaviour.
  std::chrono::steady_clock::time_point last_motor_error_tts_{};
  std::chrono::steady_clock::time_point last_low_state_late_tts_{};
  std::chrono::steady_clock::time_point last_high_temp_tts_{};
  static constexpr std::chrono::seconds TRANSIENT_WARNING_TTS_INTERVAL{30};
  static constexpr std::chrono::seconds HIGH_TEMP_TTS_INTERVAL{60};
};
