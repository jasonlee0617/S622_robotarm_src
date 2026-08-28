#include "fairino_hardware/fairino_hardware_interface.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>

namespace fairino_hardware
{
namespace
{
constexpr auto kFeedbackStaleLimit = std::chrono::milliseconds(50);
constexpr auto kDiagnosticPeriod = std::chrono::seconds(1);

int64_t steady_now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

hardware_interface::CallbackReturn FairinoHardwareInterface::on_init(
  const hardware_interface::HardwareInfo& sysinfo)
{
  if (hardware_interface::SystemInterface::on_init(sysinfo) !=
      hardware_interface::CallbackReturn::SUCCESS) {
    return hardware_interface::CallbackReturn::ERROR;
  }
  info_ = sysinfo;

  try {
    if (const auto it = info_.hardware_parameters.find("servoj_cmd_t");
        it != info_.hardware_parameters.end()) {
      _servoj_cmd_t = std::stod(it->second);
    }
    if (const auto it = info_.hardware_parameters.find("realtime_state_period_ms");
        it != info_.hardware_parameters.end()) {
      _realtime_state_period_ms = std::stoi(it->second);
    }
  } catch (const std::exception& error) {
    RCLCPP_ERROR(rclcpp::get_logger("FairinoHardwareInterface"),
      "Invalid hardware timing parameter: %s", error.what());
    return hardware_interface::CallbackReturn::ERROR;
  }
  if (!std::isfinite(_servoj_cmd_t) || _servoj_cmd_t <= 0.0 ||
      _servoj_cmd_t > 0.1 || _realtime_state_period_ms <= 0) {
    RCLCPP_ERROR(rclcpp::get_logger("FairinoHardwareInterface"),
      "Invalid timing: servoj_cmd_t=%.6f realtime_state_period_ms=%d",
      _servoj_cmd_t, _realtime_state_period_ms);
    return hardware_interface::CallbackReturn::ERROR;
  }

  _has_finger1 = false;
  _has_finger2 = false;
  for (const auto& joint : info_.joints) {
    _has_finger1 = _has_finger1 || joint.name == "finger1_joint";
    _has_finger2 = _has_finger2 || joint.name == "finger2_joint";
    const bool is_arm_joint = joint.name == "j1" || joint.name == "j2" ||
      joint.name == "j3" || joint.name == "j4" || joint.name == "j5" ||
      joint.name == "j6";
    const bool has_position_state = std::any_of(
      joint.state_interfaces.begin(), joint.state_interfaces.end(),
      [](const auto& interface) { return interface.name == hardware_interface::HW_IF_POSITION; });
    const bool has_velocity_state = std::any_of(
      joint.state_interfaces.begin(), joint.state_interfaces.end(),
      [](const auto& interface) { return interface.name == hardware_interface::HW_IF_VELOCITY; });
    const size_t expected_state_count = is_arm_joint ? 2U : 1U;
    if (joint.command_interfaces.size() != 1 ||
        joint.command_interfaces[0].name != hardware_interface::HW_IF_POSITION ||
        joint.state_interfaces.size() != expected_state_count || !has_position_state ||
        has_velocity_state != is_arm_joint) {
      RCLCPP_FATAL(rclcpp::get_logger("FairinoHardwareInterface"),
        "Joint '%s' must expose position command/state and arm joints must also expose velocity state.",
        joint.name.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }
  }
  if (_has_finger1 != _has_finger2) {
    RCLCPP_FATAL(rclcpp::get_logger("FairinoHardwareInterface"),
      "finger1_joint and finger2_joint must be configured together.");
    return hardware_interface::CallbackReturn::ERROR;
  }
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
FairinoHardwareInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> interfaces;
  for (const auto& joint : info_.joints) {
    double* value = nullptr;
    double* velocity = nullptr;
    if (joint.name == "j1") { value = &_jnt_position_state[0]; velocity = &_jnt_velocity_state[0]; }
    else if (joint.name == "j2") { value = &_jnt_position_state[1]; velocity = &_jnt_velocity_state[1]; }
    else if (joint.name == "j3") { value = &_jnt_position_state[2]; velocity = &_jnt_velocity_state[2]; }
    else if (joint.name == "j4") { value = &_jnt_position_state[3]; velocity = &_jnt_velocity_state[3]; }
    else if (joint.name == "j5") { value = &_jnt_position_state[4]; velocity = &_jnt_velocity_state[4]; }
    else if (joint.name == "j6") { value = &_jnt_position_state[5]; velocity = &_jnt_velocity_state[5]; }
    else if (joint.name == "finger1_joint") value = &_finger_position_state[0];
    else if (joint.name == "finger2_joint") value = &_finger_position_state[1];
    if (value == nullptr) {
      RCLCPP_FATAL(rclcpp::get_logger("FairinoHardwareInterface"),
        "Unknown joint '%s'.", joint.name.c_str());
      continue;
    }
    interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, value);
    if (velocity != nullptr) {
      interfaces.emplace_back(joint.name, hardware_interface::HW_IF_VELOCITY, velocity);
    }
  }
  return interfaces;
}

std::vector<hardware_interface::CommandInterface>
FairinoHardwareInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> interfaces;
  for (const auto& joint : info_.joints) {
    double* value = nullptr;
    if (joint.name == "j1") value = &_jnt_position_command[0];
    else if (joint.name == "j2") value = &_jnt_position_command[1];
    else if (joint.name == "j3") value = &_jnt_position_command[2];
    else if (joint.name == "j4") value = &_jnt_position_command[3];
    else if (joint.name == "j5") value = &_jnt_position_command[4];
    else if (joint.name == "j6") value = &_jnt_position_command[5];
    else if (joint.name == "finger1_joint") value = &_finger_position_command[0];
    else if (joint.name == "finger2_joint") value = &_finger_position_command[1];
    if (value == nullptr) {
      RCLCPP_FATAL(rclcpp::get_logger("FairinoHardwareInterface"),
        "Unknown joint '%s'.", joint.name.c_str());
      continue;
    }
    interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, value);
  }
  return interfaces;
}

hardware_interface::CallbackReturn FairinoHardwareInterface::on_activate(
  const rclcpp_lifecycle::State&)
{
  _control_mode = 0;
  _ptr_robot = std::make_unique<FRRobot>();
  if (_ptr_robot->RPC(_controller_ip.c_str()) != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("FairinoHardwareInterface"), "RPC connection failed.");
    _ptr_robot.reset();
    return hardware_interface::CallbackReturn::ERROR;
  }

  JointPos initial{};
  if (_ptr_robot->GetActualJointPosDegree(0, &initial) != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("FairinoHardwareInterface"), "Initial joint read failed.");
    _ptr_robot->CloseRPC();
    _ptr_robot.reset();
    return hardware_interface::CallbackReturn::ERROR;
  }
  if (_ptr_robot->SetRobotRealtimeStateSamplePeriod(_realtime_state_period_ms) != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("FairinoHardwareInterface"),
      "Set realtime feedback period failed.");
    _ptr_robot->CloseRPC();
    _ptr_robot.reset();
    return hardware_interface::CallbackReturn::ERROR;
  }
  int feedback_period_ms = 0;
  if (_ptr_robot->GetRobotRealtimeStateSamplePeriod(feedback_period_ms) != 0 ||
      feedback_period_ms != _realtime_state_period_ms) {
    RCLCPP_ERROR(rclcpp::get_logger("FairinoHardwareInterface"),
      "Realtime feedback period is %d ms; expected %d ms.",
      feedback_period_ms, _realtime_state_period_ms);
    _ptr_robot->CloseRPC();
    _ptr_robot.reset();
    return hardware_interface::CallbackReturn::ERROR;
  }

  for (size_t index = 0; index < 6; ++index) {
    const double value = initial.jPos[index] * M_PI / 180.0;
    _jnt_position_command[index] = value;
    _jnt_position_state[index] = value;
    _jnt_velocity_state[index] = 0.0;
    _latest_command[index] = value;
    _latest_state[index] = value;
    _latest_velocity[index] = 0.0;
  }
  _finger_position_command[0] = _f1_close;
  _finger_position_command[1] = _f2_close;
  _finger_position_state[0] = _f1_close;
  _finger_position_state[1] = _f2_close;
  _gripper_state = GripperState::CLOSE;
  if (_has_finger1 && _ptr_robot->SetDO(GRIPPER_DO_SINGLE_ID, GRIPPER_CLOSE_LEVEL, 0, 1) != 0) {
    RCLCPP_WARN(rclcpp::get_logger("FairinoHardwareInterface"), "Initial gripper close failed.");
  }
  if (_ptr_robot->ServoMoveStart() != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("FairinoHardwareInterface"), "ServoMoveStart failed.");
    _ptr_robot->CloseRPC();
    _ptr_robot.reset();
    return hardware_interface::CallbackReturn::ERROR;
  }

  _faulted = false;
  _servo_cycles = 0;
  _cycle_overruns = 0;
  _servo_failures = 0;
  _feedback_failures = 0;
  _speed_feedback_failures = 0;
  _last_feedback_ns = steady_now_ns();
  _last_command_ns = _last_feedback_ns.load();
  _io_running = true;
  _io_thread = std::thread(&FairinoHardwareInterface::io_loop, this);
  RCLCPP_INFO(rclcpp::get_logger("FairinoHardwareInterface"),
    "ServoJ I/O started: cmdT=%.3f ms, feedback=%d ms.",
    _servoj_cmd_t * 1000.0, _realtime_state_period_ms);
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn FairinoHardwareInterface::on_deactivate(
  const rclcpp_lifecycle::State&)
{
  _io_running = false;
  if (_io_thread.joinable()) {
    _io_thread.join();
  }
  if (!_ptr_robot) {
    return hardware_interface::CallbackReturn::SUCCESS;
  }
  _ptr_robot->StopMotion();
  const int servo_end_rc = _ptr_robot->ServoMoveEnd();
  _ptr_robot->CloseRPC();
  _ptr_robot.reset();
  return servo_end_rc == 0 ? hardware_interface::CallbackReturn::SUCCESS :
    hardware_interface::CallbackReturn::ERROR;
}

void FairinoHardwareInterface::copy_gripper_state()
{
  if (!_has_finger1) {
    return;
  }
  if (_gripper_state == GripperState::OPEN) {
    _finger_position_state[0] = _f1_open;
    _finger_position_state[1] = _f2_open;
  } else {
    _finger_position_state[0] = _f1_close;
    _finger_position_state[1] = _f2_close;
  }
}

hardware_interface::return_type FairinoHardwareInterface::read(
  const rclcpp::Time&, const rclcpp::Duration&)
{
  if (_faulted || !_io_running ||
      steady_now_ns() - _last_feedback_ns.load() >
        std::chrono::duration_cast<std::chrono::nanoseconds>(kFeedbackStaleLimit).count()) {
    return hardware_interface::return_type::ERROR;
  }
  std::lock_guard<std::mutex> lock(_io_mutex);
  std::copy(_latest_state.begin(), _latest_state.end(), _jnt_position_state);
  std::copy(_latest_velocity.begin(), _latest_velocity.end(), _jnt_velocity_state);
  copy_gripper_state();
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type FairinoHardwareInterface::write(
  const rclcpp::Time&, const rclcpp::Duration&)
{
  if (_faulted || !_io_running || _control_mode != 0 ||
      !std::all_of(std::begin(_jnt_position_command), std::end(_jnt_position_command),
        [](double value) { return std::isfinite(value); })) {
    return hardware_interface::return_type::ERROR;
  }
  std::lock_guard<std::mutex> lock(_io_mutex);
  std::copy(std::begin(_jnt_position_command), std::end(_jnt_position_command),
    _latest_command.begin());
  _last_command_ns = steady_now_ns();
  return hardware_interface::return_type::OK;
}

void FairinoHardwareInterface::latch_fault(const char* reason)
{
  bool expected = false;
  if (!_faulted.compare_exchange_strong(expected, true)) {
    return;
  }
  RCLCPP_ERROR(rclcpp::get_logger("FairinoHardwareInterface"), "Hardware fault: %s", reason);
  if (_ptr_robot) {
    _ptr_robot->StopMotion();
  }
  _io_running = false;
}

void FairinoHardwareInterface::io_loop()
{
  const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(_servoj_cmd_t));
  auto next_tick = std::chrono::steady_clock::now();
  auto last_diagnostics = next_tick;
  std::array<double, 6> previous_command{};
  std::array<double, 6> previous_state{};
  bool have_previous_state = false;
  double max_command_delta = 0.0;
  double max_state_delta = 0.0;
  double max_tracking_error = 0.0;
  double max_actual_speed = 0.0;
  uint64_t consecutive_feedback_failures = 0;
  while (_io_running) {
    next_tick += period;
    std::array<double, 6> command;
    GripperState desired_gripper = _gripper_state;
    {
      std::lock_guard<std::mutex> lock(_io_mutex);
      command = _latest_command;
      if (_has_finger1) {
        const double opening = 0.5 * (std::fabs(_finger_position_command[0]) +
          std::fabs(_finger_position_command[1]));
        if ((_gripper_state == GripperState::CLOSE || _gripper_state == GripperState::UNKNOWN) &&
            opening > GRIPPER_OPEN_THRESHOLD) {
          desired_gripper = GripperState::OPEN;
        } else if (_gripper_state == GripperState::OPEN && opening < GRIPPER_CLOSE_THRESHOLD) {
          desired_gripper = GripperState::CLOSE;
        }
      }
    }
    for (size_t index = 0; index < command.size(); ++index) {
      max_command_delta = std::max(max_command_delta, std::fabs(command[index] - previous_command[index]));
    }
    previous_command = command;

    JointPos sdk_command{};
    for (size_t index = 0; index < command.size(); ++index) {
      sdk_command.jPos[index] = command[index] * 180.0 / M_PI;
    }
    ExaxisPos external_axis{0, 0, 0, 0};
    const int servo_error = _ptr_robot->ServoJ(
      &sdk_command, &external_axis, 0, 0, _servoj_cmd_t, 0, 0);
    if (servo_error != 0) {
      RCLCPP_ERROR(
        rclcpp::get_logger("FairinoHardwareInterface"),
        "ServoJ failed: error_code=%d, cmdT=%.3f ms",
        servo_error, _servoj_cmd_t * 1000.0);
      ++_servo_failures;
      latch_fault("ServoJ failed");
      break;
    }
    ++_servo_cycles;

    JointPos feedback{};
    if (_ptr_robot->GetActualJointPosDegree(1, &feedback) == 0) {
      bool finite = true;
      std::array<double, 6> state;
      for (size_t index = 0; index < state.size(); ++index) {
        state[index] = feedback.jPos[index] * M_PI / 180.0;
        finite = finite && std::isfinite(state[index]);
      }
      if (!finite) {
        latch_fault("non-finite joint feedback");
        break;
      }
      float speed_deg[6]{};
      std::array<double, 6> velocity{};
      bool valid_speed = _ptr_robot->GetActualJointSpeedsDegree(1, speed_deg) == 0;
      if (valid_speed) {
        for (size_t index = 0; index < velocity.size(); ++index) {
          velocity[index] = speed_deg[index] * M_PI / 180.0;
          valid_speed = valid_speed && std::isfinite(velocity[index]);
          max_actual_speed = std::max(max_actual_speed, std::fabs(velocity[index]));
        }
      }
      if (!valid_speed) {
        ++_speed_feedback_failures;
      }
      for (size_t index = 0; index < state.size(); ++index) {
        if (have_previous_state) {
          max_state_delta = std::max(max_state_delta, std::fabs(state[index] - previous_state[index]));
        }
        max_tracking_error = std::max(max_tracking_error, std::fabs(command[index] - state[index]));
      }
      previous_state = state;
      have_previous_state = true;
      {
        std::lock_guard<std::mutex> lock(_io_mutex);
        _latest_state = state;
        if (valid_speed) {
          _latest_velocity = velocity;
        }
        if (_has_finger1 && desired_gripper != _gripper_state) {
          const uint8_t level = desired_gripper == GripperState::OPEN ?
            GRIPPER_OPEN_LEVEL : GRIPPER_CLOSE_LEVEL;
          if (_ptr_robot->SetDO(GRIPPER_DO_SINGLE_ID, level, 0, 1) == 0) {
            _gripper_state = desired_gripper;
          } else {
            RCLCPP_WARN(rclcpp::get_logger("FairinoHardwareInterface"), "SetDO failed.");
          }
        }
      }
      consecutive_feedback_failures = 0;
      _last_feedback_ns = steady_now_ns();
    } else if (++consecutive_feedback_failures >= 3) {
      ++_feedback_failures;
      latch_fault("joint feedback failed three consecutive times");
      break;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now >= next_tick) {
      ++_cycle_overruns;
      next_tick = now;
    }
    if (now - last_diagnostics >= kDiagnosticPeriod) {
      _servo_cycles.exchange(0);
      _cycle_overruns.exchange(0);
      // RCLCPP_INFO(rclcpp::get_logger("FairinoHardwareInterface"),
      //   "ServoJ diagnostics: rate=%.1fHz command_age=%.1fms command_delta=%.5frad "
      //   "state_delta=%.5frad actual_speed=%.4frad/s tracking_error=%.5frad "
      //   "overruns=%lu servo_errors=%lu feedback_errors=%lu speed_feedback_errors=%lu age=%.1fms",
      //   rate, (steady_now_ns() - _last_command_ns.load()) / 1.0e6,
      //   max_command_delta, max_state_delta, max_actual_speed, max_tracking_error,
      //   _cycle_overruns.exchange(0), _servo_failures.load(), _feedback_failures.load(),
      //   _speed_feedback_failures.load(), (steady_now_ns() - _last_feedback_ns.load()) / 1.0e6);
      max_command_delta = 0.0;
      max_state_delta = 0.0;
      max_tracking_error = 0.0;
      max_actual_speed = 0.0;
      last_diagnostics = now;
    }
    std::this_thread::sleep_until(next_tick);
  }
}
}  // namespace fairino_hardware

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(fairino_hardware::FairinoHardwareInterface, hardware_interface::SystemInterface)
