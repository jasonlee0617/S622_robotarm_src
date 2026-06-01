#ifndef _FR_HARDWARE_INTERFACE_
#define _FR_HARDWARE_INTERFACE_

#include "rclcpp/rclcpp.hpp"  //引入 ROS2 C++ 客户端库，提供 Node、logger、Time、Duration 等核心能力。
#include "rclcpp/macros.hpp"
#include <hardware_interface/hardware_info.hpp> //ros2_control 的硬件描述信息结构（从 URDF/xacro 解析出来的 joints/interfaces 参数会在这里）
#include <hardware_interface/system_interface.hpp> //ros2_control 的 SystemInterface 基类：你这个硬件插件就是继承它来实现 on_init/read/write/...
#include <hardware_interface/types/hardware_interface_return_values.hpp> //hardware_interface::return_type、CallbackReturn 等返回值类型定义
#include "hardware_interface/types/hardware_interface_type_values.hpp" //HW_IF_POSITION / VELOCITY / EFFORT 等接口名常量
#include "visibility_control.h" //通常用于导出/隐藏符号（Windows/Linux 下的 dll/so 可见性控制），给 pluginlib 用
#include <vector> //使用 std::vector
#include "libfairino/include/robot.h" //引入厂家 SDK 的头文件（FRRobot 类就在这里）


#define CONTROLLER_IP_ADDRESS "192.168.58.2" //定义控制器默认 IP 地址字符串常量。.cpp 里会用它做 RPC 连接

#define GRIPPER_DO_SINGLE_ID  0    // [MOD] DO0 控制电磁阀线圈
#define GRIPPER_OPEN_LEVEL    0    // [MOD] DO=0 表示张开
#define GRIPPER_CLOSE_LEVEL   1    // [MOD] DO=1 表示关闭
#define GRIPPER_OPEN_THRESHOLD   0.010  // opening > 0.01 -> OPEN
#define GRIPPER_CLOSE_THRESHOLD  0.005  // opening < 0.005 -> CLOSE（滞回，防抖）

namespace fairino_hardware
{

class FairinoHardwareInterface: public hardware_interface::SystemInterface{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(FairinoHardwareInterface)

  FAIRINO_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo& info) override;

  //FAIRINO_HARDWARE_PUBLIC
  //hardware_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State &) override;

  FAIRINO_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  
  FAIRINO_HARDWARE_PUBLIC
  hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;
  
  FAIRINO_HARDWARE_PUBLIC
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  
  FAIRINO_HARDWARE_PUBLIC
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  
  // hardware_interface::return_type prepare_command_mode_switch(
  //   const std::vector<std::string> & start_interfaces,
  //   const std::vector<std::string> & stop_interfaces) override;
  // hardware_interface::return_type perform_command_mode_switch(
  //   const std::vector<std::string>& start_interfaces,
  //   const std::vector<std::string>& stop_interfaces) override;

  FAIRINO_HARDWARE_PUBLIC
  hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
  
  FAIRINO_HARDWARE_PUBLIC
  hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;
  
private:
  double _jnt_position_command[6]; //6 个关节的“位置指令缓冲区”。controller（如 joint_trajectory_controller）写入这里
  double _jnt_velocity_command[6]; //预留：速度指令缓冲区
  double _jnt_torque_command[6]; //预留：力矩指令缓冲区
  double _jnt_position_state[6]; //6 个关节的“反馈位置状态缓冲区”。read() 会写入这里，供 controller/MoveIt 读取
  double _jnt_velocity_state[6]; //预留：反馈速度状态缓冲区
  double _jnt_torque_state[6]; //预留：反馈力矩状态缓冲区

  double _finger_position_command[2]{0.0, 0.0}; // [MOD]
  double _finger_position_state[2]{0.0, 0.0};   // [MOD]

  // [MOD] 标记是否在URDF里存在 finger joint
  bool _has_finger1{false}; // [MOD]
  bool _has_finger2{false}; // [MOD]

  // [MOD] 夹爪状态机（用于回填state、做滞回）
  enum class GripperState { UNKNOWN, OPEN, CLOSE }; // [MOD]
  GripperState _gripper_state{GripperState::UNKNOWN}; // [MOD]

  int _control_mode; //控制模式： 0-位置控制，1-扭矩控制 2-速度控制
  std::string _controller_ip = CONTROLLER_IP_ADDRESS; //控制器 IP，默认用宏
  std::unique_ptr<FRRobot> _ptr_robot; //厂家 SDK 对象指针：on_activate() 创建，on_deactivate() 释放，read/write 里调用 SDK 方法

  // 给finger回填用的“名义位置”（与URDF group_state一致）
  double _f1_open{0.0305};   // [MOD]
  double _f2_open{-0.0305};   // [MOD]
  double _f1_close{0.0};      // [MOD]
  double _f2_close{0.0};      // [MOD]
};

} //end namespace


#endif