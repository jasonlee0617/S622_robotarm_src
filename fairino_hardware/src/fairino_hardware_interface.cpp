#include "fairino_hardware/fairino_hardware_interface.hpp"

namespace fairino_hardware{

hardware_interface::CallbackReturn FairinoHardwareInterface::on_init(const hardware_interface::HardwareInfo& sysinfo){
    //ros2_control 生命周期：初始化阶段,这里主要做硬件描述解析、接口数量检查、参数读取（这份主要是检查 joint interface）
    if (hardware_interface::SystemInterface::on_init(sysinfo) != hardware_interface::CallbackReturn::SUCCESS)//先调用父类 on_init，父类会把 sysinfo 存到基类成员并做基础检查
    {
        return hardware_interface::CallbackReturn::ERROR; //父类失败则直接失败
    }
    info_ = sysinfo;//info_是父类中定义的变量,保存硬件信息（info_ 是 SystemInterface 基类里常见的成员）。后面导出接口、遍历 joints 都靠它。
    
    // =========================
    // [MOD] 先扫描 joints，识别 finger1/finger2 是否存在
    // =========================
    _has_finger1 = false;
    _has_finger2 = false;
    for (const auto& joint : info_.joints) {
        if (joint.name == "finger1_joint") _has_finger1 = true;
        if (joint.name == "finger2_joint") _has_finger2 = true;
    }

    for (const hardware_interface::ComponentInfo& joint : info_.joints) //遍历 URDF/ros2_control 中声明的每个 joint
    {

        //指令部分,命令接口检查
        if (joint.command_interfaces.size() != 1) {//开放servoJ,要求每个关节只有 1 个 command interface
            RCLCPP_FATAL(rclcpp::get_logger("FairinoHardwareInterface"),
                        "Joint '%s' has %zu command interfaces found. 1 expected.", joint.name.c_str(),
                        joint.command_interfaces.size());
            return hardware_interface::CallbackReturn::ERROR;//如果不是 1 个，直接报错退出
        }

        if (joint.command_interfaces[0].name != hardware_interface::HW_IF_POSITION) //强制要求 command interface 名称必须是 "position"
        {
            RCLCPP_FATAL(rclcpp::get_logger("FairinoHardwareInterface"),
                   "Joint '%s' have %s command interfaces found as first command interface. '%s' expected.",
                   joint.name.c_str(), joint.command_interfaces[0].name.c_str(), hardware_interface::HW_IF_POSITION);
            return hardware_interface::CallbackReturn::ERROR;//不符合就失败
        }
        //预留未来做“扭矩直接控制”
        // if (joint.command_interfaces[1].name != hardware_interface::HW_IF_EFFORT){//预留，用于关节扭矩直接控制
        //     RCLCPP_FATAL(rclcpp::get_logger("FairinoHardwareInterface"),
        //            "Joint '%s' have %s command interfaces found as first command interface. '%s' expected.",
        //            joint.name.c_str(), joint.command_interfaces[1].name.c_str(), hardware_interface::HW_IF_EFFORT);
        //     return hardware_interface::CallbackReturn::ERROR;
        // }

        //关节状态部分,状态接口检查
        if (joint.state_interfaces.size() != 1) //要求每关节只有 1 个 state interface
        {
            RCLCPP_FATAL(rclcpp::get_logger("FairinoHardwareInterface"), "Joint '%s' has %zu state interface. 3 expected.",
                        joint.name.c_str(), joint.state_interfaces.size());
            return hardware_interface::CallbackReturn::ERROR;
        }

        if (joint.state_interfaces[0].name != hardware_interface::HW_IF_POSITION) //强制要求 state interface 名称为 "position"
        {
            RCLCPP_FATAL(rclcpp::get_logger("FairinoHardwareInterface"),
                        "Joint '%s' have %s state interface as first state interface. '%s' expected.", joint.name.c_str(),
                        joint.state_interfaces[0].name.c_str(), hardware_interface::HW_IF_POSITION);
            return hardware_interface::CallbackReturn::ERROR;//不匹配就失败
        }
        //未来可能扩展更多状态接口
        // if (joint.state_interfaces[1].name != hardware_interface::HW_IF_VELOCITY) {
        //     RCLCPP_FATAL(rclcpp::get_logger("FairinoHardwareInterface"),
        //                 "Joint '%s' have %s state interface as second state interface. '%s' expected.", joint.name.c_str(),
        //                 joint.state_interfaces[1].name.c_str(), hardware_interface::HW_IF_VELOCITY);
        //     return hardware_interface::CallbackReturn::ERROR;
        // }

        // if (joint.state_interfaces[2].name != hardware_interface::HW_IF_EFFORT) {
        //     RCLCPP_FATAL(rclcpp::get_logger("FairinoHardwareInterface"),
        //                 "Joint '%s' have %s state interface as third state interface. '%s' expected.", joint.name.c_str(),
        //                 joint.state_interfaces[2].name.c_str(), hardware_interface::HW_IF_EFFORT);
        //     return hardware_interface::CallbackReturn::ERROR;
        // }

    }
    // =========================
    // [MOD]强制要求 finger1/finger2 必须成对出现，可以加检查
    // =========================
    if (_has_finger1 != _has_finger2) {
        RCLCPP_FATAL(rclcpp::get_logger("FairinoHardwareInterface"),
                     "URDF has only one finger joint. finger1=%d finger2=%d. They should be both present for scheme A.",
                     (int)_has_finger1, (int)_has_finger2);
        return hardware_interface::CallbackReturn::ERROR;
    }

    return hardware_interface::CallbackReturn::SUCCESS;//所有 joint 检查通过，初始化成功
}//end on_init


//把“状态缓冲区地址”暴露给 ros2_control
std::vector<hardware_interface::StateInterface> FairinoHardwareInterface::export_state_interfaces()//ros2_control 会调用它拿到“状态接口列表”
{
  std::vector<hardware_interface::StateInterface> state_interfaces;//创建要返回的容器

  //导出关节相关的状态接口(位置，速度，扭矩)
//   for (size_t i = 0; i < info_.joints.size(); ++i)//对每个 joint 导出一个 state interface
//   {
//     //joint 名称 = info_.joints[i].name,接口名 = "position",数据地址 = _jnt_position_state[i],controller/MoveIt 读取状态时，本质就是读这块内存
//     state_interfaces.emplace_back(hardware_interface::StateInterface(
//         info_.joints[i].name, hardware_interface::HW_IF_POSITION, &_jnt_position_state[i]));

//     // state_interfaces.emplace_back(hardware_interface::StateInterface(
//     //     info_.joints[i].name, hardware_interface::HW_IF_VELOCITY, &_jnt_velocity_state.at(i)));

//     // state_interfaces.emplace_back(hardware_interface::StateInterface(
//     //     info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &_jnt_torque_state.at(i)));
//   }
  for (const auto& joint : info_.joints)
  {
    if (joint.name == "j1") {
      state_interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, &_jnt_position_state[0]);
    } else if (joint.name == "j2") {
      state_interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, &_jnt_position_state[1]);
    } else if (joint.name == "j3") {
      state_interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, &_jnt_position_state[2]);
    } else if (joint.name == "j4") {
      state_interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, &_jnt_position_state[3]);
    } else if (joint.name == "j5") {
      state_interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, &_jnt_position_state[4]);
    } else if (joint.name == "j6") {
      state_interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, &_jnt_position_state[5]);
    } else if (joint.name == "finger1_joint") { // [MOD]
      state_interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, &_finger_position_state[0]);
    } else if (joint.name == "finger2_joint") { // [MOD]
      state_interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, &_finger_position_state[1]);
    } else {
      // [MOD] 未识别的joint直接报错
      RCLCPP_FATAL(rclcpp::get_logger("FairinoHardwareInterface"),
                   "Unknown joint name '%s' in ros2_control. Please check URDF joints list.",
                   joint.name.c_str());
    }
  }

  //导出
  return state_interfaces;//返回接口列表
}

//把“指令缓冲区地址”暴露给 ros2_control
std::vector<hardware_interface::CommandInterface> FairinoHardwareInterface::export_command_interfaces()//ros2_control 会拿到“命令接口列表”
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;//创建容器
//   for (size_t i = 0; i < info_.joints.size(); ++i) //遍历 joints
//   {
//     //controller 写指令时写的就是 _jnt_position_command[i]
//     command_interfaces.emplace_back(hardware_interface::CommandInterface(
//         info_.joints[i].name, hardware_interface::HW_IF_POSITION, &_jnt_position_command[i]));
//     //预留扭矩控制接口
// //     command_interfaces.emplace_back(hardware_interface::CommandInterface(//预留的扭矩控制接口
// //         info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &_jnt_torque_command.at(i)));
//   }
  for (const auto& joint : info_.joints)
  {
    if (joint.name == "j1") {
      command_interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, &_jnt_position_command[0]);
    } else if (joint.name == "j2") {
      command_interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, &_jnt_position_command[1]);
    } else if (joint.name == "j3") {
      command_interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, &_jnt_position_command[2]);
    } else if (joint.name == "j4") {
      command_interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, &_jnt_position_command[3]);
    } else if (joint.name == "j5") {
      command_interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, &_jnt_position_command[4]);
    } else if (joint.name == "j6") {
      command_interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, &_jnt_position_command[5]);
    } else if (joint.name == "finger1_joint") { // [MOD]
      command_interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, &_finger_position_command[0]);
    } else if (joint.name == "finger2_joint") { // [MOD]
      command_interfaces.emplace_back(joint.name, hardware_interface::HW_IF_POSITION, &_finger_position_command[1]);
    } else {
      RCLCPP_FATAL(rclcpp::get_logger("FairinoHardwareInterface"),
                   "Unknown joint name '%s' in ros2_control. Please check URDF joints list.",
                   joint.name.c_str());
    }
  }

  return command_interfaces;//返回接口列表
}


//启动硬件（连接 SDK，读取初始状态，避免“上电跳变”）
hardware_interface::CallbackReturn FairinoHardwareInterface::on_activate(const rclcpp_lifecycle::State& previous_state)//生命周期：激活硬件,一般在 controller 启动前调用
{
    using namespace std::chrono_literals;//允许写 200ms 这种字面量
    RCLCPP_INFO(rclcpp::get_logger("FairinoHardwareInterface"), "Starting ...please wait...");//提示启动中
    //做变量的初始化工作
    _ptr_robot = std::make_unique<FRRobot>();//创建机器人实例,创建厂家 SDK 的机器人对象
    for(int i=0;i<6;i++){//初始化变量
        _jnt_position_command[i] = 0;//初始化 command/state 缓冲区为 0,这只是初始化值，真正安全关键在后面“读取反馈并同步到 command”
        _jnt_velocity_command[i] = 0;
        _jnt_torque_command[i] = 0;
        _jnt_position_state[i] = 0;
        _jnt_velocity_state[i] = 0;
        _jnt_torque_state[i] = 0;
    }
    // =========================
    // [MOD] 初始化 finger 缓冲区
    // =========================
    _finger_position_command[0] = _f1_close;
    _finger_position_command[1] = _f2_close;
    _finger_position_state[0]   = _f1_close;
    _finger_position_state[1]   = _f2_close;
    _gripper_state = GripperState::CLOSE;

    _control_mode = 0;//默认是位置控制,0-位置控制，1-扭矩控制 2-速度控制
    errno_t returncode = _ptr_robot->RPC(_controller_ip.c_str());//建立xmlrpc连接,用 SDK 的 RPC 接口连接控制器（XML-RPC）
    rclcpp::sleep_for(200ms);//等待一段时间让控制器的rpc连接建立完毕,等待连接建立
    if(returncode != 0){
        RCLCPP_INFO(rclcpp::get_logger("FairinoHardwareInterface"), "机械臂SDK连接失败！请检查端口时候被占用");
        return hardware_interface::CallbackReturn::ERROR;//连接失败则报错并返回 ERROR
    }else{
        RCLCPP_INFO(rclcpp::get_logger("FairinoHardwareInterface"), "机械臂SDK连接成功！");//提示 SDK 连接成功
    }
    //做第一步的工作，读取当前状态数据
    JointPos jntpos;//厂家 SDK 的关节位置结构体
    returncode = _ptr_robot->GetActualJointPosDegree(0,&jntpos);
    /*
    获取反馈位置后同步到指令位置以维持当前状态，如果发现读取失败，那么就无法激活插件，
    因为错误的反馈位置会导致初始指令位置下发出现严重偏差导致事故
    */
    if(returncode == 0)//成功读取反馈
    {
        for(int j=0;j<6;j++)
        {
            _jnt_position_command[j] = jntpos.jPos[j]/180.0*M_PI;//把“度”转“弧度”，并把command 初始化为当前实际角度
        }

        // =========================
        // [MOD] 上电时把夹爪DO也置到一个“已知状态”
        // 避免你上电后 valve 状态不确定
        // =========================
        if (_has_finger1 && _has_finger2) {
            _ptr_robot->SetDO(GRIPPER_DO_SINGLE_ID, GRIPPER_CLOSE_LEVEL, 0, 1);
            _gripper_state = GripperState::CLOSE;
        }

        RCLCPP_INFO(rclcpp::get_logger("FairinoHardwareInterface"),"初始指令位置: %f,%f,%f,%f,%f,%f",_jnt_position_command[0],\
        _jnt_position_command[1],_jnt_position_command[2],_jnt_position_command[3],_jnt_position_command[4],_jnt_position_command[5]);    
        RCLCPP_INFO(rclcpp::get_logger("FairinoHardwareInterface"), "机械臂硬件启动成功!");//激活成功
        return hardware_interface::CallbackReturn::SUCCESS;
    }else{
        RCLCPP_INFO(rclcpp::get_logger("FairinoHardwareInterface"), "读取初始关节角度错误，硬件无法启动！请检查通讯内容");//读取初始角度失败就不允许激活（安全设计正确）
        return hardware_interface::CallbackReturn::ERROR;
    }
    // int rc_servo = _ptr_robot->ServoMoveStart();
    // if (rc_servo != 0) {
    //     RCLCPP_ERROR(rclcpp::get_logger("FairinoHardwareInterface"),
    //                 "ServoMoveStart failed, rc=%d", rc_servo);
    //     return hardware_interface::CallbackReturn::ERROR;
    // }
    // RCLCPP_INFO(rclcpp::get_logger("FairinoHardwareInterface"),
    //             "ServoMoveStart success");

}


//停止运动并断开 SDK
hardware_interface::CallbackReturn FairinoHardwareInterface::on_deactivate(const rclcpp_lifecycle::State& previous_state)
{
    RCLCPP_INFO(rclcpp::get_logger("FairinoHardwareInterface"), "Stopping ...please wait...");//提示停止
    _ptr_robot->StopMotion();//停止机器人
    _ptr_robot->CloseRPC();//销毁实例，连接断开
    _ptr_robot.release();
    RCLCPP_INFO(rclcpp::get_logger("FairinoHardwareInterface"), "System successfully stopped!");
    // int rc_end = _ptr_robot->ServoMoveEnd();
    // if (rc_end != 0) {
    //     RCLCPP_WARN(rclcpp::get_logger("FairinoHardwareInterface"),
    //                 "ServoMoveEnd failed, rc=%d", rc_end);
    // }

    return hardware_interface::CallbackReturn::SUCCESS;//停止完成
}


//从硬件读取状态，写入 _jnt_position_state[]
hardware_interface::return_type FairinoHardwareInterface::read(const rclcpp::Time& time,const rclcpp::Duration& period)//控制循环中被 ros2_control 周期调用（与 controller 更新频率一致）
{//从RTDE反馈数据中获取所需的位置，速度和扭矩信息
    JointPos state_data;//存放读取的关节角（度）
    error_t returncode = _ptr_robot->GetActualJointPosDegree(1,&state_data);//从 SDK 读取当前关节角（度）
    if(returncode == 0)//成功读取
    {
        for(int i=0;i<6;i++)
        {
            _jnt_position_state[i] = state_data.jPos[i]/180.0*M_PI;//注意单位转换，moveit统一用弧度,度→弧度，写到状态缓冲区。MoveIt/controller 就靠它感知当前姿态
            //_jnt_torque_state[i] = state_data.jt_cur_tor[i];//注意单位转换
        }
    }else{
        hardware_interface::return_type::ERROR;
    }
    //RCLCPP_INFO(rclcpp::get_logger("FairinoHardwareInterface"), "System successfully read: %f,%f,%f,%f,%f,%f",_jnt_position_state[0],\
    _jnt_position_state[1],_jnt_position_state[2],_jnt_position_state[3],_jnt_position_state[4],_jnt_position_state[5]);

    // =========================
    // [MOD] finger 状态回填（用“最近一次状态”虚拟反馈）
    // MoveIt/控制器会看这个 state 判断是否到位
    // =========================
    if (_has_finger1 && _has_finger2) {
        if (_gripper_state == GripperState::OPEN) {
            _finger_position_state[0] = _f1_open;
            _finger_position_state[1] = _f2_open;
        } else if (_gripper_state == GripperState::CLOSE) {
            _finger_position_state[0] = _f1_close;
            _finger_position_state[1] = _f2_close;
        } else {
            // UNKNOWN：先回填命令值，避免突变
            _finger_position_state[0] = _finger_position_command[0];
            _finger_position_state[1] = _finger_position_command[1];
        }
    }

  return hardware_interface::return_type::OK;//向 ros2_control 表示读取成功

}

//把 _jnt_position_command[] 下发给硬件（ServoJ）
hardware_interface::return_type FairinoHardwareInterface::write(const rclcpp::Time& time,const rclcpp::Duration& period)//控制循环中被周期调用,controller 每周期更新 command 缓冲区，这里把它发送给机械臂
{
    if(_control_mode == 0)
    {//位置控制模式
        if (std::any_of(&_jnt_position_command[0], &_jnt_position_command[5],\
            [](double c) { return not std::isfinite(c); })) {
            return hardware_interface::return_type::ERROR;//如果发现 NaN/inf，返回错误，不下发指令
        }
        JointPos cmd;//构造要发送到 SDK 的关节角结构体（单位度）
        ExaxisPos extcmd{0,0,0,0};//外部轴命令（4 轴）初始化为 0
        for(auto j=0;j<6;j++){
            cmd.jPos[j] = _jnt_position_command[j]/M_PI*180; //注意单位转换,弧度→度（与 read 相反），因为 SDK 接口要度
        }
        //RCLCPP_INFO(rclcpp::get_logger("FairinoHardwareInterface"), "ServoJ下发位置:%f,%f,%f,%f,%f,%f",\
            cmd.jPos[0],cmd.jPos[1],cmd.jPos[2],cmd.jPos[3],cmd.jPos[4],cmd.jPos[5]);
        // int returncode = _ptr_robot->ServoJ(&cmd,&extcmd,0,0,0.0016,0,0);//把关节目标以 ServoJ 方式发送给控制器
        int returncode = _ptr_robot->ServoJ(&cmd,&extcmd,0,0,0.0016,0,0);//把关节目标以 ServoJ 方式发送给控制器
        if(returncode != 0)
        {
            RCLCPP_INFO(rclcpp::get_logger("FairinoHardwareInterface"), "ServoJ指令下发错误,错误码:%d",returncode);
        }
        // =========================
        // [MOD] 夹爪：把 finger position command -> SetDO
        // 核心：position是“语义接口”，真实硬件是DO两态，所以在这里做适配
        // =========================
        if (_has_finger1 && _has_finger2) 
        {
            const double f1 = _finger_position_command[0];
            const double f2 = _finger_position_command[1];

            // opening：用绝对值估计“张开量”
            const double opening = 0.5 * (std::fabs(f1) + std::fabs(f2));

            // 带滞回判断，避免轨迹插值抖动导致DO频繁翻转
            auto target = _gripper_state;

            if (_gripper_state == GripperState::CLOSE || _gripper_state == GripperState::UNKNOWN) {
                if (opening > GRIPPER_OPEN_THRESHOLD) target = GripperState::OPEN;
            } else if (_gripper_state == GripperState::OPEN) {
                if (opening < GRIPPER_CLOSE_THRESHOLD) target = GripperState::CLOSE;
            }

            if (target != _gripper_state) {
                const uint8_t level = (target == GripperState::OPEN) ? GRIPPER_OPEN_LEVEL : GRIPPER_CLOSE_LEVEL;

                // 这里就进入你要的链路：hardware_interface -> SetDO -> 控制器IO -> 电磁阀 -> 气缸动作
                const int io_rc = _ptr_robot->SetDO(GRIPPER_DO_SINGLE_ID, level, 0, 1);

                if (io_rc != 0) {
                    RCLCPP_WARN(rclcpp::get_logger("FairinoHardwareInterface"),
                                "SetDO failed. do_id=%d level=%d rc=%d",
                                GRIPPER_DO_SINGLE_ID, (int)level, io_rc);
                } else {
                    _gripper_state = target;
                }
            }
        }

    }else if(_control_mode == 1){//扭矩控制模式,预留扭矩模式
        if (std::any_of(&_jnt_torque_command[0], &_jnt_torque_command[5],\
            [](double c) { return not std::isfinite(c); })) {
            return hardware_interface::return_type::ERROR;
        }
        //_ptr_robot->write(_jnt_torque_command);//注意单位转换
    }else{
        RCLCPP_INFO(rclcpp::get_logger("FairinoHardwareInterface"), "指令发送错误:未识别当前所处控制模式");
        return hardware_interface::return_type::ERROR;
    }
 
    return hardware_interface::return_type::OK;
}


}//end namesapce

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(fairino_hardware::FairinoHardwareInterface, hardware_interface::SystemInterface)