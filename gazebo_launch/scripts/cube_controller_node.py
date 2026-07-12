#!/usr/bin/env python3
import sys
import select
import termios
import tty
import threading

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Bool


class CubeVelocityKeyboardNode(Node):
    def __init__(self):
        super().__init__('cube_velocity_keyboard_node')
        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter('model_name', 'cube_model')
        self.declare_parameter('cmd_topic', '/model/cube_model/cmd_vel')
        self.declare_parameter('cmd_internal_topic', '/cube_truth/cmd_vel_command_internal')

        # 运动轨迹类型: 'circle' 或 'rectangle'
        self.declare_parameter('trajectory_type', 'circle')  # 默认矩形轨迹

        # 圆周运动参数：机体前向线速度 + 偏航角速度
        self.declare_parameter('linear_x', 0.05)      # m/s
        self.declare_parameter('linear_y', 0.0)       # m/s
        self.declare_parameter('angular_z', 0.5)     # rad/s

        # 矩形运动参数
        self.declare_parameter('rect_length_x', 0.2)  # 矩形 X 方向长度 (m)
        self.declare_parameter('rect_length_y', 0.2)  # 矩形 Y 方向长度 (m)
        self.declare_parameter('rect_speed', 0.05)    # 矩形运动速度 (m/s)

        # 运动总时长（秒）
        self.declare_parameter('max_motion_time', 30.0)

        # 是否启动后自动开始运动
        self.declare_parameter('auto_start', False)

        # /clock 稳定后再允许控制
        self.declare_parameter('startup_hold_sec', 0.005)

        # 命令重发频率
        self.declare_parameter('publish_rate', 30.0)

        # 提取参数
        self.model_name = str(self.get_parameter('model_name').value)
        self.cmd_topic = str(self.get_parameter('cmd_topic').value)
        self.cmd_internal_topic = str(self.get_parameter('cmd_internal_topic').value)
        self.trajectory_type = str(self.get_parameter('trajectory_type').value)

        self.linear_x = float(self.get_parameter('linear_x').value)
        self.linear_y = float(self.get_parameter('linear_y').value)
        self.angular_z = float(self.get_parameter('angular_z').value)

        self.rect_length_x = float(self.get_parameter('rect_length_x').value)
        self.rect_length_y = float(self.get_parameter('rect_length_y').value)
        self.rect_speed = max(0.001, float(self.get_parameter('rect_speed').value))

        self.max_motion_time = float(self.get_parameter('max_motion_time').value)
        self.auto_start = bool(self.get_parameter('auto_start').value)
        self.startup_hold_sec = float(self.get_parameter('startup_hold_sec').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)

        # -------------------------
        # Runtime states
        # -------------------------
        self.sim_time_sec = None
        self.clock_first_seen = None

        self.motion_enabled = self.auto_start
        self.motion_paused = False
        self.motion_start_time = None
        self.total_paused_duration = 0.0
        self.pause_begin_time = None

        self.shutdown_requested = False
        self.keyboard_thread = None

        # -------------------------
        # ROS I/O
        # -------------------------
        self.create_subscription(Clock, '/clock', self.clock_cb, 10)
        self.create_subscription(Bool, "/cube_auto_start", self.auto_start_cb, 60)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.cmd_internal_pub = self.create_publisher(TwistStamped, self.cmd_internal_topic, 10)

        timer_period = 1.0 / max(self.publish_rate, 1.0)
        self.timer = self.create_timer(timer_period, self.timer_cb)

        # -------------------------
        # Keyboard thread
        # -------------------------
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.keyboard_thread.start()

        self.print_instructions()

        self.get_logger().info(
            f'Cube Controller started. Mode: {self.trajectory_type.upper()}'
        )

    # -------------------------
    # Basic printing
    # -------------------------
    def print_instructions(self):
        print("\n================ Cube Keyboard Control ================")
        print("s : start motion      (开始运动)")
        print("p : pause motion      (暂停运动)")
        print("r : resume motion     (恢复运动)")
        print("x : stop & reset      (停止并重置时间)")
        print("1 : switch to CIRCLE  (切换至圆形轨迹)")
        print("2 : switch to RECT    (切换至矩形轨迹)")
        print("q : stop and quit     (退出程序)")
        print("=======================================================\n")

    # -------------------------
    # Clock callback
    # -------------------------
    def clock_cb(self, msg: Clock):
        t = msg.clock.sec + msg.clock.nanosec * 1e-9
        self.sim_time_sec = t
        if self.clock_first_seen is None:
            self.clock_first_seen = t

    # -------------------------
    # auto_start callback
    # -------------------------       
    def auto_start_cb(self, msg: Bool):
        enable = bool(msg.data)
        if enable:
            self.get_logger().info("Received /cube_auto_start = True. Starting motion.")
            self.start_motion()
        else:
            self.get_logger().info("Received /cube_auto_start = False. Stopping motion.")
            self.stop_motion(reset_timer=False)

    # -------------------------
    # Keyboard loop
    # -------------------------
    def keyboard_loop(self):
        if not sys.stdin.isatty():
            self.get_logger().warn("stdin is not a TTY. Keyboard control unavailable.")
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)
            while rclpy.ok() and (not self.shutdown_requested):
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    self.handle_key(ch)
        except Exception as e:
            self.get_logger().warn(f"Keyboard loop exception: {e}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def handle_key(self, ch: str):
        ch = ch.lower()

        if ch == 's':
            self.start_motion()
        elif ch == 'p':
            self.pause_motion()
        elif ch == 'r':
            self.resume_motion()
        elif ch == 'x':
            self.stop_motion(reset_timer=True)
        elif ch == '1':
            self.trajectory_type = 'circle'
            self.get_logger().info(">> Switched to CIRCLE trajectory <<")
        elif ch == '2':
            self.trajectory_type = 'rectangle'
            self.get_logger().info(">> Switched to RECTANGLE trajectory <<")
        elif ch == 'q':
            self.get_logger().info("Quit requested by keyboard.")
            self.stop_motion(reset_timer=True)
            self.shutdown_requested = True

    # -------------------------
    # Motion state machine
    # -------------------------
    def can_control_now(self):
        if self.sim_time_sec is None or self.clock_first_seen is None:
            return False
        if (self.sim_time_sec - self.clock_first_seen) < self.startup_hold_sec:
            return False
        return True

    def start_motion(self):
        if not self.can_control_now():
            self.get_logger().warn("Clock not ready, cannot start yet.")
            return

        self.motion_enabled = True
        self.motion_paused = False
        self.total_paused_duration = 0.0
        self.pause_begin_time = None
        self.motion_start_time = self.sim_time_sec
        self.get_logger().info("Motion STARTED.")

    def pause_motion(self):
        if not self.motion_enabled or self.motion_paused:
            return
        self.motion_paused = True
        self.pause_begin_time = self.sim_time_sec
        self.publish_cmd_vel(0.0, 0.0, 0.0)
        self.get_logger().info("Motion PAUSED.")

    def resume_motion(self):
        if not self.motion_enabled or not self.motion_paused:
            return
        if self.pause_begin_time is not None and self.sim_time_sec is not None:
            self.total_paused_duration += (self.sim_time_sec - self.pause_begin_time)
        self.pause_begin_time = None
        self.motion_paused = False
        self.get_logger().info("Motion RESUMED.")

    def stop_motion(self, reset_timer=True):
        self.motion_enabled = False
        self.motion_paused = False
        self.pause_begin_time = None
        self.publish_cmd_vel(0.0, 0.0, 0.0)

        if reset_timer:
            self.motion_start_time = None
            self.total_paused_duration = 0.0

        self.get_logger().info("Motion STOPPED.")

    def get_elapsed_motion_time(self):
        if self.motion_start_time is None or self.sim_time_sec is None:
            return 0.0
        elapsed = self.sim_time_sec - self.motion_start_time - self.total_paused_duration
        if self.motion_paused and self.pause_begin_time is not None:
            elapsed -= (self.sim_time_sec - self.pause_begin_time)
        return max(0.0, elapsed)

    # -------------------------
    # Publish cmd_vel via ROS bridge
    # -------------------------
    def publish_cmd_vel(self, vx: float, vy: float, wz: float):
        cmd_msg = Twist()
        cmd_msg.linear.x = float(vx)
        cmd_msg.linear.y = float(vy)
        cmd_msg.linear.z = 0.0
        cmd_msg.angular.x = 0.0
        cmd_msg.angular.y = 0.0
        cmd_msg.angular.z = float(wz)
        self.cmd_pub.publish(cmd_msg)

        internal_msg = TwistStamped()
        internal_msg.header.stamp = self.get_clock().now().to_msg()
        internal_msg.header.frame_id = f"{self.model_name}/body"
        internal_msg.twist = cmd_msg
        self.cmd_internal_pub.publish(internal_msg)

    # -------------------------
    # Timer loop: Trajectory Generation
    # -------------------------
    def timer_cb(self):
        if not self.can_control_now():
            return

        if self.shutdown_requested or not self.motion_enabled or self.motion_paused:
            self.publish_cmd_vel(0.0, 0.0, 0.0)
            if self.shutdown_requested:
                rclpy.shutdown()
            return

        elapsed = self.get_elapsed_motion_time()

        if elapsed >= self.max_motion_time:
            self.get_logger().info("Reached max motion time. Auto stop.")
            self.stop_motion(reset_timer=False)
            return

        # ------------------------------------------------
        # 轨迹算法核心
        # ------------------------------------------------
        if self.trajectory_type == 'circle':
            # 圆周运动 (固定线速度与角速度)
            self.publish_cmd_vel(self.linear_x, self.linear_y, self.angular_z)

        elif self.trajectory_type == 'rectangle':
            # 矩形运动 (基于时间的开环分段平移)
            # 计算跑完每条边需要的时间
            t_x = self.rect_length_x / self.rect_speed
            t_y = self.rect_length_y / self.rect_speed
            
            # 一个完整矩形的周期时间
            T_cycle = 2 * (t_x + t_y)
            
            # 当前时间位于周期内的什么位置
            t_current = elapsed % T_cycle

            vx, vy, wz = 0.0, 0.0, 0.0

            if t_current < t_x:
                # 边 1: 往 +X 方向走
                vx = self.rect_speed
                vy = 0.0
            elif t_current < (t_x + t_y):
                # 边 2: 往 +Y 方向走
                vx = 0.0
                vy = self.rect_speed
            elif t_current < (2 * t_x + t_y):
                # 边 3: 往 -X 方向走
                vx = -self.rect_speed
                vy = 0.0
            else:
                # 边 4: 往 -Y 方向走
                vx = 0.0
                vy = -self.rect_speed

            self.publish_cmd_vel(vx, vy, wz)


def main(args=None):
    rclpy.init(args=args)
    node = CubeVelocityKeyboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_cmd_vel(0.0, 0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
