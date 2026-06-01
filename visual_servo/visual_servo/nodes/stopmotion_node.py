#!/usr/bin/env python3
import sys, termios, tty, select
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

class KeyboardAbort(Node):
    def __init__(self):
        super().__init__("keyboard_abort_space")
        # 创建发布者：发布 /manual_abort (Bool)
        self.pub = self.create_publisher(Bool, "/manual_abort", 10)

        # ========= 终端键盘读取准备 =========
        # sys.stdin.fileno()：获取标准输入的文件描述符
        # termios.tcgetattr：读取当前终端属性（用于后续恢复）
        # tty.setcbreak：设置成 cbreak 模式（按键无需回车即可读到）
        #stdin 必须是 TTY，否则 tcgetattr 会失败
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)    # 设置为“按键立即可读”的模式

        # 定时器：每 0.05s 检查一次键盘输入（20Hz）
        self.timer = self.create_timer(0.05, self.tick)
        self.get_logger().info("Press <SPACE> to ABORT (stop & go home).")

    def tick(self):
        """
        非阻塞检查是否有按键输入：
        - select.select([stdin], [], [], 0.0)：timeout=0 表示立刻返回
        - 如果 stdin 可读，读一个字符
        - 如果是空格 ' '，就发布 /manual_abort = True
        """
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not r:  # 没按键，直接返回，不阻塞
            return
        ch = sys.stdin.read(1)
        # 空格触发
        if ch == " ":
            self.pub.publish(Bool(data=True))
            self.get_logger().warn("ABORT sent: /manual_abort = true")

    def destroy_node(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        super().destroy_node()

def main():
    rclpy.init()
    n = KeyboardAbort()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
