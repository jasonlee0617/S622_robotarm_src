"""ROS-only adapters shared by GraspNet executors."""

from geometry_msgs.msg import Pose, PoseStamped
from rclpy.duration import Duration
from rclpy.time import Time
import tf2_geometry_msgs


def pose_to_base(
    tf_buffer,
    base_frame: str,
    header,
    pose: Pose,
    *,
    default_frame: str,
    timeout_sec: float = 0.5,
) -> Pose | None:
    """Transform one capture-time GraspNet pose to the configured base frame."""
    frame = header.frame_id or default_frame
    stamped = PoseStamped()
    stamped.header.frame_id = frame
    stamped.header.stamp = header.stamp
    stamped.pose = pose
    try:
        transform = tf_buffer.lookup_transform(
            base_frame,
            frame,
            Time.from_msg(header.stamp),
            timeout=Duration(seconds=float(timeout_sec)),
        )
        return tf2_geometry_msgs.do_transform_pose_stamped(stamped, transform).pose
    except Exception:
        return None
