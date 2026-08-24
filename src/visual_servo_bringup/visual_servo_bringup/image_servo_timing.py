"""ROS-independent image timestamp handling for IBVS."""


def feature_timestamp_ns(stamp, fallback_ns: int) -> int:
    """Use the image timestamp unless a driver supplied an unset stamp."""
    timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return timestamp_ns if timestamp_ns > 0 else fallback_ns
