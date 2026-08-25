"""ROS-independent image timestamp handling for IBVS diagnostics."""


def source_timestamp_ns(stamp) -> int | None:
    """Return a driver timestamp when present; it is not a watchdog clock."""
    timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return timestamp_ns if timestamp_ns > 0 else None
