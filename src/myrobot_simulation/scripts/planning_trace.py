"""Trace publication helpers kept separate from planning and benchmark logic."""

from geometry_msgs.msg import Point


def append_trace_point(marker, xyz, max_points):
    point = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
    marker.points.append(point)
    if len(marker.points) > max_points:
        marker.points = marker.points[-max_points:]
