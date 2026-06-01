#!/usr/bin/env python3
"""Helpers for demo obstacle model generation and spawn action creation."""

from __future__ import annotations

import os
import yaml
from launch_ros.actions import Node


def make_dynamic_box_sdf(model_name, size_xyz, mass=0.2, enable_collision=False):
    sx, sy, sz = size_xyz
    ixx = mass / 12.0 * (sy * sy + sz * sz)
    iyy = mass / 12.0 * (sx * sx + sz * sz)
    izz = mass / 12.0 * (sx * sx + sy * sy)

    collision_block = ""
    if enable_collision:
        collision_block = f"""
      <collision name="collision">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
      </collision>
"""

    return f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{model_name}">
    <static>false</static>
    <allow_auto_disable>false</allow_auto_disable>
    <link name="link">
      <gravity>false</gravity>
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx}</ixx><ixy>0.0</ixy><ixz>0.0</ixz>
          <iyy>{iyy}</iyy><iyz>0.0</iyz><izz>{izz}</izz>
        </inertia>
      </inertial>
{collision_block}
      <visual name="visual">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <material><ambient>1.0 0.0 0.0 0.8</ambient><diffuse>1.0 0.0 0.0 0.8</diffuse></material>
      </visual>
    </link>
    <plugin filename="ignition-gazebo6-velocity-control-system" name="ignition::gazebo::systems::VelocityControl">
      <topic>/model/{model_name}/cmd_vel</topic>
    </plugin>
  </model>
</sdf>
"""


def load_spawn_config(path):
    with open(path, "r", encoding="utf-8") as f:
        root = yaml.safe_load(f)
    if "spawn" in root:
        return root["spawn"]
    return root["demo_spawn"]


def build_spawn_actions(pkg_share_dir, spawn_cfg):
    actions = []
    for st in spawn_cfg.get("static", []):
        if not st.get("enabled", True):
            continue
        urdf_path = os.path.join(pkg_share_dir, "config", st["file"])
        with open(urdf_path, "r", encoding="utf-8") as f:
            urdf_txt = f.read()
        pose = st["pose"]
        actions.append(Node(
            package="ros_gz_sim", executable="create",
            arguments=["-string", urdf_txt, "-x", str(pose["x"]), "-y", str(pose["y"]), "-z", str(pose["z"]), "-name", st["name"]],
            output="screen"
        ))

    for dyn in spawn_cfg.get("dynamic", []):
        if not dyn.get("enabled", True):
            continue
        pose = dyn["pose"]
        sdf = make_dynamic_box_sdf(dyn["name"], tuple(dyn["size"]), mass=dyn.get("mass", 0.2), enable_collision=dyn.get("enable_collision", False))
        actions.append(Node(
            package="ros_gz_sim", executable="create",
            arguments=["-string", sdf, "-x", str(pose["x"]), "-y", str(pose["y"]), "-z", str(pose["z"]), "-name", dyn["name"]],
            output="screen"
        ))
    return actions
