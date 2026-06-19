#!/usr/bin/env python3
import math
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yaml
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class SceneObstacle:
    name: str
    position: Tuple[float, float, float]
    shape: str = "box"
    size: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    asset: str = ""
    color: Tuple[float, float, float, float] = (0.95, 0.30, 0.05, 0.60)
    radius: Optional[float] = None
    height: Optional[float] = None
    use_asset_for_gazebo: bool = True

    @classmethod
    def box(cls, name, position, size, rpy_deg=None):
        return SceneLoader.make_obstacle(
            name=name,
            position=position,
            size=size,
            rpy_deg=rpy_deg,
            shape="box",
            use_asset_for_gazebo=False,
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


def pose_quat_from_rpy(rpy_deg):
    quat = R.from_euler("xyz", rpy_deg, degrees=True).as_quat()
    return tuple(float(v) for v in quat)


class SceneLoader:
    def __init__(self, scene_name, scene_config_file, logger):
        self.scene_name = scene_name
        self.scene_config_file = scene_config_file
        self.logger = logger
        self.benchmark = {}

    @staticmethod
    def _parse_float_list(value) -> List[float]:
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        text = str(value).replace(";", ",").replace(" ", ",")
        return [float(v) for v in text.split(",") if v.strip()]

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

    @classmethod
    def parse_obstacle_boxes(cls, value):
        text = str(value).strip()
        if not text:
            return []

        boxes = []
        for spec in text.split(";"):
            spec = spec.strip()
            if not spec:
                continue
            parts = [p.strip() for p in spec.split(":")]
            if len(parts) != 3:
                raise ValueError(
                    "obstacle_boxes 格式必须为 name:x,y,z:sx,sy,sz;name2:x,y,z:sx,sy,sz")
            name, position_text, size_text = parts
            if not name:
                raise ValueError("obstacle_boxes 中的 name 不能为空")
            position = tuple(cls._parse_float_list(position_text))
            size = tuple(cls._parse_float_list(size_text))
            if len(position) != 3 or len(size) != 3:
                raise ValueError("obstacle_boxes 中每个 position/size 都必须包含 3 个数值")
            boxes.append((name, position, size))
        return boxes

    @classmethod
    def make_obstacle(
        cls,
        name,
        position,
        size=None,
        rpy_deg=None,
        asset="",
        color=None,
        shape="box",
        radius=None,
        height=None,
        use_asset_for_gazebo=True,
    ):
        rpy = tuple(float(v) for v in (rpy_deg or (0.0, 0.0, 0.0)))
        rgba = tuple(float(v) for v in (color or (0.95, 0.30, 0.05, 0.60)))
        shape = str(shape or "box").strip().lower()
        if shape not in ("box", "cylinder", "sphere"):
            raise ValueError(f"Unsupported obstacle shape '{shape}'")

        if shape == "box":
            if size is None:
                raise ValueError(f"box obstacle '{name}' requires size")
            dimensions = tuple(float(v) for v in size)
            if len(dimensions) != 3:
                raise ValueError(f"box obstacle '{name}' size must contain 3 values")
            radius_value = None
            height_value = None
        elif shape == "cylinder":
            if radius is None or height is None:
                raise ValueError(f"cylinder obstacle '{name}' requires radius and height")
            radius_value = float(radius)
            height_value = float(height)
            dimensions = (2.0 * radius_value, 2.0 * radius_value, height_value)
        else:
            if radius is None:
                raise ValueError(f"sphere obstacle '{name}' requires radius")
            radius_value = float(radius)
            height_value = None
            dimensions = (2.0 * radius_value, 2.0 * radius_value, 2.0 * radius_value)

        return SceneObstacle(
            name=str(name),
            position=tuple(float(v) for v in position),
            shape=shape,
            size=dimensions,
            radius=radius_value,
            height=height_value,
            rpy_deg=rpy,
            asset=str(asset or ""),
            color=rgba,
            use_asset_for_gazebo=bool(use_asset_for_gazebo),
        )

    def obstacle_from_yaml(self, item: Dict, index: int):
        name = str(item.get("name", f"{self.scene_name}_obstacle_{index}"))
        raw_pose = item.get("pose", item.get("xyz", item.get("position")))
        if raw_pose is None:
            raise ValueError(f"scene obstacle '{name}' missing pose/position")

        pose_values = self._parse_float_list(raw_pose)
        if len(pose_values) == 3:
            position = tuple(pose_values)
            rpy_deg = (0.0, 0.0, 0.0)
        elif len(pose_values) == 6:
            position = tuple(pose_values[:3])
            rpy_deg = tuple(pose_values[3:])
        else:
            raise ValueError(f"scene obstacle '{name}' pose must contain 3 or 6 values")

        shape = str(item.get("shape", "box")).strip().lower()
        size = None
        radius = None
        height = None
        if shape == "box":
            size = tuple(self._parse_float_list(item.get("size", item.get("scale"))))
        elif shape == "cylinder":
            radius = float(item["radius"])
            height = float(item["height"])
        elif shape == "sphere":
            radius = float(item["radius"])
        else:
            raise ValueError(f"scene obstacle '{name}' has unsupported shape '{shape}'")

        color = item.get("color", (0.95, 0.30, 0.05, 0.60))
        color_values = tuple(self._parse_float_list(color))
        if len(color_values) not in (3, 4):
            raise ValueError(f"scene obstacle '{name}' color must contain 3 or 4 values")
        if len(color_values) == 3:
            color_values = (*color_values, 0.60)

        return self.make_obstacle(
            name=name,
            position=position,
            size=size,
            rpy_deg=rpy_deg,
            asset=str(item.get("asset", "")),
            color=color_values,
            shape=shape,
            radius=radius,
            height=height,
            use_asset_for_gazebo=self._as_bool(item.get("use_asset_for_gazebo", True)),
        )

    def load(self, obstacle_boxes, default_name, default_position, default_size):
        if obstacle_boxes:
            obstacles = [
                self.make_obstacle(name, position, size, use_asset_for_gazebo=False)
                for name, position, size in obstacle_boxes
            ]
            self.logger.info(f"使用 obstacle_boxes 覆盖场景障碍物: count={len(obstacles)}")
            return obstacles

        if os.path.exists(self.scene_config_file):
            with open(self.scene_config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            scenes = data.get("scenes", data)
            if self.scene_name not in scenes:
                known = ", ".join(sorted(str(k) for k in scenes.keys()))
                raise ValueError(
                    f"scene_name='{self.scene_name}' not found in {self.scene_config_file}. "
                    f"known scenes: {known}"
                )
            scene = scenes[self.scene_name] or {}
            self.benchmark = scene.get("benchmark", {}) or {}
            obstacles = [
                self.obstacle_from_yaml(item, index)
                for index, item in enumerate(scene.get("obstacles", []))
            ]
            self.logger.info(
                f"加载路径规划场景: scene={self.scene_name}, obstacles={len(obstacles)}, "
                f"config={self.scene_config_file}"
            )
            if self.benchmark:
                self.logger.info(
                    f"场景 benchmark: start_pose={self.benchmark.get('start_pose')}, "
                    f"goal_pose={self.benchmark.get('goal_pose')}"
                )
            return obstacles

        self.logger.warn(
            f"场景配置文件不存在，回退旧单障碍物参数: {self.scene_config_file}")
        return [
            self.make_obstacle(
                default_name,
                default_position,
                default_size,
                use_asset_for_gazebo=False,
            )
        ]


class PlanningSceneManager:
    def __init__(self, node, base_frame_name, publish_enabled=True, obstacle_padding_m=0.0):
        self.node = node
        self.base_frame_name = base_frame_name
        self.publish_enabled = publish_enabled
        self.obstacle_padding_m = max(0.0, float(obstacle_padding_m))
        self.demo_collision_objects = set()
        qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.publisher = node.create_publisher(PlanningScene, "/planning_scene", qos)

    def _primitive_for_obstacle(self, obstacle):
        primitive = SolidPrimitive()
        shape = obstacle.shape
        padding = self.obstacle_padding_m
        if shape == "box":
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [
                float(obstacle.size[0]) + 2.0 * padding,
                float(obstacle.size[1]) + 2.0 * padding,
                float(obstacle.size[2]) + 2.0 * padding,
            ]
        elif shape == "cylinder":
            primitive.type = SolidPrimitive.CYLINDER
            primitive.dimensions = [
                float(obstacle.height) + 2.0 * padding,
                float(obstacle.radius) + padding,
            ]
        elif shape == "sphere":
            primitive.type = SolidPrimitive.SPHERE
            primitive.dimensions = [float(obstacle.radius) + padding]
        else:
            raise ValueError(f"Unsupported obstacle shape '{shape}'")
        return primitive

    def add_obstacle(self, obstacle, frame_id=None):
        if not self.publish_enabled:
            return

        frame_id = frame_id or self.base_frame_name
        collision_object = CollisionObject()
        collision_object.header.frame_id = frame_id
        collision_object.header.stamp = self.node.get_clock().now().to_msg()
        collision_object.id = obstacle.name
        collision_object.operation = CollisionObject.ADD

        object_pose = Pose()
        object_pose.position.x = float(obstacle.position[0])
        object_pose.position.y = float(obstacle.position[1])
        object_pose.position.z = float(obstacle.position[2])
        quat = pose_quat_from_rpy(obstacle.rpy_deg)
        object_pose.orientation.x = quat[0]
        object_pose.orientation.y = quat[1]
        object_pose.orientation.z = quat[2]
        object_pose.orientation.w = quat[3]

        primitive = self._primitive_for_obstacle(obstacle)
        collision_object.primitives.append(primitive)
        collision_object.primitive_poses.append(object_pose)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(collision_object)

        time.sleep(0.5)
        self.publisher.publish(scene)
        self.demo_collision_objects.add(obstacle.name)
        self.node.get_logger().info(
            f"添加碰撞物体: {obstacle.name}, shape={obstacle.shape}, "
            f"pos={obstacle.position}, size={obstacle.size}, "
            f"padding_m={self.obstacle_padding_m:.3f}, primitive_dims={list(primitive.dimensions)}"
        )
        time.sleep(0.5)

    def remove_object(self, name):
        if not self.publish_enabled or not name:
            return

        collision_object = CollisionObject()
        collision_object.header.frame_id = self.base_frame_name
        collision_object.header.stamp = self.node.get_clock().now().to_msg()
        collision_object.id = name
        collision_object.operation = CollisionObject.REMOVE

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(collision_object)

        time.sleep(0.5)
        self.publisher.publish(scene)
        self.demo_collision_objects.discard(name)
        self.node.get_logger().info(f"移除碰撞物体: {name}")

    def clear(self, obstacles=None, extra_names=None):
        names = set(self.demo_collision_objects)
        for obstacle in obstacles or []:
            names.add(obstacle.name)
        for name in extra_names or []:
            names.add(name)

        for name in list(names):
            self.remove_object(name)
        self.demo_collision_objects.clear()


class MarkerPublisher:
    def __init__(self, node, base_frame_name, topic, publish_enabled=True):
        self.node = node
        self.base_frame_name = base_frame_name
        self.topic = topic
        self.publish_enabled = publish_enabled
        self.publisher = node.create_publisher(MarkerArray, topic, 10)

    @staticmethod
    def _marker_type_for_obstacle(obstacle):
        if obstacle.shape == "box":
            return Marker.CUBE
        if obstacle.shape == "cylinder":
            return Marker.CYLINDER
        if obstacle.shape == "sphere":
            return Marker.SPHERE
        raise ValueError(f"Unsupported obstacle shape '{obstacle.shape}'")

    def publish(self, obstacles):
        if not self.publish_enabled:
            return

        stamp = self.node.get_clock().now().to_msg()
        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.header.frame_id = self.base_frame_name
        delete_all.header.stamp = stamp
        delete_all.ns = "pathplanning_obstacles"
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for index, obstacle in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = self.base_frame_name
            marker.header.stamp = stamp
            marker.ns = "pathplanning_obstacles"
            marker.id = index
            marker.type = self._marker_type_for_obstacle(obstacle)
            marker.action = Marker.ADD
            marker.pose.position.x = obstacle.position[0]
            marker.pose.position.y = obstacle.position[1]
            marker.pose.position.z = obstacle.position[2]
            quat = pose_quat_from_rpy(obstacle.rpy_deg)
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
            marker.scale.x = obstacle.size[0]
            marker.scale.y = obstacle.size[1]
            marker.scale.z = obstacle.size[2]
            marker.color.r = obstacle.color[0]
            marker.color.g = obstacle.color[1]
            marker.color.b = obstacle.color[2]
            marker.color.a = obstacle.color[3]
            marker_array.markers.append(marker)

        self.publisher.publish(marker_array)
        self.node.get_logger().info(
            f"发布路径规划障碍物 marker: topic={self.topic}, count={len(obstacles)}")

    def clear(self):
        if not self.publish_enabled:
            return
        marker = Marker()
        marker.header.frame_id = self.base_frame_name
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "pathplanning_obstacles"
        marker.action = Marker.DELETEALL
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        self.publisher.publish(marker_array)


class GazeboSceneSpawner:
    def __init__(self, scene_name, scene_assets_dir, gazebo_world, logger):
        self.scene_name = scene_name
        self.scene_assets_dir = scene_assets_dir
        self.gazebo_world = gazebo_world
        self.logger = logger
        self.spawned_models = set()

    @staticmethod
    def _safe_name(text: str) -> str:
        return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in text)

    def model_name(self, obstacle: SceneObstacle) -> str:
        return f"{self._safe_name(self.scene_name)}_{self._safe_name(obstacle.name)}"

    def asset_path(self, obstacle: SceneObstacle) -> str:
        if not obstacle.asset:
            return ""
        if os.path.isabs(obstacle.asset):
            return obstacle.asset
        return os.path.join(self.scene_assets_dir, obstacle.asset)

    def remove_model(self, model_name, quiet=False):
        if not model_name:
            return False
        # Ignition Gazebo 6: 用 ign service 原生 transport（非 ROS2 DDS）
        # ros_gz_bridge 不桥接 /world/*/remove 服务
        cmd = [
            "ign", "service",
            "-s", f"/world/{self.gazebo_world}/remove",
            "--reqtype", "ignition.msgs.Entity",
            "--reptype", "ignition.msgs.Boolean",
            "--req", f"name: '{model_name}', type: 2",
            "--timeout", "3000",
        ]
        try:
            result = subprocess.run(
                cmd, check=False, timeout=5.0, capture_output=True, text=True)
        except Exception as exc:
            if not quiet:
                self.logger.warn(f"Gazebo 模型删除异常: {model_name}: {exc}")
            return False
        if result.returncode != 0:
            if not quiet:
                stderr = (result.stderr or result.stdout or "").strip()
                self.logger.warn(f"Gazebo 模型删除失败: {model_name}: {stderr}")
            return False
        if not quiet:
            self.logger.info(f"Gazebo 场景模型已删除: {model_name}")
        return True

    def spawn_obstacle(self, obstacle: SceneObstacle):
        model_name = self.model_name(obstacle)
        if not obstacle.use_asset_for_gazebo:
            self.logger.warn(
                f"跳过 Gazebo 场景模型 spawn，use_asset_for_gazebo=false: name={obstacle.name}")
            return False

        spawn_path = self.asset_path(obstacle)
        if not spawn_path or not os.path.exists(spawn_path):
            self.logger.warn(
                f"跳过 Gazebo 场景模型 spawn，asset 不存在: name={obstacle.name} asset={spawn_path}")
            return False

        roll, pitch, yaw = [math.radians(v) for v in obstacle.rpy_deg]
        cmd = [
            "ros2", "run", "ros_gz_sim", "create",
            "-world", self.gazebo_world,
            "-file", spawn_path,
            "-name", model_name,
            "-x", str(obstacle.position[0]),
            "-y", str(obstacle.position[1]),
            "-z", str(obstacle.position[2]),
            "-R", str(roll),
            "-P", str(pitch),
            "-Y", str(yaw),
            "-allow_renaming", "true",
        ]
        try:
            result = subprocess.run(
                cmd, check=False, timeout=8.0, capture_output=True, text=True)
        except Exception as exc:
            self.logger.warn(f"Gazebo 场景模型 spawn 失败: {model_name}: {exc}")
            return False
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            self.logger.warn(f"Gazebo 场景模型 spawn 返回失败: {model_name}: {stderr}")
            return False

        self.spawned_models.add(model_name)
        self.logger.info(
            f"Gazebo 场景模型已 spawn: {model_name}, shape={obstacle.shape}, asset={spawn_path}")
        return True

    def clear(self):
        for model_name in sorted(self.spawned_models):
            self.remove_model(model_name, quiet=False)
        self.spawned_models.clear()


class SceneEnvironmentManager:
    def __init__(
        self,
        node,
        base_frame_name,
        scene_name,
        scene_config_file,
        scene_assets_dir,
        gazebo_world,
        obstacle_marker_topic,
        publish_planning_scene=True,
        publish_obstacle_markers=True,
        spawn_gazebo_scene_models=False,
        planning_scene_obstacle_padding_m=0.0,
    ):
        self.node = node
        self.base_frame_name = base_frame_name
        self.spawn_gazebo_scene_models = spawn_gazebo_scene_models
        self.loader = SceneLoader(scene_name, scene_config_file, node.get_logger())
        self.planning_scene = PlanningSceneManager(
            node,
            base_frame_name,
            publish_enabled=publish_planning_scene,
            obstacle_padding_m=planning_scene_obstacle_padding_m,
        )
        self.markers = MarkerPublisher(
            node,
            base_frame_name,
            obstacle_marker_topic,
            publish_enabled=publish_obstacle_markers,
        )
        self.gazebo = GazeboSceneSpawner(
            scene_name,
            scene_assets_dir,
            gazebo_world,
            node.get_logger(),
        )

    @property
    def benchmark(self):
        return self.loader.benchmark

    def load_scene(self, obstacle_boxes, default_name, default_position, default_size):
        return self.loader.load(
            obstacle_boxes,
            default_name,
            default_position,
            default_size,
        )

    def add_scene(self, obstacles):
        for obstacle in obstacles:
            self.planning_scene.add_obstacle(obstacle, frame_id=self.base_frame_name)
        self.markers.publish(obstacles)
        if self.spawn_gazebo_scene_models:
            for obstacle in obstacles:
                self.gazebo.spawn_obstacle(obstacle)

    def clear_scene(self, obstacles=None, extra_names=None):
        self.planning_scene.clear(obstacles=obstacles, extra_names=extra_names)
        self.markers.clear()
        self.gazebo.clear()

    def add_collision_box(self, name, position, size, frame_id=None, rpy_deg=None):
        obstacle = SceneObstacle.box(name, position, size, rpy_deg=rpy_deg)
        self.planning_scene.add_obstacle(obstacle, frame_id=frame_id or self.base_frame_name)
