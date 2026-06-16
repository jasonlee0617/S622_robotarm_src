"""YAML loading helpers used by gazebo_launch launch files."""

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Dict

from ament_index_python.packages import get_package_share_directory
import yaml


def load_yaml(package_name: str, relative_path: str) -> Dict[str, Any]:
    """Load a YAML file from a package share directory as a Python dict."""
    pkg_path = get_package_share_directory(package_name)
    abs_path = os.path.join(pkg_path, relative_path)
    with open(abs_path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data or {}


def load_ros_parameters_yaml(
    package_name: str,
    relative_path: str,
    node_scope: str,
) -> Dict[str, Any]:
    """Load `ros__parameters` for a node scope from a ROS2 params YAML file."""
    try:
        data = load_yaml(package_name, relative_path)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    scoped = data.get(node_scope, {})
    if not isinstance(scoped, dict):
        return {}

    params = scoped.get("ros__parameters", {})
    return params if isinstance(params, dict) else {}


def package_file(package_name: str, relative_path: str) -> str:
    """Return an absolute path inside a package share directory."""
    return os.path.join(get_package_share_directory(package_name), relative_path)


def wrap_yaml_as_ros_params_file(
    package_name: str,
    relative_path: str,
    node_scope: str = "/**",
) -> str:
    """Wrap a plain YAML mapping into a ROS2 params file and return its temp path."""
    payload = load_yaml(package_name, relative_path)
    wrapped = {
        node_scope: {
            "ros__parameters": payload,
        }
    }

    content_hash = hashlib.sha1(
        yaml.safe_dump(wrapped, sort_keys=False).encode("utf-8")
    ).hexdigest()[:10]
    file_stem = Path(relative_path).stem
    temp_path = os.path.join(
        tempfile.gettempdir(),
        f"{package_name}_{file_stem}_{content_hash}_ros_params.yaml",
    )
    with open(temp_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(wrapped, stream, sort_keys=False)
    return temp_path
