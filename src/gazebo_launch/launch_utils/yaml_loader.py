"""YAML loading helpers used by gazebo_launch launch files."""

import os
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


def package_file(package_name: str, relative_path: str) -> str:
    """Return an absolute path inside a package share directory."""
    return os.path.join(get_package_share_directory(package_name), relative_path)

