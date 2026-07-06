"""Bootstrap helpers: user-site filtering, cv2/cv_bridge import protection, build stamp.

本模块在采集器启动时运行，负责：
1. 过滤用户级 site-packages，避免与系统 OpenCV 冲突
2. 安全导入 cv2，确保包含 aruco 模块
3. 生成当前脚本的构建戳（用于运行时日志）
"""

import hashlib
import importlib
import os
import site
import sys
from typing import List

# ----------------------------------------------------------------------
# 常量定义
# ----------------------------------------------------------------------

# 环境变量 AUTO_COLLECTOR_ALLOW_USER_SITE 的合法值集合
_USER_SITE_ALLOW_VALUES = ("1", "true", "yes", "on")

# 采集器延迟启动时间（秒），保证 ROS 2 基础设施完全就绪
_COLLECTOR_START_DELAY_SEC = 0.5

# 图像编码到通道数的映射表，用于无 cv_bridge 时的后备转换
_IMAGE_CHANNELS_BY_ENCODING = {
    "bgr8": 3,
    "rgb8": 3,
    "mono8": 1,
    "bgra8": 4,
    "rgba8": 4,
    "8uc1": 1,
    "8uc3": 3,
    "8uc4": 4,
}


# ----------------------------------------------------------------------
# 用户 site-packages 路径处理
# ----------------------------------------------------------------------

def _user_site_paths() -> List[str]:
    """
    获取当前 Python 环境的用户级 site-packages 路径列表。
    若无法获取或为空，则返回空列表；所有路径均转为绝对路径。
    """
    paths = []
    try:
        user_site = site.getusersitepackages()
        # 处理可能返回单个字符串或列表的情况
        if isinstance(user_site, str):
            paths.append(user_site)
        else:
            paths.extend(user_site)
    except Exception:
        pass
    # 确保所有路径为绝对路径，去除空值
    return [os.path.abspath(path) for path in paths if path]


def _prefer_system_python_extensions() -> str:
    """
    过滤 sys.path 中的用户 site-packages，优先使用系统级 Python 扩展。

    返回字符串说明操作结果：
    - 若环境变量允许用户 site，直接返回提示
    - 若无用户 site，返回提示
    - 若成功移除，返回移除的路径列表
    """
    # 检查环境变量，若显式允许则跳过过滤
    if os.environ.get("AUTO_COLLECTOR_ALLOW_USER_SITE", "").strip().lower() in _USER_SITE_ALLOW_VALUES:
        return "user site enabled by AUTO_COLLECTOR_ALLOW_USER_SITE"

    user_paths = _user_site_paths()
    if not user_paths:
        return "no user site packages detected"

    filtered = []
    removed = []
    for path in sys.path:
        abs_path = os.path.abspath(path or os.getcwd())
        # 判断当前路径是否属于用户 site-packages
        if any(abs_path == user_path or abs_path.startswith(user_path + os.sep) for user_path in user_paths):
            removed.append(path)
            continue
        filtered.append(path)

    if not removed:
        return "user site already absent from sys.path"

    # 替换 sys.path 为过滤后的列表
    sys.path[:] = filtered
    # 尝试禁止 site 模块自动再次添加用户路径
    try:
        site.ENABLE_USER_SITE = False
    except Exception:
        pass
    return f"removed user site packages from sys.path: {', '.join(removed)}"


# ----------------------------------------------------------------------
# OpenCV 安全导入
# ----------------------------------------------------------------------

def _cv2_location(module) -> str:
    """返回 OpenCV 模块的位置和版本信息，便于诊断日志。"""
    return f"{getattr(module, '__file__', 'unknown')} ({getattr(module, '__version__', 'unknown')})"


def _import_cv2_with_aruco():
    """
    安全导入 cv2，确保包含 aruco 子模块。

    若首次导入的 cv2 不包含 aruco，则尝试移除用户 site-packages 后重新导入，
    以获取系统级的、编译了 aruco 的 OpenCV。

    返回 (cv2_module, note_string)：
    - cv2_module: 最终使用的 cv2 模块对象，或 None（导入失败）
    - note_string: 描述导入过程和结果的字符串
    """
    try:
        imported_cv2 = importlib.import_module("cv2")
    except Exception as exc:
        return None, f"OpenCV import failed: {exc}"

    first_note = _cv2_location(imported_cv2)
    # 如果已包含 aruco，直接返回
    if hasattr(imported_cv2, "aruco"):
        return imported_cv2, f"cv2={first_note}"

    # 未包含 aruco，尝试系统级回退
    user_paths = _user_site_paths()
    if not user_paths:
        return imported_cv2, f"cv2 lacks aruco: {first_note}"

    old_path = list(sys.path)  # 备份原始 sys.path
    removed_path = False
    try:
        filtered_path = []
        for path in sys.path:
            abs_path = os.path.abspath(path or os.getcwd())
            if any(abs_path == user_path or abs_path.startswith(user_path + os.sep) for user_path in user_paths):
                removed_path = True
                continue
            filtered_path.append(path)
        if not removed_path:
            return imported_cv2, f"cv2 lacks aruco: {first_note}"

        # 从 sys.modules 中移除已加载的 cv2 相关模块，避免缓存干扰
        for name in list(sys.modules):
            if name == "cv2" or name.startswith("cv2."):
                del sys.modules[name]
        sys.path = filtered_path
        fallback_cv2 = importlib.import_module("cv2")
        fallback_note = _cv2_location(fallback_cv2)
        if hasattr(fallback_cv2, "aruco"):
            return fallback_cv2, (
                "using system OpenCV with aruco after ignoring user site; "
                f"first={first_note}; selected={fallback_note}"
            )
        # 回退失败，恢复原始 cv2 并返回
        sys.modules["cv2"] = imported_cv2
        return imported_cv2, (
            f"cv2 lacks aruco after fallback; first={first_note}; "
            f"fallback={fallback_note}"
        )
    except Exception as exc:
        sys.modules["cv2"] = imported_cv2
        return imported_cv2, f"cv2 lacks aruco: {first_note}; system fallback failed: {exc}"
    finally:
        # 始终恢复 sys.path，避免对其他模块产生副作用
        sys.path = old_path


# ----------------------------------------------------------------------
# 构建戳生成
# ----------------------------------------------------------------------

def _script_build_stamp(file_path: str = __file__) -> str:
    """
    计算指定脚本文件的 SHA-1 摘要的前 12 位十六进制字符串，
    作为运行时构建戳，用于日志中区分不同版本的脚本。
    """
    try:
        with open(file_path, "rb") as stream:
            digest = hashlib.sha1(stream.read()).hexdigest()
        return digest[:12]
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------
# 模块级变量（在导入时自动执行一次）
# ----------------------------------------------------------------------

# 记录用户 site 过滤结果，在日志中输出
_PYTHON_SITE_NOTE = _prefer_system_python_extensions()

# cv2 导入说明符，将在主模块尝试导入 cv2 时被更新
_CV2_IMPORT_NOTE = ""