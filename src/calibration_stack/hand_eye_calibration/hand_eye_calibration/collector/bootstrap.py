"""Bootstrap helpers: user-site filtering, cv2/cv_bridge import protection, build stamp."""

import hashlib
import importlib
import os
import site
import sys
from typing import List

_USER_SITE_ALLOW_VALUES = ("1", "true", "yes", "on")
_COLLECTOR_START_DELAY_SEC = 0.5
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


def _user_site_paths() -> List[str]:
    paths = []
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            paths.append(user_site)
        else:
            paths.extend(user_site)
    except Exception:
        pass
    return [os.path.abspath(path) for path in paths if path]


def _prefer_system_python_extensions() -> str:
    if os.environ.get("AUTO_COLLECTOR_ALLOW_USER_SITE", "").strip().lower() in _USER_SITE_ALLOW_VALUES:
        return "user site enabled by AUTO_COLLECTOR_ALLOW_USER_SITE"

    user_paths = _user_site_paths()
    if not user_paths:
        return "no user site packages detected"

    filtered = []
    removed = []
    for path in sys.path:
        abs_path = os.path.abspath(path or os.getcwd())
        if any(abs_path == user_path or abs_path.startswith(user_path + os.sep) for user_path in user_paths):
            removed.append(path)
            continue
        filtered.append(path)

    if not removed:
        return "user site already absent from sys.path"

    sys.path[:] = filtered
    try:
        site.ENABLE_USER_SITE = False
    except Exception:
        pass
    return f"removed user site packages from sys.path: {', '.join(removed)}"


def _cv2_location(module) -> str:
    return f"{getattr(module, '__file__', 'unknown')} ({getattr(module, '__version__', 'unknown')})"


def _import_cv2_with_aruco():
    try:
        imported_cv2 = importlib.import_module("cv2")
    except Exception as exc:
        return None, f"OpenCV import failed: {exc}"

    first_note = _cv2_location(imported_cv2)
    if hasattr(imported_cv2, "aruco"):
        return imported_cv2, f"cv2={first_note}"

    user_paths = _user_site_paths()
    if not user_paths:
        return imported_cv2, f"cv2 lacks aruco: {first_note}"

    old_path = list(sys.path)
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
        sys.modules["cv2"] = imported_cv2
        return imported_cv2, (
            f"cv2 lacks aruco after fallback; first={first_note}; "
            f"fallback={fallback_note}"
        )
    except Exception as exc:
        sys.modules["cv2"] = imported_cv2
        return imported_cv2, f"cv2 lacks aruco: {first_note}; system fallback failed: {exc}"
    finally:
        sys.path = old_path


def _script_build_stamp(file_path: str = __file__) -> str:
    try:
        with open(file_path, "rb") as stream:
            digest = hashlib.sha1(stream.read()).hexdigest()
        return digest[:12]
    except Exception:
        return "unknown"


# Module-level bootstrap note.
_PYTHON_SITE_NOTE = _prefer_system_python_extensions()
_CV2_IMPORT_NOTE = ""
