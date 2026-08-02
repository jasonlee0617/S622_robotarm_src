# D435 hardware profiles

This directory contains canonical, per-resolution D435 `CameraInfo` profiles
used by the Fairino Gazebo camera model. They are hardware/profile data, not
hand-eye calibration results or device archives.

Capture a profile with:

```bash
ros2 run hand_eye_calibration capture_d435_profile.py --ros-args \
  -p color_profile:=<color-width>x<color-height>x<fps> \
  -p depth_profile:=<depth-width>x<depth-height>x<fps>
```

该脚本会原子性地替换所选彩色/深度配置对应的标准文件，并在 captured_at 字段中记录采集时间。
在 Gazebo 中使用标准配置文件时，通过 camera_profile:=<文件名主干> 指定，例如
camera_profile:=d435_color_640x480x30_depth_640x480x30。
仅当需要临时外部配置文件时才使用 camera_profile_file:=<yaml 文件>。
camera_profile 与 camera_profile_file 互斥；对于默认使用命名配置文件的入口节点，若要改用外部文件，需传入 camera_profile:='' 并同时指定外部文件。

文件名中的 x30 后缀仅记录采集时的模式。被选用的配置文件会改变彩色图像的几何与畸变参数，而不会影响启动时的 camera_fps，因此采集到的 640×480 标定数据可以有意地用在 60 FPS 的视觉伺服仿真中。