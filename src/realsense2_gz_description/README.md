# Realsense Gazebo description

[![Format Status](https://github.com/MarqRazz/realsense2_gz_description/actions/workflows/format.yaml/badge.svg)](https://github.com/MarqRazz/realsense2_gz_description/actions/workflows/format.yaml)

Description: This ROS 2 package is designed to be used in unison with [realsense2_description](https://github.com/IntelRealSense/realsense-ros/tree/ros2-master/realsense2_description) and allows for easy definition of Realsense cameras that can be simulated in Gazebo Fortress and newer. It may support other versions of Ignition Gazebo but this has not been tested.

## Running Example Launch

This package includes a launch file to start Gazebo, bridge the data to ROS 2 and display the simulated camera data in Rviz.

Once you have built this package and sourced your workspace you can run
```bash
ros2 launch realsense2_gz_description example_realsense_gazebo.launch.py
```

> Note: you can specify `headless:=false` and it will also open the Gazebo GUI.

Which should start a simulated camera in Gazebo with a few objects in front of it. In the Rviz window that launches you can see the RGB images streaming along with the point cloud in the main view.

<img src="doc/realsense_gazebo.png"  width="50%" >


# Example Usage in URDF

In your robots urdf.xacro include the desired Realsense model along with its Gazebo description.
```xml
<xacro:include filename="$(find realsense2_description)/urdf/_d415.urdf.xacro" />
<xacro:include filename="$(find realsense2_gz_description)/urdf/_d415.gazebo.xacro" />

```
Then call the xacros and specify the same `name` and other optional arguments.
```xml
<!-- URDF xacro-->
<xacro:sensor_d415 parent="world" name="$(arg camera_name)" ...>
  <origin xyz="0 0 0" rpy="0 0 0"/>
</xacro:sensor_d415>
<!-- Gazebo xacro-->
<xacro:gazebo_d415 name="$(arg camera_name)" .../>
```

> Note: Gazebo plugins can only be included once so the xacros in this repo assume that a parent will include/run the required Sensors plugin when starting simulation.
This plugin can be started from your URDF or world.sdf file.
```xml
<gazebo>
  <plugin filename="libignition-gazebo-sensors-system.so" name="ignition::gazebo::systems::Sensors">
    <render_engine>ogre2</render_engine>
  </plugin>
</gazebo>
```

you can also refer to the the [example.urdf.xacro](./urdf/example_d415_gazebo.urdf.xacro) included.

## Gazebo only features

Gazebo offers a triggered based RGB camera that can be enabled by passing `triggered="true"` to the Gazebo description xacro.
Currently this feature does not appear to not work with the RGBD sensor, but will hopefully be added soon.
Switching the camera to only `trigger` when requested allows developers to better control the computation load required by each sensor.
Note triggering is not a feature the Realsense hardware offers unfortunately.

To trigger the camera from the command line you can publish on the Gazebo topic `camera_name/trigger`.
```bash
ign topic -t "/name/trigger" -m Boolean -p "data: true" -n 1
```

## Contributing

pre-commit is a tool to automatically run formatting checks on each commit, which saves you from manually running them.
This repo requires formatting to pass before changes will be accepted.

Install pre-commit like this:

```
pip3 install pre-commit
```

Run this in the top directory of the repo to set up the git hooks:

```
pre-commit install
```
