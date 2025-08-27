# Instructions (ROS2)
## citation
Please cite this repo if you are using it or part of it.
```
@software{h2r_franka_ROS2,
  author = {Jia, Mingxi and Tellex, Stefanie},
  doi = {10.5281/zenodo.15098249},
  month = {3},
  title = {{h2r_franka_ROS2}},
  url = {https://github.com/SaulBatman/h2r_franka_ROS2},
  version = {1.0.0},
  year = {2025}
}
```
## documentation
here is some more info (https://happy-wound-99f.notion.site/Tigress-documentation-15fc376d918d80daab7dfa9f3d2e9d65?pvs=4)

## robot
```
# (option 1) open robot without moveit
ros2 launch franka_bringup franka.launch.py robot_ip:=<ip> arm_id:=fr3 use_rviz:=true
# (option 2) open robot with official moveit
ros2 launch launch/moveit.launch.py robot_ip:=<ip>
# (option 3) open robot with modified real-time servoing moveit
ros2 launch franka bringup franka.launch.py robot_ip:=<ip>
ros2 service call /servo_node/start_servo std_srvs/srv/Trigger
ros2 run ~/panda_ws/code/spacemouse_servo_pub.py
# open gripper
ros2 launch franka_gripper gripper.launch.py robot_ip:<ip>
```
## robot commands
```
ros2 action send_goal -f /franka_gripper/grasp franka_msgs/action/Grasp "{width: 0.00, speed: 0.03, force: 1}"
ros2 action send_goal -f /franka_gripper/homing franka_msgs/action/Homing {}
```
## cameras
```
# open cameras
ros2 launch launch/launch_all_cameras.launch.py
```

## calibration
```
  cd ros_ws/src
  git clone https://github.com/marcoesposito1988/easy_handeye2
  cd ros_ws
  rosdep install -iyr --from-paths src
  colcon build
  # remember to change the parameters in launch file
  # please make sure the topics are available and the aruco tag is visible, or
  # the GUI won't work
  ros2 launch h2r_franka_ROS2/launch/eye_on_base.launch.py # eye-on-base
  ros2 launch h2r_franka_ROS2/launch/eye_in_hand.launch.py # eye-in-hand
```
# tricks (ROS2)
```
# rqt
rqt --clear-config
# rviz 
ros2 run rviz2 rviz2
# see transformation
ros2 run tf2_ros tf2_echo fr3_link0 dave_link
# load parameters
ros2 param load /dave/dave_camera configs/dave_config.yaml
```

### Instructions (ROS1)
1. Cameras
    ```
    roslaunch realsense2_camera rs_camera.launch serial_no:=234322306820 camera:=bob align_depth:=true depth_width:=640 depth_height:=480 depth_fps:=15 color_width:=640 color_height:=480 color_fps:=15
   roslaunch realsense2_camera rs_camera.launch serial_no:=f1420123 camera:=kevin align_depth:=true depth_width:=640 depth_height:=480 depth_fps:=15 color_width:=640 color_height:=480 color_fps:=15
   roslaunch realsense2_camera rs_camera.launch serial_no:=234222301686 camera:=stuart align_depth:=true depth_width:=640 depth_height:=480 depth_fps:=15 color_width:=640 color_height:=480 color_fps:=15
   roslaunch realsense2_camera rs_camera.launch serial_no:=239222303414 camera:=mel align_depth:=true depth_width:=640 depth_height:=480 depth_fps:=15 color_width:=640 color_height:=480 color_fps:=15
   roslaunch realsense2_camera rs_camera.launch serial_no:=239222303046 camera:=dave align_depth:=true depth_width:=640 depth_height:=480 depth_fps:=15 color_width:=640 color_height:=480 color_fps:=15

    ```
2. publish poses
    ```
    roslaunch camera_pose_publisher.launch
    ```
3. open gripper with command line
    ```
    rostopic pub --once /franka_gripper/move/goal franka_gripper/MoveActionGoal "goal: { width: 0.1, speed: 0.1 }"
    ```
