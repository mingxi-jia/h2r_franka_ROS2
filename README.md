# Instructions (ROS2)


# tricks (ROS2)


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
