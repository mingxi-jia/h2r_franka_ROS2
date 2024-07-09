


### Instructions
1. Cameras
    ```
    roslaunch realsense2_camera rs_camera.launch serial_no:=234322306820 camera:=kevin align_depth:=true depth_width:=640 depth_height:=480 depth_fps:=30 color_width:=840 color_height:=480 color_fps:=30
   roslaunch realsense2_camera rs_camera.launch serial_no:=234222301686 camera:=stuart align_depth:=true depth_width:=640 depth_height:=480 depth_fps:=30 color_width:=640 color_height:=480 color_fps:=30
   roslaunch realsense2_camera rs_camera.launch serial_no:=234222300548 camera:=bob align_depth:=true depth_width:=640 depth_height:=480 depth_fps:=30 color_width:=640 color_height:=480 color_fps:=30
    ```
2. publish poses
```
roslaunch camera_pose_publisher.launch
```
