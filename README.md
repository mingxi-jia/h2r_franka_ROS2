# stack

## Requirements
* fmauch_universal_robot
* robotiq
* Universal_Robots_ROS_Driver
* Catkin make with python 3.7
   ```   
   catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3.7
   ```
## Dependencies


### Instructions
1. Start UR driver.
    ```
    roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=10.75.15.168 limited:=true headless_mode:=true
    ```

2. Start gripper driver.
    ```
    rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /dev/ttyUSB0  
    ```

3. Start depth sensor driver.
    ```
    roslaunch openni2_launch openni2.launch
    ```

## Troubleshooting
1. Install python3 catkin packages:
   ```
   sudo apt-get install python-catkin-pkg
   sudo apt-get install python3-catkin-pkg-modules
   sudo apt-get install python3-rospkg-modules
   ```
   https://answers.ros.org/question/245967/importerror-no-module-named-rospkg-python3-solved/?answer=298221#post-id-298221
1. Opening openni2 driver: ResourceNotFound: rgbd_launch:
   ```
   sudo apt install ros-kinetic-rgbd-launch
   ```
1. Could not find "controller interface"
   ```
   sudo apt-get install ros-kinetic-ros-control ros-kinetic-ros-controllers
   ```