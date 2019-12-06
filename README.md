# stack

## Requirements
* fmauch_universal_robot
* robotiq
* Universal_Robots_ROS_Driver

### Instructions

1. Start UR driver.
    ```
    roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=10.75.15.199 limited:=true headless_mode:=true
    ```

2. Start gripper driver.
    ```
    rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /dev/ttyUSB0  
    ```

3. Start depth sensor driver.
    ```
    roslaunch openni2_launch openni2.launch
    ```
