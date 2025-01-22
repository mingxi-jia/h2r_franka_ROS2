import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup
from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume, RobotState, JointConstraint
from sensor_msgs.msg import Image, JointState, Joy
from sensor_msgs.msg  import JointState, Joy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import Pose, PoseStamped, TwistStamped
from shape_msgs.msg import SolidPrimitive
from franka_msgs.action import Grasp, Homing, Move
from franka_msgs.msg import GraspEpsilon

import numpy as np
import time
import copy
import threading
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

from panda_utils.configs import EE_HOME, JOINT_HOME

def calculate_waypoints(current_xyz, goal_xyz, num_waypoints):
    current_xyz, goal_xyz = np.asarray(current_xyz), np.asarray(goal_xyz)
    intermediate_distance = (goal_xyz - current_xyz) / (num_waypoints + 1)

    waypoints = []
    for i in range(num_waypoints + 1):
        waypoints.append(current_xyz + intermediate_distance * (i + 1))
    return waypoints

class ArmControl(Node):
    def __init__(self, joint_home:dict=None):
        super().__init__('goal_pose_subscriber')
        self.base_link = "base"
        self.ee_link = "fr3_hand_tcp"
        self.z_min = 0.1 # with small fin gripper
        self.z_max = 2.0

        self.current_joint_state = None
        self.ik_solution = None
        self.current_ee_pose = None
        self.joint_home = joint_home if joint_home is not None else JOINT_HOME

        # Create 
        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/franka/joint_states',
            self.joint_state_callback,
            20
        )
        while self.current_joint_state is None:
            self.get_logger().warn("Waiting for current_joint_state...")
            time.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=1.0)
        # initialize gripper state because the init open width is always different, causing problems for motion planning
        # gripper_init_state = self.current_joint_state.position[-1]
        # self.joint_home['fr3_finger_joint1'] = self.joint_home['fr3_finger_joint2'] = gripper_init_state

        self.ee_pose_subscription = self.create_subscription(PoseStamped, '/franka_robot_state_broadcaster/current_pose', self.ee_pose_callback, 50)

        # moveit utils
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.move_group_client = ActionClient(self, MoveGroup, 'move_action')
        self.move_group_client.wait_for_server()
        self.get_logger().info('Connected to move_group action server.')
        
        # setup franka gripping
        self.grasp_client = ActionClient(self, Grasp, '/fr3_gripper/grasp')
        self.grasp_client.wait_for_server()
        self.get_logger().info(f'grasp agent ready!')

        self.homing_client = ActionClient(self, Move, '/fr3_gripper/move')
        self.homing_client.wait_for_server()
        self.get_logger().info(f'homing agent ready!')

        self.gripper_speed = 0.2
        self.grasp_action_triggered = False
        self.homing_action_triggered = False
        self.remote_grasp_triggered = False

        self.open_gripper()
        time.sleep(2.0)
        self.robot_reset_by_JOINT()

    #----------------------------low-level apis------------------------------
    def ee_pose_callback(self, msg: PoseStamped):
        self.current_ee_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def joint_state_callback(self, msg: JointState):
        """Callback to update current joint states."""
        self.current_joint_state = msg
        # self.get_logger().info(f'Updated current joint state: {msg.name}')
    
    def close_gripper(self, width=0.00, force=50.0):
        # self.joint_home['fr3_finger_joint1'] = self.joint_home['fr3_finger_joint2'] = 0.
        epsilon = GraspEpsilon()
        epsilon.inner = 0.05
        epsilon.outer = 0.05
        goal_msg = Grasp.Goal(width=width, speed=self.gripper_speed, force=force, epsilon=epsilon)
        future = self.grasp_client.send_goal_async(goal_msg)
        future.add_done_callback(self.grasp_response_callback)
        time.sleep(2.0)
    
    def grasp_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Grasp goal rejected.')
            return
        self.get_logger().info('Grasp goal accepted.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.grasp_result_callback)

    def grasp_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Grasp result: {result}')

    def send_homing_goal(self):
        goal_msg = Move.Goal(width=0.1, speed=self.gripper_speed)
        future = self.homing_client.send_goal_async(goal_msg)
        future.add_done_callback(self.homing_response_callback)
    
    def open_gripper(self):
        # self.joint_home['fr3_finger_joint1'] = self.joint_home['fr3_finger_joint2'] = self.current_joint_state.position[-1]
        goal_msg = Move.Goal(width=0.1, speed=self.gripper_speed)
        future = self.homing_client.send_goal_async(goal_msg)
        future.add_done_callback(self.homing_response_callback)
        time.sleep(2.0)
    
    def homing_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Homing goal rejected.')
            return
        self.get_logger().info('Homing goal accepted.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.homing_result_callback)

    def homing_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Homing result: {result}')

    def move_robot(self, goal_xyz, goal_pose_quat, gripper_bool):
        # Send movement command to the robot based on the twist action
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = goal_xyz[0]
        pose_msg.pose.position.y = goal_xyz[1]
        pose_msg.pose.position.z = np.clip(goal_xyz[2], self.z_min, self.z_max)
        pose_msg.pose.orientation.x = goal_pose_quat[0]
        pose_msg.pose.orientation.y = goal_pose_quat[1]
        pose_msg.pose.orientation.z = goal_pose_quat[2]
        pose_msg.pose.orientation.w = goal_pose_quat[3]
        

        # Here we would publish to a relevant topic to move the robot
        self.ik_request(pose_msg)
        # self.get_logger().info(f"Movement command sent: {twist}")
        if round(gripper_bool) == 0:
            self.homing_action_triggered = False
            if not self.grasp_action_triggered:
                self.grasp_action_triggered = True
                self.close_gripper()
        else:
            self.grasp_action_triggered = False
            if not self.homing_action_triggered:
                self.homing_action_triggered = True
                self.open_gripper()

    def robot_reset_by_EE(self, EE_HOME: dict):
        while self.current_joint_state is None:
            time.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0.1)
        home_msg = PoseStamped()
        home_msg.pose.position.x = EE_HOME['x']
        home_msg.pose.position.y = EE_HOME['y']
        home_msg.pose.position.z = EE_HOME['z']
        home_msg.pose.orientation.x = EE_HOME['qx']
        home_msg.pose.orientation.y = EE_HOME['qy']
        home_msg.pose.orientation.z = EE_HOME['qz']
        home_msg.pose.orientation.w = EE_HOME['qw']
        self.ik_request(home_msg)
        time.sleep(0.1)
        self.get_logger().info('Waiting for resetting.')

    def robot_reset_by_JOINT(self):
        
        while self.current_joint_state is None:
            self.get_logger().warn("Waiting for current_joint_state...")
            time.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=1.0)

        goal_msg = MoveGroup.Goal()
        # Fill goal_msg with relevant goal constraints from PoseStamped
        goal_msg.request.workspace_parameters.header.frame_id = self.base_link
        goal_msg.request.workspace_parameters.min_corner.x = -1.0
        goal_msg.request.workspace_parameters.min_corner.y = -1.0
        goal_msg.request.workspace_parameters.min_corner.z = -1.0
        goal_msg.request.workspace_parameters.max_corner.x = 1.0
        goal_msg.request.workspace_parameters.max_corner.y = 1.0
        goal_msg.request.workspace_parameters.max_corner.z = 1.0

        start_state = RobotState()
        start_state.joint_state = self.current_joint_state
        goal_msg.request.start_state.is_diff = False
        goal_msg.request.start_state = start_state

        # Set goal constraints using the IK solution
        constraints = Constraints()
        for joint, position in self.joint_home.items():
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint
            joint_constraint.position = position
            joint_constraint.tolerance_above = 0.0001
            joint_constraint.tolerance_below = 0.0001
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)

        # print(constraints)
        goal_msg.request.goal_constraints.append(constraints)
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.group_name = 'fr3_arm'
        goal_msg.request.pipeline_id = 'move_group'
        goal_msg.request.max_velocity_scaling_factor = 0.3
        goal_msg.request.max_acceleration_scaling_factor = 0.3
        # goal_msg.request.planner_id = 'RRTConnectkConfigDefault'

        # print(goal_msg)

        self.get_logger().info("Sending reset goal... Approximately 5s.")
        future = self.move_group_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)
        time.sleep(5.0)
        return True  # Return early to avoid blocking

    def goal_pose_callback(self, msg: PoseStamped):
        if not self.current_joint_state:
            self.get_logger().warning('No joint state received yet. Waiting...')
            return
        self.get_logger().info(f'Goal received: {msg.pose}')
        self.ik_request(msg)
        
    def ik_request(self, msg: PoseStamped):
        # IK Request
        request = GetPositionIK.Request()
        request.ik_request.group_name = 'fr3_arm'
        request.ik_request.robot_state.joint_state = self.current_joint_state
        # request.ik_request.pose_stamped.header.frame_id = self.base_link
        request.ik_request.pose_stamped = msg
        request.ik_request.ik_link_name = self.ee_link
        request.ik_request.timeout.sec = 1

        if self.ik_client.service_is_ready():
            future = self.ik_client.call_async(request)
            future.add_done_callback(self.ik_response_callback)
        else:
            self.get_logger().error('IK service not available.')

    def ik_response_callback(self, future):
        try:
            response = future.result()
            if response.error_code.val == response.error_code.SUCCESS:
                joint_names = response.solution.joint_state.name[:7] # abondon gripper states
                joint_positions = response.solution.joint_state.position[:7]
                ik_solution = dict(zip(
                    joint_names,
                    joint_positions
                ))
                self.ik_solution = ik_solution
                # self.get_logger().info(f'IK solution found: {self.ik_solution}')
                self.send_motion_plan()
            else:
                self.get_logger().error('Failed to find IK solution.')
                self.ik_solution = None
        except Exception as e:
            self.get_logger().error(f'IK service call failed: {e}')
            self.ik_solution = None

    def send_motion_plan(self):
        if not self.ik_solution:
            self.get_logger().error('No IK solution available for motion planning.')
            return
        # print(self.ik_solution)
        goal_msg = MoveGroup.Goal()
        # Fill goal_msg with relevant goal constraints from PoseStamped
        goal_msg.request.workspace_parameters.header.frame_id = self.base_link
        goal_msg.request.workspace_parameters.min_corner.x = -1.0
        goal_msg.request.workspace_parameters.min_corner.y = -1.0
        goal_msg.request.workspace_parameters.min_corner.z = -1.0
        goal_msg.request.workspace_parameters.max_corner.x = 1.0
        goal_msg.request.workspace_parameters.max_corner.y = 1.0
        goal_msg.request.workspace_parameters.max_corner.z = 1.0

        start_state = RobotState()
        start_state.joint_state = self.current_joint_state
        # print(self.current_joint_state)
        goal_msg.request.start_state = start_state
        goal_msg.request.start_state.is_diff = False

        # self.ik_solution['fr3_finger_join1'] = self.ik_solution['fr3_finger_join1'] = self.current_joint_state.position[-1]

        # Set goal constraints using the IK solution
        constraints = Constraints()
        for joint, position in self.ik_solution.items():
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint
            joint_constraint.position = position
            joint_constraint.tolerance_above = 0.0001
            joint_constraint.tolerance_below = 0.0001
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)

        goal_msg.request.goal_constraints.append(constraints)
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 0.1
        goal_msg.request.group_name = 'fr3_arm'
        goal_msg.request.pipeline_id = 'move_group'
        goal_msg.request.max_velocity_scaling_factor = 0.4
        goal_msg.request.max_acceleration_scaling_factor = 0.4
        # print(goal_msg)

        self.get_logger().info('Sending motion plan goal.')
        self.move_group_client.send_goal_async(goal_msg).add_done_callback(self.goal_response_callback)
        
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by move_group.')
            return

        self.get_logger().info('Goal accepted, waiting for result...')
        goal_handle.get_result_async().add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Motion plan result: {result.error_code}')

    def clear_cache(self):
        self.current_joint_state = None
        self.ik_solution = None
        self.current_ee_pose = None
        self.grasp_action_triggered = False
        self.homing_action_triggered = False
        self.remote_grasp_triggered = False

    #----------------------------high-level apis------------------------------
    def joint_is_reach(self, ik_solution: dict):
        if ik_solution is None:
            return False
        
        while self.current_joint_state is None:
            time.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0.1)

        for joint, position in ik_solution.items():
            index = self.current_joint_state.name.index(joint)
            if abs(self.current_joint_state.position[index]-position) >= 0.02:
                return False
        return True

    def wait_for_motion(self, x, y, z):
        manhattan_dist = np.array([x, y, z]) - self.current_ee_pose
        arm_speed = 0.08
        estimated_time = np.abs(manhattan_dist).sum() / arm_speed
        self.get_logger().info(f"EST: {estimated_time}")

    def goto(self, x, y, z, quaternion_xyzw):
        self.current_ee_pose = None
        while self.current_joint_state is None or self.current_ee_pose is None:
            time.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0.1)

        goto_msg = PoseStamped()
        goto_msg.pose.position.x = x
        goto_msg.pose.position.y = y
        goto_msg.pose.position.z = z
        goto_msg.pose.orientation.x = quaternion_xyzw[0]
        goto_msg.pose.orientation.y = quaternion_xyzw[1]
        goto_msg.pose.orientation.z = quaternion_xyzw[2]
        goto_msg.pose.orientation.w = quaternion_xyzw[3]
        self.ik_request(goto_msg)
        self.get_logger().info('Waiting for getting to the goal.')

        starting_time = time.time()
        while not self.joint_is_reach(self.ik_solution):
            time.sleep(0.01)
            rclpy.spin_once(self, timeout_sec=0.1)
            if (time.time() - starting_time) > 6.0:
                print("moveit timeout")
                break #timeout

        self.get_logger().info('Goal reached.')
        self.ik_solution = None

    def waypoints_goto(self, x, y, z, quaternion_xyzw):
        self.current_ee_pose = None
        while self.current_joint_state is None or self.current_ee_pose is None:
            time.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0.1)
        current_ee_pose = copy.copy(self.current_ee_pose)
        waypoints: list[np.array] = calculate_waypoints(current_xyz=current_ee_pose, goal_xyz=[x, y, z], num_waypoints=2)

        for wp in waypoints:
            x, y, z = wp
            self.goto(x, y, z, quaternion_xyzw)

    def reset(self):
        self.robot_reset_by_JOINT()

    def pick(self, x, y, r, z=None):
        print(f"picking at x:{x}, y:{y}, r{r/np.pi*180} degree")
        if z == None:
            z = self.z_min
        quaternion_xyzw = R.from_euler('XYZ', [np.pi, 0., r]).as_quat()
        self.waypoints_goto(x, y, z + 0.2, quaternion_xyzw)
        self.waypoints_goto(x, y, z, quaternion_xyzw)
        self.close_gripper()
        self.goto(x, y, z + 0.2, quaternion_xyzw)


    def place(self, x, y, r, z=None):
        print(f"placing at x:{x}, y:{y}, r{r/np.pi*180} degree")
        if z == None:
            z = self.z_min
        quaternion_xyzw = R.from_euler('XYZ', [np.pi, 0., r]).as_quat()
        place_z_offset = 0.05
        self.goto(x, y, z + 0.2, quaternion_xyzw)
        self.waypoints_goto(x, y, z + place_z_offset, quaternion_xyzw)
        self.open_gripper()
        self.waypoints_goto(x, y, z + 0.2, quaternion_xyzw)

    def push(self, push_start_action, push_end_action):
        x1, y1, r1 = push_start_action
        quaternion_xyzw1 = R.from_euler('XYZ', [np.pi, 0., r1]).as_quat()
        x2, y2, r2 = push_end_action
        # quaternion_xyzw2 = R.from_euler('XYZ', [np.pi, 0., r2]).as_quat()
        quaternion_xyzw2 = quaternion_xyzw1

        z_max, z_min = 0.3, 0.1
        # go to start pose
        self.goto(x1, y1, z_max, quaternion_xyzw1)
        self.waypoints_goto(x1, y1, z_min, quaternion_xyzw1)
        self.close_gripper()
        # push
        self.waypoints_goto(x2, y2, z_min, quaternion_xyzw2)
        self.open_gripper()
        self.goto(x2, y2, z_max, quaternion_xyzw2)
        

class DummyRobot():
    def __init__(self):
        pass

    def goto(x, y, z, quaternion_xyzw):
        pass

    def reset(self):
        pass

    def pick(self, x, y, r, z=None):
        pass

    def place(self, x, y, r, z=None):
        pass

    def open_gripper():
        pass

    def destroy_node():
        pass

def main(args=None):
    rclpy.init(args=args)
    node = ArmControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
