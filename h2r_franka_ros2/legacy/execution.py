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
import threading
from pynput import keyboard
from configs import EE_HOME, JOINT_HOME

# ros2 topic pub /my_pose geometry_msgs/msg/PoseStamped "header:
#   stamp:
#     sec: 1731568855
#     nanosec: 414478014
#   frame_id: base
# pose:
#   position:
#     x: 0.4337632181884935
#     y: -0.02329914460294792
#     z: 0.2822889512813378
#   orientation:
#     x: 0.9965204797386643
#     y: 0.059661637466405355
#     z: -0.04985858023698448
#     w: 0.030025586841789735"

class Executor(Node):
    def __init__(self, camera_name):
        super().__init__('goal_pose_subscriber')
        self.base_link = "base"
        self.ee_link = "fr3_hand_tcp"
        self.z_min = 0.06 # with small fin gripper
        self.z_max = 2.0

        self.current_joint_state = None
        self.ik_solution = None
        self.rgb, self.depth, self.current_pose, self.franka_joint = None, None, None, None

        # data getter
        self.camera_name = camera_name
        self.collect_freq = 10
        self.time_diff = 1. / self.collect_freq
        self.rgb_sub = Subscriber(self, Image, f'/{camera_name}/color/image_raw')
        self.depth_sub = Subscriber(self, Image, f'/{camera_name}/aligned_depth_to_color/image_raw')
        self.action_sub = Subscriber(self, TwistStamped, '/servo_node/delta_twist_cmds')
        self.pose_sub = Subscriber(self, PoseStamped, '/franka_robot_state_broadcaster/current_pose')
        self.franka_joint_sub = Subscriber(self, JointState, '/joint_states')
        self.gripper_action_sub = Subscriber(self, Joy, '/spacemouse/grasp_signals')
        self.get_logger().info("All data subscriber ready.")
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.pose_sub, self.franka_joint_sub],
            queue_size=20,          
            slop=self.time_diff
        )
        self.ts.registerCallback(self.sync_callback)
        self.get_logger().info("Data Synchronizer ready.")

        self.subscription = self.create_subscription(
            PoseStamped,
            '/my_pose',
            self.goal_pose_callback,
            20
        )

        # Create 
        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            20
        )
        while self.current_joint_state is None:
            self.get_logger().warn("Waiting for current_joint_state...")
            time.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=1.0)
        # initialize gripper state because the init open width is always different, causing problems for motion planning
        gripper_init_state = self.current_joint_state.position[-1]
        JOINT_HOME['fr3_finger_joint1'] = JOINT_HOME['fr3_finger_joint2'] = gripper_init_state

        # moveit utils
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.move_group_client = ActionClient(self, MoveGroup, 'move_action')
        self.move_group_client.wait_for_server()
        self.get_logger().info('Connected to move_group action server.')

        # grasp util (need to open spacemouse)
        self.grasp_publisher_ = self.create_publisher(
            Joy, 
            '/servo_node/grasp_cmds', 
            50)
        
        # setup franka gripping
        self.grasp_client = ActionClient(self, Grasp, '/franka_gripper/grasp')
        self.grasp_client.wait_for_server()
        self.get_logger().info(f'grasp agent ready!')

        self.homing_client = ActionClient(self, Move, '/franka_gripper/move')
        self.homing_client.wait_for_server()
        self.get_logger().info(f'homing agent ready!')

        self.gripper_speed = 0.05
        self.grasp_action_triggered = False
        self.homing_action_triggered = False
        self.remote_grasp_triggered = False

        # self.robot_reset_by_EE(EE_HOME)
        # self.robot_reset_by_JOINT()

        # Initialize sequence index
        # self.sequence_index = 0
        # self.action_sequence = self.get_action_sequence()
        # self.timer = self.create_timer(0.3, self.execute_next_action)

    

    def joint_state_callback(self, msg: JointState):
        """Callback to update current joint states."""
        self.current_joint_state = msg
        # self.get_logger().info(f'Updated current joint state: {msg.name}')
    
    def sync_callback(self, rgb: Image, depth: Image, pose_stamped: PoseStamped, franka_joint_states: JointState):
        # Collect synchronized data
        rgb_data = np.array(rgb.data).reshape((rgb.height, rgb.width, -1))
        depth_data = np.array(depth.data).reshape((depth.height, depth.width, -1))
        pose = pose_stamped.pose
        current_pose = np.array([pose.position.x, pose.position.y, pose.position.z,
                              pose.orientation.x, pose.orientation.y,
                              pose.orientation.z, pose.orientation.w])
        franka_joint_data = np.array(franka_joint_states.position)

        # Perform an action on the synchronized data
        self.rgb, self.depth, self.current_pose, self.franka_joint = rgb_data, depth_data, current_pose, franka_joint_data
        # self.get_logger().info('Refreshed observation.')


    def get_sensor_input(self):
        if self.rgb is None or self.depth is None or self.current_pose is None or self.franka_joint is None:
            self.get_logger().warn("Incomplete sensor data. Returning None.")
            return None

        return {
            'rgb': self.rgb,
            'depth': self.depth,
            'current_pose': self.current_pose,
            'franka_joint': self.franka_joint,
        }
        # return sensor_mock
    
    def clear_cache(self):
        self.rgb, self.depth, self.current_pose, self.franka_joint = None, None, None, None

    def close_gripper(self, width=0.00, force=50.0):
        epsilon = GraspEpsilon()
        epsilon.inner = 0.05
        epsilon.outer = 0.05
        goal_msg = Grasp.Goal(width=width, speed=self.gripper_speed, force=force, epsilon=epsilon)
        future = self.grasp_client.send_goal_async(goal_msg)
        future.add_done_callback(self.grasp_response_callback)
    
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
        goal_msg = Move.Goal(width=0.1, speed=self.gripper_speed)
        future = self.homing_client.send_goal_async(goal_msg)
        future.add_done_callback(self.homing_response_callback)
    
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

    
    def get_action_sequence(self):
        """
        Define a sequence of actions as a list of tuples:
        Each tuple contains (xyzrpy, gripper_action)
        """
        episode = np.load("test_2024-11-18-13-36.npy", allow_pickle=True)[()]['episode_0']
        action_seq = []
        # action_abs
        for step in episode:
            action = step['action_abs']
            gripper_action = step['grasp_action']
            action = np.append(action, gripper_action) #x y z qx qy qz qw g
            action_seq.append(action)
        return np.stack(action_seq)
    
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
        for joint, position in JOINT_HOME.items():
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
    
    def execute_next_action(self):
        if self.sequence_index >= len(self.action_sequence):
            self.get_logger().info('Action sequence completed.')
            self.timer.cancel()  # Stop the timer
            return

        xyzqxqyqzqwg = self.action_sequence[self.sequence_index]
        # print(xyzqxqyqzqwg)
        pose_msg = PoseStamped()
        pose_msg.pose.position.x = xyzqxqyqzqwg[0]
        pose_msg.pose.position.y = xyzqxqyqzqwg[1]
        pose_msg.pose.position.z = xyzqxqyqzqwg[2]
        pose_msg.pose.orientation.x = xyzqxqyqzqwg[3]
        pose_msg.pose.orientation.y = xyzqxqyqzqwg[4]
        pose_msg.pose.orientation.z = xyzqxqyqzqwg[5]
        pose_msg.pose.orientation.w = xyzqxqyqzqwg[6]
        self.ik_request(pose_msg)

        gripper_msg = Joy()
        gripper_msg.buttons = [int(xyzqxqyqzqwg[7])]
        self.grasp_publisher_.publish(gripper_msg)

        self.sequence_index += 1

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
                self.ik_solution = dict(zip(
                    response.solution.joint_state.name,
                    response.solution.joint_state.position
                ))
                # self.get_logger().info(f'IK solution found: {self.ik_solution}')
                self.send_motion_plan()
            else:
                self.get_logger().error('Failed to find IK solution.')
        except Exception as e:
            self.get_logger().error(f'IK service call failed: {e}')
        
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
        goal_msg.request.max_velocity_scaling_factor = 0.1
        goal_msg.request.max_acceleration_scaling_factor = 0.1
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

def main(args=None):
    rclpy.init(args=args)
    node = Executor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
