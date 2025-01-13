import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped, TwistStamped
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume, RobotState, JointConstraint
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg  import JointState, Joy
from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from control_msgs.msg import JointJog
from std_srvs.srv import Trigger  

import numpy as np
import time
from config.configs import EE_HOME, JOINT_HOME
from scipy.spatial.transform import Rotation as R

JOINT_NAMES = ['fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7', 'fr3_finger_joint1', 'fr3_finger_joint2']

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

class GoalPoseSubscriber(Node):
    def __init__(self):
        super().__init__('vr_servo_pub')
        self.base_link = "fr3_link0"
        self.ee_link = "fr3_hand_tcp"

        self.subscription = self.create_subscription(
            PoseStamped,
            '/my_pose',
            self.goal_pose_callback,
            20
        )

        # Subscription to joint_states
        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            20
        )

        # Subscription to joint_states
        self.ee_state_subscription = self.create_subscription(
            PoseStamped,
            '/franka_robot_state_broadcaster/current_pose',
            self.ee_state_callback,
            20
        )

        self.grasp_publisher_ = self.create_publisher(Joy, '/servo_node/grasp_cmds', 50)

        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.move_group_client = ActionClient(self, MoveGroup, 'move_action')
        self.move_group_client.wait_for_server()
        self.get_logger().info("move_group_client ready.")
        self.current_ee_pose: PoseStamped = None
        self.current_joint_state = None
        self.ik_solution = None

        while self.current_joint_state is None:
            self.get_logger().warn("Waiting for current_joint_state...")
            time.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=1.0)
        gripper_init_state = self.current_joint_state.position[-1]
        JOINT_HOME['fr3_finger_joint1'] = JOINT_HOME['fr3_finger_joint2'] = gripper_init_state

        # Service client for pausing servo node
        self.pause_servo_client = self.create_client(Trigger, '/servo_node/pause_servo')
        self.start_servo_client = self.create_client(Trigger, '/servo_node/start_servo')


        self.twist_publisher_ = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 50)
        self.get_logger().info('Connected to move_group action server.')

        self.pause_servo()
        if self.reset_robot():
            self.start_servo()

        # Initialize sequence index
        # self.sequence_index = 0
        # self.action_sequence = self.get_action_sequence()
        # self.timer = self.create_timer(0.3, self.execute_next_action)

    def ee_state_callback(self, msg: JointState):
        """Callback to update current joint states."""
        self.current_ee_pose = msg
        # self.get_logger().info(f'Updated current joint state: {msg.name}')
    
    def joint_state_callback(self, msg: JointState):
        """Callback to update current joint states."""
        self.current_joint_state = msg
        # self.get_logger().info(f'Updated current joint state: {msg.name}')
        
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


    def goal_pose_callback(self, msg: PoseStamped):
        if not self.current_ee_pose:
            self.get_logger().warning('No ee pose received yet. Waiting...')
            return
        self.get_logger().info(f'Goal received: {msg.pose}')
        goal_xyz = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        goal_quaternion = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        current_xyz = np.array([self.current_ee_pose.pose.position.x, self.current_ee_pose.pose.position.y, self.current_ee_pose.pose.position.z])
        current_quaternion = np.array([self.current_ee_pose.pose.orientation.x, self.current_ee_pose.pose.orientation.y, self.current_ee_pose.pose.orientation.z, self.current_ee_pose.pose.orientation.w])

        goal_T = R.from_quat(goal_quaternion).as_matrix()
        current_T = R.from_quat(current_quaternion).as_matrix()
        difference_T = np.linalg.inv(current_T) @ goal_T
        difference_euler = R.from_matrix(difference_T).as_euler('xyz')
        # difference_euler = [0.,0.,0.]

        servo_twist_command = TwistStamped()
        servo_twist_command.header.stamp = self.get_clock().now().to_msg()
        servo_twist_command.header.frame_id = 'fr3_link0'

        xyz_twist = goal_xyz - current_xyz
        servo_twist_command.twist.linear.x = xyz_twist[0]
        servo_twist_command.twist.linear.y = xyz_twist[1]
        servo_twist_command.twist.linear.z = xyz_twist[2]
        servo_twist_command.twist.angular.x = difference_euler[0]
        servo_twist_command.twist.angular.y = difference_euler[1]
        servo_twist_command.twist.angular.z = difference_euler[2]
        self.twist_publisher_.publish(servo_twist_command)
    
    def reset_robot(self):
        
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

    def goal_response_callback(self, future):
        """Callback for handling goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Reset goal was rejected.")
            return

        self.get_logger().info("Reset goal accepted, waiting 5 seconds for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        """Callback for handling motion result."""
        result = future.result()
        if result.status == 4:  # Assuming 1 indicates success
            self.get_logger().info("Robot successfully reset.")
        else:
            self.get_logger().error(f"Failed to reset robot. Error code: {result.status}")

    def pause_servo(self):
        """Pauses the servo node by calling the pause service."""
        self.get_logger().info("pause servo")
        if not self.pause_servo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Pause servo service not available.")
            return

        request = Trigger.Request()
        future = self.pause_servo_client.call_async(request)
        future.add_done_callback(self.pause_servo_response)
        time.sleep(1)

    def pause_servo_response(self, future):
        """Callback for handling the pause service response."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Servo node successfully paused.")
            else:
                self.get_logger().error(f"Failed to pause servo node: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Exception while pausing servo: {e}")

    def start_servo(self):
        """Pauses the servo node by calling the pause service."""
        if not self.start_servo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Pause servo service not available.")
            return

        request = Trigger.Request()
        future = self.start_servo_client.call_async(request)
        future.add_done_callback(self.start_servo_response)
        time.sleep(1)

    def start_servo_response(self, future):
        """Callback for handling the pause service response."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Servo node successfully started.")
            else:
                self.get_logger().error(f"Failed to start servo node: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Exception while starting servo: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = GoalPoseSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
