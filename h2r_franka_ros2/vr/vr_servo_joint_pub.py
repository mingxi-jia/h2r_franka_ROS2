import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume, RobotState, JointConstraint
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg  import JointState, Joy
from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from control_msgs.msg import JointJog

import numpy as np
import time
from config.configs import EE_HOME

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
        self.ee_link = "fr3_link8"

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

        self.grasp_publisher_ = self.create_publisher(Joy, '/servo_node/grasp_cmds', 50)

        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.current_joint_state = None
        self.ik_solution = None

        self.joint_publisher_ = self.create_publisher(JointJog, '/servo_node/delta_joint_cmds', 50)
        self.get_logger().info('Connected to move_group action server.')

        self.robot_reset(EE_HOME)

        # Initialize sequence index
        # self.sequence_index = 0
        # self.action_sequence = self.get_action_sequence()
        # self.timer = self.create_timer(0.3, self.execute_next_action)

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
    
    def robot_reset(self, EE_HOME: dict):
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
                self.send_joint_servo()
            else:
                self.get_logger().error('Failed to find IK solution.')
        except Exception as e:
            self.get_logger().error(f'IK service call failed: {e}')
        
    def send_joint_servo(self):
        if not self.ik_solution:
            self.get_logger().error('No IK solution available for motion planning.')
            return
        # print(self.ik_solution)
        servo_joint_command = JointJog()
        servo_joint_command.header.stamp = self.get_clock().now().to_msg()
        servo_joint_command.header.frame_id = 'fr3_link0'
        servo_joint_command.joint_names = JOINT_NAMES

        current_joint_state = np.array(self.current_joint_state.position)
        goal_joint_states = np.array(list(self.ik_solution.values()))
        self.get_logger().info(f"goal_joint_states:{goal_joint_states}")
        self.get_logger().info(f"current_joint_state:{current_joint_state}")
        displacement = (goal_joint_states - current_joint_state).tolist()
        self.get_logger().info(f"displacement:{displacement}")
        # print(displacement)
        servo_joint_command.displacements = displacement
        servo_joint_command.duration = 0.3
        
        self.joint_publisher_.publish(servo_joint_command)
        
        # self.get_logger().info('Sending motion plan goal.')

def main(args=None):
    rclpy.init(args=args)
    node = GoalPoseSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
