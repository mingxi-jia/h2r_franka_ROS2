import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from scipy.spatial.transform import Rotation as R
import numpy as np

from std_msgs.msg import String, Bool
from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState  
# from franka_msgs.action import Grasp, Homing, Move
# from franka_msgs.msg import GraspEpsilon

from droid.oculus_controller import VRPolicy

class SpacemouseServoPub(Node):

    def __init__(self, four_dof_control):
        super().__init__('vr_pub_vel')
        self.vr_policy = VRPolicy()
        self.four_dof_control = four_dof_control
        control_mode = "4DoF" if four_dof_control else "6DoF"
        self.get_logger().info(f'Using {control_mode} control')

        #subscribers
        self.cartesian_pose_sub = self.create_subscription(
            PoseStamped,
            '/franka_robot_state_broadcaster/current_pose',
            self.cartesian_pose_callback,
            10)

        self.gripper_state_sub = self.create_subscription(
            JointState,
            '/fr3_gripper/joint_states',
            self.gripper_state_callback,
            10)

        self.current_robot_state = {
            "cartesian_position": [None, None, None, None, None, None],  # Cartesian pose placeholder
            "gripper_position": 0.03953560069203377,  # Gripper state placeholder
        }


        self.grasp_subscription = self.create_subscription(Joy, '/servo_node/grasp_cmds', self.grasp_callback, 50)
        self.grasp_subscription  # prevent unused variable warning

        self.twist_publisher_ = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 50)
        self.get_logger().info(f'spacemouse_servo_pub ready!')

        self.grasp_publisher_ = self.create_publisher(Joy, '/spacemouse/grasp_signals', 50)
        self.get_logger().info(f'spacemouse_servo_pub ready!')
        
        # self.grasp_client = ActionClient(self, Grasp, '/fr3_gripper/grasp')
        # self.grasp_client.wait_for_server()
        # self.get_logger().info(f'grasp agent ready!')

        # self.homing_client = ActionClient(self, Move, '/fr3_gripper/move')
        # self.homing_client.wait_for_server()
        # self.get_logger().info(f'homing agent ready!')

        self.grasp_action_triggered = False
        self.homing_action_triggered = False
        self.remote_grasp_triggered = False

        self.speed = 0.05

        self.timer = self.create_timer(0.1, self.publish_vr_state)  # Publish at 10 Hz

    def publish_vr_state(self):
            # Ensure robot state is available
        if None in self.current_robot_state["cartesian_position"]:
            self.get_logger().info("No franka robot state available. Waiting.")
            return

        # Use VRPolicy's forward function with the current robot state
        action, info = self.vr_policy.forward({"robot_state": self.current_robot_state}, include_info=True)
        # if info != {}:
        #     print(time.time(), info["target_cartesian_position"])
        print(action)
        # if np.allclose(action, 0.0):
        #     self.get_logger().info("Default action detected. Skipping publishing.")
        #     return

        orig_x, orig_y, orig_z, orig_rx, orig_ry, orig_rz, gripper = action
        # button_grasp, button_homing = msg.buttons
        # panda frame
        # xyzrpy = [-orig_y, -orig_x, orig_z, -orig_ry, -orig_rx, orig_rz]
        xyzrpy = [-orig_x, -orig_y, orig_z, -orig_rx, -orig_ry, orig_rz,]
        print(xyzrpy)
        self.publish_twist(xyzrpy)

        # self.get_logger().info(f"Published VR State: {pose_msg.pose}")

        # # Trigger Grasp action on button press
        # if button_grasp == 1 and not self.grasp_action_triggered:
        #     self.grasp_action_triggered = True
        #     self.send_grasp_goal()

        # # Reset Grasp action state on button release
        # if button_homing == 1 and self.grasp_action_triggered:
        #     self.grasp_action_triggered = False

        # # Trigger Homing action on button press
        # if button_homing == 1 and not self.homing_action_triggered:
        #     self.homing_action_triggered = True
        #     self.send_homing_goal()

        # # Reset Homing action state on button release
        # if button_grasp == 1 and self.homing_action_triggered:
        #     self.homing_action_triggered = False

    # subscription functions
    def cartesian_pose_callback(self, msg: PoseStamped):
        # Extract position
        position = msg.pose.position
        x, y, z = position.x, position.y, position.z

        # Extract orientation (quaternion)
        orientation = msg.pose.orientation
        quat_x, quat_y, quat_z, quat_w = orientation.x, orientation.y, orientation.z, orientation.w
        roll, pitch, yaw = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler('xyz')

        # Store the data in the class variable
        self.current_robot_state["cartesian_position"] = [x, y, z, roll, pitch, yaw]

        # self.get_logger().info(f"Updated Cartesian Position: {self.current_robot_state['cartesian_position']}")

    def gripper_state_callback(self, msg: JointState):
        # Extract the gripper state (assuming it's the first joint's position)
        gripper_position = msg.position[0]

        # Store the gripper state in the class variable
        self.current_robot_state["gripper_position"] = gripper_position

        # self.get_logger().info(f"Updated Gripper Position: {self.current_robot_state['gripper_position']}")


    def grasp_callback(self, msg: Joy):
        button_grasp = msg.buttons[0]
        # Trigger Grasp action on button press
        if button_grasp == 1 and not self.remote_grasp_triggered:
            self.remote_grasp_triggered = True
            self.send_grasp_goal()

        # Reset Grasp action state on button release
        if button_grasp == 0 and self.remote_grasp_triggered:
            self.remote_grasp_triggered = False
            self.send_homing_goal()

        # # Trigger Homing action on button press
        # if button_homing == 1 and not self.homing_action_triggered:
        #     self.homing_action_triggered = True
        #     self.send_homing_goal()

        # # Reset Homing action state on button release
        # if button_grasp == 1 and self.homing_action_triggered:
        #     self.homing_action_triggered = False

    def send_grasp_goal(self, width=0.00, force=50.0):
        epsilon = GraspEpsilon()
        epsilon.inner = 0.05
        epsilon.outer = 0.05
        goal_msg = Grasp.Goal(width=width, speed=self.speed, force=force, epsilon=epsilon)
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
        goal_msg = Move.Goal(width=0.1, speed=self.speed)
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

    def publish_twist(self, xyzrpy):
        x, y, z, rx, ry, rz = xyzrpy
        if self.four_dof_control:
            rx, ry = 0., 0.
        msg = TwistStamped()
        stamp = self.get_clock().now().to_msg()
        msg.header.stamp = stamp
        msg.header.frame_id = 'fr3_link0'
        msg.twist.linear.x = x
        msg.twist.linear.y = y
        msg.twist.linear.z = z
        msg.twist.angular.x = rx
        msg.twist.angular.y = ry
        msg.twist.angular.z = rz
        self.twist_publisher_.publish(msg)

        grasp_msg = Joy()
        grasp_msg.header.stamp = stamp
        grasp_msg.buttons = [1] if self.grasp_action_triggered else [0]
        self.grasp_publisher_.publish(grasp_msg)
        # self.get_logger().info(f'Published: "{msg.twist}"')
        
         

def main(args=None):
    rclpy.init(args=args)
    node = SpacemouseServoPub(four_dof_control=False)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()