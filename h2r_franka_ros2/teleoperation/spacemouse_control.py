import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import String, Bool
from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped, PoseStamped
from franka_msgs.action import Grasp, Homing, Move
from franka_msgs.msg import GraspEpsilon

from scipy.spatial.transform import Rotation as R
import numpy as np

class SpacemouseServoPub(Node):

    def __init__(self, four_dof_control):
        super().__init__('spacemouse_servo_pub')
        self.four_dof_control = four_dof_control
        control_mode = "4DoF" if four_dof_control else "6DoF"
        self.get_logger().info(f'Using {control_mode} control')

        self.joy_subscription = self.create_subscription(Joy, '/joy', self.listener_callback, 50)
        self.joy_subscription  # prevent unused variable warning
        self.ee_pose_subscription = self.create_subscription(PoseStamped, '/franka_robot_state_broadcaster/current_pose', self.ee_pose_callback, 50)
        self.ee_pose_subscription  # prevent unused variable warning
        self.current_ee_pose: PoseStamped = None
        self.current_ee_pose_prev: PoseStamped = None

        self.grasp_subscription = self.create_subscription(Joy, '/servo_node/grasp_cmds', self.grasp_callback, 50)
        self.grasp_subscription  # prevent unused variable warning

        self.twist_publisher_ = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 50)
        self.get_logger().info(f'spacemouse_servo_pub ready!')

        self.grasp_publisher_ = self.create_publisher(Joy, '/spacemouse/grasp_signals', 50)
        self.get_logger().info(f'spacemouse_servo_pub ready!')
        
        self.grasp_client = ActionClient(self, Grasp, '/fr3_gripper/grasp')
        self.grasp_client.wait_for_server()
        self.get_logger().info(f'grasp agent ready!')

        self.homing_client = ActionClient(self, Move, '/fr3_gripper/move')
        self.homing_client.wait_for_server()
        self.get_logger().info(f'homing agent ready!')

        self.grasp_action_triggered = False
        self.homing_action_triggered = False
        self.remote_grasp_triggered = False
        self.latest_xyzrpy = [0.,0.,0.,0.,0.,0.]

        self.speed = 0.05

        publish_freq = 1 / 50
        self.timer = self.create_timer(publish_freq, self.time_callback)
    
    def time_callback(self):
        x, y, z, rx, ry, rz = self.latest_xyzrpy
        # need to change this part so turning on False would work
        dz = 0.
        if self.four_dof_control:
            # rx, ry = 0., 0.
            dz, rx, ry = self.compensate_rx_ry(x, y, z, rz)
        msg = TwistStamped()
        stamp = self.get_clock().now().to_msg()
        msg.header.stamp = stamp
        msg.header.frame_id = 'fr3_link0'
        msg.twist.linear.x = x
        msg.twist.linear.y = y
        msg.twist.linear.z = z + dz
        msg.twist.angular.x = rx
        msg.twist.angular.y = ry
        msg.twist.angular.z = rz
        self.twist_publisher_.publish(msg)


        grasp_msg = Joy()
        grasp_msg.header.stamp = stamp
        grasp_msg.buttons = [1] if self.grasp_action_triggered else [0]
        self.grasp_publisher_.publish(grasp_msg)
        self.get_logger().info(f'Published: "{msg.twist}"')
        
            
    def ee_pose_callback(self, msg: PoseStamped):
        self.current_ee_pose = msg

    def listener_callback(self, msg: Joy):
        if (msg is None) or (self.current_ee_pose is None):
            return 
        orig_x, orig_y, orig_z, orig_rx, orig_ry, orig_rz, = msg.axes
        button_grasp, button_homing = msg.buttons
        # panda frame
        xyzrpy = [-orig_y, -orig_x, orig_z, -orig_ry, -orig_rx, orig_rz]
        
        self.latest_xyzrpy = xyzrpy
        
        # Trigger Grasp action on button press
        if button_grasp == 1 and not self.grasp_action_triggered:
            self.grasp_action_triggered = True
            self.send_grasp_goal()

        # Reset Grasp action state on button release
        if button_homing == 1 and self.grasp_action_triggered:
            self.grasp_action_triggered = False

        # Trigger Homing action on button press
        if button_homing == 1 and not self.homing_action_triggered:
            self.homing_action_triggered = True
            self.send_homing_goal()

        # Reset Homing action state on button release
        if button_grasp == 1 and self.homing_action_triggered:
            self.homing_action_triggered = False
    
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
            # rx, ry = 0., 0.
            dz, rx, ry = self.compensate_rx_ry(x, y, z, rz)
        msg = TwistStamped()
        stamp = self.get_clock().now().to_msg()
        msg.header.stamp = stamp
        msg.header.frame_id = 'fr3_link0'
        msg.twist.linear.x = x
        msg.twist.linear.y = y
        msg.twist.linear.z = z + dz
        msg.twist.angular.x = rx
        msg.twist.angular.y = ry
        msg.twist.angular.z = rz
        self.twist_publisher_.publish(msg)


        grasp_msg = Joy()
        grasp_msg.header.stamp = stamp
        grasp_msg.buttons = [1] if self.grasp_action_triggered else [0]
        self.grasp_publisher_.publish(grasp_msg)
        self.get_logger().info(f'Published: "{msg.twist}"')
        
    def compensate_rx_ry(self, x, y, z, yaw):
        goal_euler = np.array([-np.pi, 0., yaw])
        goal_R = R.from_euler('xyz', goal_euler).as_matrix()

        current_quat = [self.current_ee_pose.pose.orientation.x, self.current_ee_pose.pose.orientation.y, self.current_ee_pose.pose.orientation.z, self.current_ee_pose.pose.orientation.w]
        current_R = R.from_quat(current_quat).as_matrix()
        diff_R = current_R.T @ goal_R
        diff_euler = R.from_matrix(diff_R).as_euler('xyz')
        print(diff_euler)

        compensate_threshold = 0.12

        rx, ry, rz = diff_euler[0], diff_euler[1], diff_euler[2]
        rx = 0. if abs(rx) < compensate_threshold else rx
        ry = 0. if abs(ry) < compensate_threshold else ry
        rz = 0. if abs(rz) < compensate_threshold else rz
        if abs(rx) > compensate_threshold or abs(ry) > compensate_threshold:
            dz = 0.15
        else:
            dz = 0.
        return dz, rx*1.5, -ry*1.5


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