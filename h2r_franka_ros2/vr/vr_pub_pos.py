# in this script we need to broadcast the transformation matrix that we gathered
# from the droid folder reader.py to the /my_pose topic

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
import threading
# from reader import OculusReader  # Import the OculusReader class
from droid.oculus_reader.oculus_reader import OculusReader
from droid.oculus_controller import VRPolicy
from sensor_msgs.msg import JointState  
from scipy.spatial.transform import Rotation as R
import time

class VRPublisher(Node):
    def __init__(self):
        super().__init__('vr_publisher')
        self.vr_policy = VRPolicy(rmat_reorder=[2, 1, -3, 4])
        # ROS 2 publisher for /my_pose topic
        self.pose_pub = self.create_publisher(\
            PoseStamped, 
            '/my_pose', 
            10)

        #subscribers
        self.cartesian_pose_sub = self.create_subscription(
            PoseStamped,
            '/franka_robot_state_broadcaster/current_pose',
            self.cartesian_pose_callback,
            10)

        self.gripper_state_sub = self.create_subscription(
            JointState,
            '/franka_gripper/joint_states',
            self.gripper_state_callback,
            10)

        self.current_robot_state = {
            "cartesian_position": [None, None, None, None, None, None],  # Cartesian pose placeholder
            "gripper_position": 0.03953560069203377,  # Gripper state placeholder
        }

        self._logger.info('Initializing VRPublisher')

        # Start the timer to periodically check for new transforms
        self.timer = self.create_timer(0.05, self.publish_vr_state)  # Publish at 10 Hz

        self.get_logger().info('VR Publisher Node initialized.')


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


    def publish_vr_state(self):
            # Ensure robot state is available
        if None in self.current_robot_state["cartesian_position"]:
            self.get_logger().info("No franka robot state available. Waiting.")
            return

        # Use VRPolicy's forward function with the current robot state
        action, info = self.vr_policy.forward({"robot_state": self.current_robot_state}, include_info=True)
        if info == {}:
            self.get_logger().info("No goal state available. Waiting.")
            return
        # if info != {}:
        #     print(time.time(), info["target_cartesian_position"])

        # if np.allclose(action, 0.0):
        #     self.get_logger().info("Default action detected. Skipping publishing.")
        #     return

        # Publish the calculated pose based on VR action
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "fr3_link0"
        pose_msg.pose.position.x = info["target_cartesian_position"][0]
        pose_msg.pose.position.y = info["target_cartesian_position"][1]
        pose_msg.pose.position.z = info["target_cartesian_position"][2]
        target_euler = info["target_cartesian_position"][3:]
        rx, ry, rz = target_euler
        x, y, z, w = R.from_euler('xyz', [-rx, -ry, rz]).as_quat()
        pose_msg.pose.orientation.x = x
        pose_msg.pose.orientation.y = y
        pose_msg.pose.orientation.z = z
        pose_msg.pose.orientation.w = w

        self.pose_pub.publish(pose_msg)
        self.get_logger().info(f"Published VR State: {pose_msg.pose}")

    

def main(args=None):
    rclpy.init(args=args)
    vr_publisher = VRPublisher()

    try:
        rclpy.spin(vr_publisher)
    except KeyboardInterrupt:
        vr_publisher.get_logger().info('Shutting down VR Publisher...')
    finally:
        vr_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

# class VRPublisher(Node):
#     def __init__(self):
#         super().__init__('vr_publisher')

#         # ROS 2 publisher for /my_pose topic
#         self.publisher_ = self.create_publisher(PoseStamped, '/my_pose', 10)
#         self._logger.info('Initializing VRPublisher')

#         # Initialize OculusReader for VR data
#         self.oculus_reader = OculusReader(run=False)  # Run manually in a separate thread
#         self.reader_thread = threading.Thread(target=self.oculus_reader.run)
#         self.reader_thread.start()

#         # Start the timer to periodically check for new transforms
#         self.timer = self.create_timer(0.1, self.publish_pose)  # Publish at 10 Hz

#         self.get_logger().info('VR Publisher Node initialized.')

#     def publish_pose(self):
#         """Publish the transformation matrix as a PoseStamped message."""
#         transforms, buttons = self.oculus_reader.get_transformations_and_buttons()

#         if transforms and 'r' in transforms:  # Check for the 'r' transform (right hand)
#             matrix = transforms['r']  # Use the 'r' matrix for publishing
#             pose_msg = self.transform_to_pose(matrix)

#             # Publish the PoseStamped message
#             self.publisher_.publish(pose_msg)
#             self.get_logger().info(f'Published Pose: {pose_msg}')

#     def transform_to_pose(self, matrix):
#         """Convert a 4x4 transformation matrix to a PoseStamped message."""

#         # now we need to compare the difference in axes between the headset and the 
#         # robot gripper


#         position = matrix[:3, 3]  # Extract translation
#         rotation_matrix = matrix[:3, :3]  # Extract rotation matrix
#         quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)

#         pose_msg = PoseStamped()
#         pose_msg.header.stamp = self.get_clock().now().to_msg()
#         pose_msg.header.frame_id = 'base'  # Set the frame_id appropriately
#         pose_msg.pose.position.x = position[0]
#         pose_msg.pose.position.y = position[1]
#         pose_msg.pose.position.z = position[2]
#         # orientation is quarternion indeed
#         pose_msg.pose.orientation.x = quaternion[0]
#         pose_msg.pose.orientation.y = quaternion[1]
#         pose_msg.pose.orientation.z = quaternion[2]
#         pose_msg.pose.orientation.w = quaternion[3]

#         return pose_msg

#     @staticmethod
#     def rotation_matrix_to_quaternion(matrix):
#         """Convert a 3x3 rotation matrix to a quaternion."""
#         # trace = np.trace(matrix)

#         trace = matrix[0,0] + matrix[1,1] + matrix[2,2] + 1
#         if trace > 0:
#             s = 0.5 / np.sqrt(trace)
#             w = 0.25 / s 
#             x = (matrix[2, 1] - matrix[1, 2]) * s
#             y = (matrix[0, 2] - matrix[2, 0]) * s
#             z = (matrix[1, 0] - matrix[0, 1]) * s
#         elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
#             s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
#             w = (matrix[2, 1] - matrix[1, 2]) / s
#             x = 0.25 * s
#             y = (matrix[0, 1] + matrix[1, 0]) / s
#             z = (matrix[0, 2] + matrix[2, 0]) / s
#         elif matrix[1, 1] > matrix[2, 2]:
#             s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
#             w = (matrix[0, 2] - matrix[2, 0]) / s
#             x = (matrix[0, 1] + matrix[1, 0]) / s
#             y = 0.25 * s
#             z = (matrix[1, 2] + matrix[2, 1]) / s
#         else:
#             s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
#             w = (matrix[1, 0] - matrix[0, 1]) / s
#             x = (matrix[0, 2] + matrix[2, 0]) / s
#             y = (matrix[1, 2] + matrix[2, 1]) / s
#             z = 0.25 * s
#         return np.array([x, y, z, w])

#     def destroy_node(self):
#         """Stop the OculusReader thread when shutting down."""
#         self.oculus_reader.stop()
#         self.reader_thread.join()
#         super().destroy_node()


# def main(args=None):
#     rclpy.init(args=args)
#     vr_publisher = VRPublisher()

#     try:
#         rclpy.spin(vr_publisher)
#     except KeyboardInterrupt:
#         vr_publisher.get_logger().info('Shutting down VR Publisher...')
#     finally:
#         vr_publisher.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()
