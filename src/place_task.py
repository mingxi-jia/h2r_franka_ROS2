
from execution import Executor
import time
import rclpy
import numpy as np
from scipy.spatial.transform import Rotation

from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped
import copy

class PosePublisher(Node):
    def __init__(self):
        super().__init__('grasp_pose_publisher')
        self.publisher_ = self.create_publisher(PoseStamped, 'grasp_pose_vizualization', 1)
        self.i=0

    def publish_grasp_pose(self, pose):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'fr3_link0'

        # Set the pose position and orientation
        msg.pose.position.x = pose[0]
        msg.pose.position.y = pose[1]
        msg.pose.position.z = pose[2]

        msg.pose.orientation.x = pose[3]
        msg.pose.orientation.y = pose[4]
        msg.pose.orientation.z = pose[5]
        msg.pose.orientation.w = pose[6]

        self.publisher_.publish(msg)
        self.get_logger().info('Publishing pose')

def translateFrameNegativeZ(grasp_pose, dist):
    translated_pose = copy.deepcopy(grasp_pose)
    dist_to_move_in_cloud_frame = -1*np.matmul(grasp_pose, np.array([0, 0, dist, 0]))
    translated_pose[:3, 3] += dist_to_move_in_cloud_frame[:3]
    return translated_pose

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--transform',   required=True, type=list, help='Final end effector transform')
    # parser.add_argument('--parent_mesh',  required=True, type=str, help="filename of parent object mesh")
    # parser.add_argument('--child_mesh',  required=True, type=str, help="filename of child object mesh")

    rclpy.init(args=None)
    panda = Executor('mel')
    marker_publisher = PosePublisher()
    try:
        grasp_mat = np.array([[-0.05322813, -0.99808888, -0.03139037,  0.49457622],
                              [ 0.39897417,  0.0075606 , -0.916931  , -0.05650516],
                              [ 0.91541596, -0.06133046,  0.39780924,  0.07000888],
                              [ 0.        ,  0.        ,  0.        ,  1.        ]])
        dry_run = False


        # To adapt the grasps for this specific type of gripper
        fixed_grasp_transform  = Rotation.from_euler( 'yxz', [np.pi/2, np.pi/2, 0])
        grasp_rot = Rotation.from_matrix(np.matmul(fixed_grasp_transform.as_matrix(), grasp_mat[:3, :3],))
        grasp_mat[:3, :3] = grasp_rot.as_matrix()
        grasp_mat = translateFrameNegativeZ(grasp_mat, .02)
        pregrasp_mat = translateFrameNegativeZ(grasp_mat, .15)

        grasp_transform = np.concatenate([grasp_mat[:3, 3:].T[0], grasp_rot.as_quat()]) 
        pregrasp_transform = np.concatenate([pregrasp_mat[:3, 3:].T[0], grasp_rot.as_quat()]) 
        postgrasp_transform = np.concatenate([grasp_mat[:3, 3:].T[0], grasp_rot.as_quat()])
        postgrasp_transform[2] += .05 

        marker_publisher.publish_grasp_pose(pregrasp_transform)
        input("Check the pregrasp - press enter to execute on the real robot")
        if not dry_run:
            panda.move_robot(pregrasp_transform[:3], pregrasp_transform[3:], 1)
            for i in range(100):
                rclpy.spin_once(panda)

        marker_publisher.publish_grasp_pose(grasp_transform)
        input("Check the grasp - press enter to execute on the real robot")
        if not dry_run:
            future = panda.move_robot(grasp_transform[:3], grasp_transform[3:], 0)
            for i in range(100):
                rclpy.spin_once(panda)

        marker_publisher.publish_grasp_pose(postgrasp_transform)
        input("Going to postgrasp for segmentation- press enter to execute on the real robot")
       
        if not dry_run:
            panda.move_robot(postgrasp_transform[:3], postgrasp_transform[3:], 0)
            for i in range(100):
                rclpy.spin_once(panda)
        
        
        rclpy.spin(panda)
    except KeyboardInterrupt:
        panda.get_logger().info("Shutting down.")
    finally:
        panda.destroy_node()
        marker_publisher.destroy_node()
        rclpy.shutdown()