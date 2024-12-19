
from execution import Executor
import time
import rclpy
import numpy as np
from scipy.spatial.transform import Rotation

from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped
import copy
from place_task import PosePublisher

def pos_quat_to_mat(pos, quat):
    transform = np.zeros((4,4))
    transform[:3, :3] = Rotation.from_quat(quat).as_matrix()
    transform[:3, 3] = pos
    transform[3,3] = 1
    return transform

if __name__ == "__main__":
    rclpy.init(args=None)
    panda = Executor('mel')
    marker_publisher = PosePublisher()

    try:
        inferred_transform = #?
        dry_run = False

        ee_pose = pos_quat_to_mat(panda.current_pose.position, panda.current_pose.orientation)
        final_transform = np.matmul(inferred_transform, ee_pose)
        marker_publisher.publish_grasp_pose(final_transform)
        input("Check the pregrasp - press enter to execute on the real robot")
        if not dry_run:
            panda.move_robot(final_transform[:3], final_transform[3:], 1)
            for i in range(100):
                rclpy.spin_once(panda)
        
        rclpy.spin(panda)
    except KeyboardInterrupt:
        panda.get_logger().info("Shutting down.")
    finally:
        panda.destroy_node()
        marker_publisher.destroy_node()
        rclpy.shutdown()