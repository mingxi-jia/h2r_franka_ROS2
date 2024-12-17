
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

# def visualizeGraspPoses(pcd, grasp_poses):
#     # Given a 4x4 transformation matrix, create coordinate frame mesh at the pose
#     #     and scale down.
#     def o3dTFAtPose(pose, scale_down=10):
#         axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
#         scaling_maxtrix = np.ones((4,4))
#         scaling_maxtrix[:3, :3] = scaling_maxtrix[:3, :3]/scale_down
#         scaled_pose = pose*scaling_maxtrix
#         axes.transform(scaled_pose)
#         return axes
#     world_frame_axes = o3dTFAtPose(np.eye(4))
#     models = [world_frame_axes, pcd]
#     for grasp_pose in grasp_poses:
#         grasp_axes = o3dTFAtPose(grasp_pose, scale_down=100)
#         models.append(grasp_axes)

#     o3d.visualization.draw_geometries(models)



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--transform',   required=True, type=list, help='Final end effector transform')
    # parser.add_argument('--parent_mesh',  required=True, type=str, help="filename of parent object mesh")
    # parser.add_argument('--child_mesh',  required=True, type=str, help="filename of child object mesh")

    rclpy.init(args=None)
    panda = Executor('mel')
    marker_publisher = PosePublisher()
    try:
        grasp_mat = np.array([[-0.14387667, -0.94549284, -0.2921349,  0.61160968],
                             [-0.16334109,  0.31384691, -0.93531802, -0.13871732],
                             [ 0.97602213, -0.08685281, -0.19959308,  0.0693942 ],
                             [ 0.,          0.,          0. ,         1.        ]])


        

        # To adapt the grasps for this specific type of gripper
        fixed_grasp_transform  = Rotation.from_euler( 'yxz', [np.pi/2, np.pi/2, 0])
        grasp_rot = Rotation.from_matrix(np.matmul(fixed_grasp_transform.as_matrix(), grasp_mat[:3, :3],))
        grasp_mat[:3, :3] = grasp_rot.as_matrix()
        grasp_mat = translateFrameNegativeZ(grasp_mat, .05)
        pregrasp_mat = translateFrameNegativeZ(grasp_mat, .15)

        grasp_transform = np.concatenate([grasp_mat[:3, 3:].T[0], grasp_rot.as_quat()]) 
        pregrasp_transform = np.concatenate([pregrasp_mat[:3, 3:].T[0], grasp_rot.as_quat()]) 
        postgrasp_transform = np.concatenate([grasp_mat[:3, 3:].T[0], grasp_rot.as_quat()])
        postgrasp_transform[2] += .05 

        marker_publisher.publish_grasp_pose(pregrasp_transform)
        input("Check the pregrasp - press enter to execute on the real robot")
        panda.move_robot(pregrasp_transform[:3], pregrasp_transform[3:], 1)
        for i in range(10):
            rclpy.spin_once()

        marker_publisher.publish_grasp_pose(grasp_transform)
        input("Check the grasp - press enter to execute on the real robot")

        future = panda.move_robot(grasp_transform[:3], grasp_transform[3:], 0)
        for i in range(10):
            rclpy.spin_once()


        # input("Going to postgrasp for segmentation- press enter to execute on the real robot")
        # panda.move_robot(postgrasp_transform[:3], postgrasp_transform[3:], 0)
        # rclpy.spin_once(panda)

        
        rclpy.spin(panda)
    except KeyboardInterrupt:
        panda.get_logger().info("Shutting down.")
    finally:
        panda.destroy_node()
        marker_publisher.destroy_node()
        rclpy.shutdown()
    

    #load planning scene urdf 
    # with planning_scene_monitor.read_write() as scene:
    #     parent_collision_object = CollisionObject()
    #     parent_collision_object.header.frame_id = "panda_link0"
    #     parent_collision_object.id = "parent"

    #     parent_pose = Pose()
    #     mesh_pose_in_world = #Get the origin of the mesh in the world frame
    #     parent_pose.position.x = mesh_pose_in_world[0]
    #     parent_pose.position.y = mesh_pose_in_world[1]
    #     parent_pose.position.z = mesh_pose_in_world[2]
    #     parent_pose.orientation.x = mesh_pose_in_world[3]
    #     parent_pose.orientation.y = mesh_pose_in_world[4]
    #     parent_pose.orientation.z = mesh_pose_in_world[5]
    #     parent_pose.orientation.w = mesh_pose_in_world[6]

    #     box = Mesh()
    #     #parameters?

    #     parent_collision_object.primitives.append(box)
    #     parent_collision_object.primitive_poses.append(box_pose)
    #     parent_collision_object.operation = CollisionObject.ADD

    #     scene.apply_collision_object(collision_object)

    #     child_collision_object = CollisionObject()
    #     child_collision_object.header.frame_id = "panda_link0"
    #     child_collision_object.id = "child"

    #     child_pose = Pose()
    #     mesh_pose_in_world = #Get the origin of the mesh in the world frame
    #     child_pose.position.x = mesh_pose_in_world[0]
    #     child_pose.position.y = mesh_pose_in_world[1]
    #     child_pose.position.z = mesh_pose_in_world[2]
    #     child_pose.orientation.x = mesh_pose_in_world[3]
    #     child_pose.orientation.y = mesh_pose_in_world[4]
    #     child_pose.orientation.z = mesh_pose_in_world[5]
    #     child_pose.orientation.w = mesh_pose_in_world[6]

    #     #add to the robot's end effector

    #     #add another one 
    #     scene.current_state.update()  # Important to ensure the scene is updated

    # #Visualize the planning scene

    # input("Check the planning scene - press enter to continue")

    # #Visualize the plan

    #run rviz to see the plan? see if there's a way to check the grasp in rviz before hand?
    

    #input("Record final state - press enter to finish")


