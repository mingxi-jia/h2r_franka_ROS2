
from execution import Executor
import time
import rclpy
import numpy as np
from scipy.spatial.transform import Rotation

from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped
class MarkerPublisher(Node):

    def __init__(self):
        super().__init__('marker_publisher')
        self.publisher_ = self.create_publisher(PoseStamped, 'grasp_vizualization', 1)
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



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--transform',   required=True, type=list, help='Final end effector transform')
    # parser.add_argument('--parent_mesh',  required=True, type=str, help="filename of parent object mesh")
    # parser.add_argument('--child_mesh',  required=True, type=str, help="filename of child object mesh")

    rclpy.init(args=None)
    panda = Executor('mel')
    marker_publisher = MarkerPublisher()
    try:
        grasp_mat = np.array([[-0.01545242, -0.9881252,   0.15287187,  0.6008789 ],
                        [ 0.13703788, -0.1535403,  -0.97859389, -0.03792574],
                        [ 0.99044527,  0.00582759,  0.13778315,  0.07050603],
                        [ 0.,          0.,          0.,          1.        ]])
        
        fixed_grasp_transform  = Rotation.from_euler( 'yxz', [np.pi/2, np.pi/2, 0])
        grasp_rot =Rotation.from_matrix(np.matmul(fixed_grasp_transform.as_matrix(), grasp_mat[:3, :3],))

        grasp_transform = np.concatenate([grasp_mat[:3, 3:].T[0], grasp_rot.as_quat()])#make one up by moving the gripper a bit from where it currently is

        #TODO: add code for pregrasp transform step
        #TODO: add sequence of commands for grasp
        grasp_transform[2] += .3

        marker_publisher.publish_grasp_pose(grasp_transform)
    
        input("Check the plan - press enter to execute on the real robot")
        panda.move_robot(grasp_transform[:3], grasp_transform[3:], 0)
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


