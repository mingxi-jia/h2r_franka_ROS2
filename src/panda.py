import rospy
import moveit_commander
import moveit_msgs.msg
import sys
from geometry_msgs.msg import Pose, PoseStamped
import tf.transformations

BASE_OFFSET = 0.1
CAMERA_OFFSET = -0.12

class PandaArmControl:
    def __init__(self):

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.init_scene()
        self.group_name = "panda_arm"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

    def init_scene(self):
        rospy.sleep(0.2)
        
        table_pose = PoseStamped()
        table_name = "table"
        table_pose.header.frame_id = "panda_link0"
        table_pose.pose.position.x = 0.5
        table_pose.pose.position.y = 0.
        table_pose.pose.position.z = -0.11
        table_pose.pose.orientation.w = 1.0
        self.scene.add_box(table_name, table_pose, size=(1.8, 1.8, 0.2169))

        pole2_pose = PoseStamped()
        pole2_name = "pole_for_kevin"
        pole2_pose.header.frame_id = "panda_link0"
        pole2_pose.pose.position.x = 0.9
        pole2_pose.pose.position.y = -0.03
        pole2_pose.pose.position.z = 0.562
        pole2_pose.pose.orientation.w = 1.0
        self.scene.add_box(pole2_name, pole2_pose, size=(0.06, 0.06, 1.25))

        pole1_pose = PoseStamped()
        pole1_name = "pole_for_bob"
        pole1_pose.header.frame_id = "panda_link0"
        pole1_pose.pose.position.x = 0.565
        pole1_pose.pose.position.y = 0.61
        pole1_pose.pose.position.z = 0.5
        pole1_pose.pose.orientation.w = 1.0
        self.scene.add_box(pole1_name, pole1_pose, size=(0.06, 0.06, 1.02))

        pole3_pose = PoseStamped()
        pole3_name = "stick_for_kevin"
        pole3_pose.header.frame_id = "panda_link0"
        pole3_pose.pose.position.x = 0.668
        pole3_pose.pose.position.y = -0.03
        pole3_pose.pose.position.z = 1.11
        pole3_pose.pose.orientation.w = 1.0
        self.scene.add_box(pole3_name, pole3_pose, size=(0.465, 0.06, 0.06))

        pole4_pose = PoseStamped()
        pole4_name = "pole_for_stuart"
        pole4_pose.header.frame_id = "panda_link0"
        pole4_pose.pose.position.x = 0.23
        pole4_pose.pose.position.y = -0.53
        pole4_pose.pose.position.z = 0.5
        pole4_pose.pose.orientation.w = 1.0
        self.scene.add_box(pole4_name, pole4_pose, size=(0.06, 0.06, 1.02))
        
    def move_gripper_to_pose(self, x, y, z, rx, ry, rz):
        pose_target = Pose()
        pose_target.position.x = y
        pose_target.position.y = x
        pose_target.position.z = z + BASE_OFFSET
        
        quaternion = self.quaternion_from_euler(rx, ry, rz)
        pose_target.orientation.x = quaternion[0]
        pose_target.orientation.y = quaternion[1]
        pose_target.orientation.z = quaternion[2]
        pose_target.orientation.w = quaternion[3]
        
        self.move_group.set_pose_target(pose_target)
        plan = self.move_group.go(wait=True)
        
        self.move_group.stop()
        self.move_group.clear_pose_targets()
    
    def move_camera_to_pose(self, x, y, z, rx, ry, rz):
        pose_target = Pose()
        pose_target.position.x = y + CAMERA_OFFSET
        pose_target.position.y = x
        pose_target.position.z = z
        
        quaternion = self.quaternion_from_euler(rx, ry, rz)
        pose_target.orientation.x = quaternion[0]
        pose_target.orientation.y = quaternion[1]
        pose_target.orientation.z = quaternion[2]
        pose_target.orientation.w = quaternion[3]
        
        self.move_group.set_pose_target(pose_target)
        plan = self.move_group.go(wait=True)
        
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        
    def quaternion_from_euler(self, roll, pitch, yaw):
        return tf.transformations.quaternion_from_euler(roll, pitch, yaw)

if __name__ == "__main__":

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface', anonymous=True)
    panda_arm = PandaArmControl()
    panda_arm.move_gripper_to_pose(0.1, 0.2, 0.5, 3.14, 0, -0.8) # home position
    # panda_arm.move_gripper_to_pose(0.4, 0.1, 0.3, 3.14, 0, 2.4)  # Example target pose
    # panda_arm.move_gripper_to_pose(0.6, -0.1, 0.007, 3.14, 0, 2.4)
