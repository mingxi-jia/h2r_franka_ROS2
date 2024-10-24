import rospy
import moveit_commander
import numpy as np
import sys
from geometry_msgs.msg import Pose, PoseStamped
import tf.transformations
from franka_gripper.msg import MoveActionGoal
from scipy.spatial.transform import Rotation as R

GRIPPER_OFFSET = 0.1
CAMERA_OFFSET = -0.12
HOME_POSE = [-0.09838453584596964, -0.944262178057893, 0.22341715258813172, 
             -2.0975801058148966, 0.19633714461031365, 1.1709991560687036, 
             0.9094869062085559]
TIM_TO_TCP = [0.048, -0.012, -0.075, 0.004, -0.003, -0.707, 0.707]

class PandaArmControl:
    def __init__(self):

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "panda_arm"
        self.init_scene()
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.gripper_pub = rospy.Publisher('/franka_gripper/move/goal', MoveActionGoal, queue_size=10)

    def init_scene(self):
        rospy.sleep(0.2)
        
        table_pose = PoseStamped()
        table_name = "table"
        table_pose.header.frame_id = "panda_link0"
        table_pose.pose.position.x = 0.5
        table_pose.pose.position.y = 0.
        table_pose.pose.position.z = -0.11
        table_pose.pose.orientation.w = 1.0
        self.scene.add_box(table_name, table_pose, size=(1.8, 1.8, 0.233))

        pole2_pose = PoseStamped()
        pole2_name = "pole_for_kevin"
        pole2_pose.header.frame_id = "panda_link0"
        pole2_pose.pose.position.x = 0.85
        pole2_pose.pose.position.y = -0.03
        pole2_pose.pose.position.z = 0.562
        pole2_pose.pose.orientation.w = 1.0
        self.scene.add_box(pole2_name, pole2_pose, size=(0.08, 0.08, 1.25))

        pole1_pose = PoseStamped()
        pole1_name = "pole_for_bob"
        pole1_pose.header.frame_id = "panda_link0"
        pole1_pose.pose.position.x = 0.565
        pole1_pose.pose.position.y = 0.56
        pole1_pose.pose.position.z = 0.5
        pole1_pose.pose.orientation.w = 1.0
        self.scene.add_box(pole1_name, pole1_pose, size=(0.2, 0.2, 1.02))

        pole3_pose = PoseStamped()
        pole3_name = "stick_for_kevin"
        pole3_pose.header.frame_id = "panda_link0"
        pole3_pose.pose.position.x = 0.668
        pole3_pose.pose.position.y = -0.03
        pole3_pose.pose.position.z = 1.11
        pole3_pose.pose.orientation.w = 1.0
        self.scene.add_box(pole3_name, pole3_pose, size=(0.55, 0.2, 0.2))

        pole4_pose = PoseStamped()
        pole4_name = "pole_for_stuart"
        pole4_pose.header.frame_id = "panda_link0"
        pole4_pose.pose.position.x = 0.12
        pole4_pose.pose.position.y = -0.53
        pole4_pose.pose.position.z = 0.5
        pole4_pose.pose.orientation.w = 1.0
        self.scene.add_box(pole4_name, pole4_pose, size=(0.2, 0.2, 1.02))

        # wire = PoseStamped()
        # wire_name = "wire"
        # wire.header.frame_id = "panda_hand_tcp"
        # wire.pose.position.x = TIM_TO_TCP[0]+0.04
        # wire.pose.position.y = TIM_TO_TCP[1]
        # wire.pose.position.z = TIM_TO_TCP[2]
        # wire.pose.orientation.x = TIM_TO_TCP[3]
        # wire.pose.orientation.y = TIM_TO_TCP[4]
        # wire.pose.orientation.z = TIM_TO_TCP[5]
        # wire.pose.orientation.w = TIM_TO_TCP[6]
        # self.scene.add_box(wire_name, wire, size=(0.12, 0.04, 0.03))
        # self.scene.attach_box('panda_hand_tcp', wire_name)

        gripper = PoseStamped()
        gripper_name = "gripper"
        gripper.header.frame_id = "panda_hand_tcp"
        gripper.pose.position.x = 0
        gripper.pose.position.y = 0
        gripper.pose.position.z = 0.04
        self.scene.add_box(gripper_name, gripper, size=(0.04, 0.10, 0.06))
        self.scene.attach_box('panda_hand_tcp', gripper_name)

        wrist_cam_pose = PoseStamped()
        wrist_cam_name = "tim"
        wrist_cam_pose.header.frame_id = "panda_hand_tcp"
        wrist_cam_pose.pose.position.x = TIM_TO_TCP[0]
        wrist_cam_pose.pose.position.y = TIM_TO_TCP[1]
        wrist_cam_pose.pose.position.z = TIM_TO_TCP[2]
        wrist_cam_pose.pose.orientation.x = TIM_TO_TCP[3]
        wrist_cam_pose.pose.orientation.y = TIM_TO_TCP[4]
        wrist_cam_pose.pose.orientation.z = TIM_TO_TCP[5]
        wrist_cam_pose.pose.orientation.w = TIM_TO_TCP[6]
        self.scene.add_box(wrist_cam_name, wrist_cam_pose, size=(0.03, 0.035, 0.03))

        self.scene.attach_box('panda_link8', wrist_cam_name)
    
    def add_safe_guard(self):
        safe_guard_pose = PoseStamped()
        safe_guard_name = "safe_guard"
        safe_guard_pose.header.frame_id = "panda_link0"
        safe_guard_pose.pose.position.z = 0.05
        safe_guard_pose.pose.orientation.w = 1.0
        self.scene.add_box(safe_guard_name, safe_guard_pose, size=(1.8, 1.8, 0.05))
    
    def remove_safe_guard(self):
        self.scene.remove_world_object("safe_guard")
    
    def move_gripper_width(self, width):
        msg = MoveActionGoal()
        msg.goal.width = width
        msg.goal.speed = 0.1
        self.gripper_pub.publish(msg)
        
    def move_gripper_to_pose(self, x, y, z, rx, ry, rz):
        self.move_group.set_end_effector_link("panda_hand_tcp")
        pose_target = Pose()
        pose_target.position.x = x
        pose_target.position.y = y
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

    def transform_camera_to_tcp(self, x, y, z, rx, ry, rz):
        """
        Transform the camera position and orientation to the TCP position and orientation.

        Parameters:
        xyzrpy (list or np.array): Position and orientation of the camera [x, y, z, roll, pitch, yaw].
        relative_tcp_xyzrpy (list or np.array): Relative position and orientation of the TCP to the camera [x, y, z, roll, pitch, yaw].

        Returns:
        np.array: Position and orientation of the TCP [x, y, z, roll, pitch, yaw].
        """

        t_tim_to_tcp = np.array(TIM_TO_TCP[:3])
        r_tim_to_tcp = R.from_euler('xyz', TIM_TO_TCP[3:]).as_matrix()

        t_camera = np.array([x, y, z])
        r_camera = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
        
        # Hand's position and orientation
        r_tcp = r_camera @ r_tim_to_tcp
        t_tcp = r_camera @ t_tim_to_tcp + t_camera
        r_tcp = R.from_matrix(r_tcp).as_euler('xyz')

        tcp_xyzrpy = np.concatenate((t_tcp, r_tcp))
        return tcp_xyzrpy
    
    def move_camera_to_pose(self, x, y, z, rx, ry, rz):
        self.move_group.set_end_effector_link("tim")
        pose_target = Pose()
        pose_target.position.x = x
        pose_target.position.y = y
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
    
    def go_home(self):
        self.move_group.go(HOME_POSE, wait=True)

if __name__ == "__main__":

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface', anonymous=True)
    panda_arm = PandaArmControl()
    panda_arm.go_home()
    # panda_arm.move_gripper_to_pose(0.1, 0.2, 0.5, np.pi, 0, 0)
    # panda_arm.move_camera_to_pose(0.1, 0.2, 0.5, np.pi, 0, np.pi/2)
    # panda_arm.go_home()
    # panda_arm.move_gripper_to_pose(0.1, 0.2, 0.5, 3.14, 0, -0.8)
    # panda_arm.move_gripper_to_pose(0.4, 0.1, 0.3, 3.14, 0, 2.4)  # Example target pose
    # panda_arm.move_gripper_to_pose(0.6, -0.1, 0.007, 3.14, 0, 2.4)
