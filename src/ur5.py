from std_msgs.msg import String
from sensor_msgs.msg import JointState
import rospy
import numpy as np
from src.robotiq_gripper import Gripper

class UR5:
    def __init__(self):
        self.gripper = Gripper(True)
        self.gripper.reset()
        self.gripper.activate()
        self.pub = rospy.Publisher('/ur_hardware_interface/script_command', String, queue_size=10)
        self.home_joint_values = [-0.03911955, -2.14088709, 2.0448904, -1.4727586, -1.55525238, -1.61120922]

        self.joint_names_speedj = ['shoulder_pan_joint', 'shoulder_lift_joint',
                                   'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.joint_values = np.array([0] * 6)
        self.joints_sub = rospy.Subscriber("/joint_states", JointState, self.jointsCallback)

        self.pick_offset = 0.1
        self.place_offset = 0.1

    def jointsCallback(self, msg):
        '''Callback function for the joint_states ROS topic.'''

        def jointStateToDict(msg):
            '''Convert JointState message to dictionary.'''
            dict_out = {}
            for i in range(len(msg.position)):
                dict_out[msg.name[i]] = msg.position[i]

            return dict_out
        positions_dict = jointStateToDict(msg)
        self.joint_values = np.array([positions_dict[i] for i in self.joint_names_speedj])

    def waitUntilNotMoving(self):
        while True:
            prev_joint_position = self.joint_values.copy()
            rospy.sleep(0.5)
            print(self.joint_values)
            if np.allclose(prev_joint_position, self.joint_values, atol=1e-3):
                break

    def moveToJ(self, joint_pos):
        s = 'movej({})'.format(joint_pos)
        rospy.sleep(0.5)
        self.pub.publish(s)
        self.waitUntilNotMoving()


    def moveToP(self, x, y, z, rx, ry, rz):
        rz = -np.pi/2 + rz
        pose = [x, y, z, rx, ry, rz]
        s = 'movel(p{})'.format(pose)
        rospy.sleep(0.5)
        self.pub.publish(s)
        self.waitUntilNotMoving()

    def moveToHome(self):
        self.moveToJ(self.home_joint_values)

    def pick(self, x, y, z, r):
        self.moveToP(x, y, z+self.pick_offset, 0, 0, r)
        self.moveToP(x, y, z, 0, 0, r)
        self.gripper.closeGripper(force=5)
        self.moveToP(x, y, z+self.pick_offset, 0, 0, r)
        self.moveToHome()

    def place(self, x, y, z, r):
        self.moveToP(x, y, z+self.place_offset, 0, 0, r)
        self.moveToP(x, y, z, 0, 0, r)
        self.gripper.openGripper(100)
        self.moveToP(x, y, z+self.place_offset, 0, 0, r)
        self.moveToHome()


if __name__ == '__main__':
    rospy.init_node('ur5')
    ur5 = UR5()
    # ur5.moveToP(-0.26, -0.1, 0.3, 0, 0, 0)
    ur5.moveToJ(ur5.home_joint_values)
