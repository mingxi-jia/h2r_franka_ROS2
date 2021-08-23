from std_msgs.msg import String
from sensor_msgs.msg import JointState
import rospy
import numpy as np
from src.robotiq_gripper import Gripper
from src.utils.rpy_to_rot_vector import rpyToRotVector
# from src.tf_proxy import TFProxy
import src.utils.transformation as transformation

class UR5:
    def __init__(self, pick_offset=0.1, place_offset=0.1, place_open_pos=0):
        self.gripper = Gripper(True)
        self.gripper.reset()
        self.gripper.activate()
        self.pub = rospy.Publisher('/ur_hardware_interface/script_command', String, queue_size=10)
        self.home_joint_values = [-0.394566837941305, -2.294720474873678, 2.2986323833465576, -1.5763557592975062, -1.5696309248553675, -0.3957274595843714]

        self.joint_names_speedj = ['shoulder_pan_joint', 'shoulder_lift_joint',
                                   'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.joint_values = np.array([0] * 6)
        self.joints_sub = rospy.Subscriber("/joint_states", JointState, self.jointsCallback)

        self.pick_offset = pick_offset
        self.place_offset = place_offset

        self.holding_state = 0
        self.place_open_pos = place_open_pos

        # self.tf_proxy = TFProxy()

    # def getEEPose(self):
    #     rTe = self.tf_proxy.lookupTransform('base_link', 'ee_link')
    #     hTr = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    #     eTt = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]).dot(np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])).dot(np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    #     T = hTr.dot(rTe).dot(eTt)
    #     pos = T[:3, 3]
    #     rot = transformation.euler_from_matrix(T)
    #
    #     return np.concatenate((pos, rot))

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
            rospy.sleep(0.2)
            if np.allclose(prev_joint_position, self.joint_values, atol=1e-3):
                break

    def moveToJ(self, joint_pos):
        for _ in range(1):
            s = 'movej({}, v=2)'.format(joint_pos)
            rospy.sleep(0.2)
            self.pub.publish(s)
            self.waitUntilNotMoving()
            if np.allclose(joint_pos, self.joint_values, atol=1e-2):
                return


    def moveToP(self, x, y, z, rx, ry, rz):
        # rz = np.pi/2 + rz

        # z -= 0.15
        # T = transformation.euler_matrix(rx, ry, rz)
        # pos = np.array([x, y, z])
        # pos += 0.15 * T[:3, 2]
        # x, y, z = pos

        rx, ry, rz = rpyToRotVector(rx, ry, rz)
        pose = [x, y, z, rx, ry, rz]
        for _ in range(1):
            s = 'movel(p{}, v=0.5)'.format(pose)
            rospy.sleep(0.2)
            self.pub.publish(s)
            self.waitUntilNotMoving()
            # if np.allclose(self.getEEPose()[:3], pose[:3], atol=1e-2) and np.allclose(self.getEEPose()[3:], pose[3:],  atol=1e-1):
            #     return

    def moveToHome(self):
        self.moveToJ(self.home_joint_values)

    def only_pick(self, x, y, z, r):
        if self.holding_state:
            return
        rx, ry, rz = r
        # rz = np.pi/2 + rz
        T = transformation.euler_matrix(rx, ry, rz)
        pre_pos = np.array([x, y, z])
        # pre_pos += self.pick_offset * T[:3, 2]
        pre_pos[2] += self.pick_offset

        self.moveToP(*pre_pos, rx, ry, rz)
        self.moveToP(x, y, z, rx, ry, rz)
        self.gripper.closeGripper(force=1)
        rospy.sleep(0.5)
        self.holding_state = 1
        if self.gripper.isClosed():
            self.gripper.openGripper()
            self.holding_state = 0
        self.moveToP(*pre_pos, rx, ry, rz)

    def pick(self, x, y, z, r):
        self.only_pick(x, y, z, r)
        self.moveToHome()

    def only_place(self, x, y, z, r, return_isClosed=False):
        if not self.holding_state:
            return
        rx, ry, rz = r
        # rz = np.pi/2 + rz
        T = transformation.euler_matrix(rx, ry, rz)
        pre_pos = np.array([x, y, z])
        # pre_pos += self.pick_offset * T[:3, 2]
        pre_pos[2] += self.place_offset
        self.moveToP(*pre_pos, rx, ry, rz)
        self.moveToP(x, y, z, rx, ry, rz)
        if return_isClosed:
            isClosed = self.gripper.isClosed()
        self.gripper.openGripper(speed=100, position=self.place_open_pos)
        rospy.sleep(0.5)
        self.holding_state = 0
        self.moveToP(*pre_pos, rx, ry, rz)
        self.gripper.openGripper()
        if return_isClosed:
            return isClosed

    def place(self, x, y, z, r):
        self.only_place(x, y, z, r)
        self.moveToHome()


if __name__ == '__main__':
    rospy.init_node('ur5')
    ur5 = UR5()
    ur5.moveToHome()
    ur5.moveToP(-0.330, -0.02, 0.149, 0, 0, 0)
    ur5.moveToP(-0.330, -0.02, 0.149, 0, np.pi/4, 0)
    ur5.moveToP(-0.330, -0.02, 0.149, np.pi/4, 0, 0)
    ur5.moveToP(-0.330, -0.02, 0.149, 0, 0, np.pi/4)
    ur5.moveToP(-0.330, -0.02, 0.149, 0, 0, np.pi/2)
    ur5.moveToP(-0.527, -0.02, 0.08, 0, 0, np.pi/4)
    ur5.moveToP(-0.527, -0.02, 0.08, 0, 0, np.pi/2)
    print(1)
    # ur5.moveToP(-0.26, -0.1, 0.3, -np.pi/6, np.pi/6, 0)
    # ur5.moveToP(-0.26, -0.1, 0.3, np.pi/6, np.pi/6, 0)
    # ur5.moveToP(-0.26, -0.1, 0.3, np.pi/6, -np.pi/6, 0)
    # ur5.moveToP(-0.26, -0.1, 0.3, np.pi/6, np.pi/6, 0)
    # ur5.moveToP(-0.26, -0.1, 0.5, 0.3848295, 0.3588603, 0.7143543)
    # ur5.moveToJ(ur5.home_joint_values)
