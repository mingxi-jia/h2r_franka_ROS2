from std_msgs.msg import String
from sensor_msgs.msg import JointState
from ur_dashboard_msgs.msg import SafetyMode
from tf2_msgs.msg import TFMessage
from std_srvs.srv import Trigger
import rospy
import numpy as np
from src.robotiq_gripper import Gripper
from src.utils.rpy_to_rot_vector import rpyToRotVector
# from src.tf_proxy import TFProxy
import src.utils.transformation as transformation
import time
from src.collision_detector import CollisionDetector

class UR5:
    def __init__(self, pick_offset=0.1, place_offset=0.1, place_open_pos=0):
        self.collision_detection = CollisionDetector(max_z=40)
        self.grasp_collision_detection = CollisionDetector(max_z=80)
        self.gripper = Gripper(True)
        self.gripper.reset()
        self.gripper.activate()
        self.pub = rospy.Publisher('/ur_hardware_interface/script_command', String, queue_size=10)
        self.home_joint_values = [-0.394566837941305, -2.294720474873678, 2.2986323833465576,
                                  -1.5763557592975062,-1.5696309248553675, -0.3957274595843714]
        self.bins_center_joint_values = [[-0.36723787, -1.71223528, 1.87438631,
                                          -1.71623451, -1.57518703, 1.21666324]]

        # Joint states subscriber
        self.joint_names_speedj = ['shoulder_pan_joint', 'shoulder_lift_joint',
                                   'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.joint_values = np.array([0] * 6)
        self.joints_sub = rospy.Subscriber("/joint_states", JointState, self.jointsCallback)

        # Tool position subscriber
        self.position_names = ['x', 'y', 'z']
        self.tool_position = np.array([0] * 3)
        self.tool_sub = rospy.Subscriber("/tf", TFMessage, self.tfCallback)

        self.pick_offset = pick_offset
        self.place_offset = place_offset

        self.holding_state = 0
        self.place_open_pos = place_open_pos

        # self.tf_proxy = TFProxy()
        self.safety_mode = None
        self.safety_mode_sub = rospy.Subscriber('/ur_hardware_interface/safety_mode', SafetyMode, self.safetyModeCallback)
        self.release_protective_stop = rospy.ServiceProxy('/ur_hardware_interface/dashboard/unlock_protective_stop', Trigger)
        self.collided = False
        self.collision_flag = False

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

    def tfCallback(self, msg):
        '''Callback function for the tf ROS topic.'''

        for i in range(len(msg.transforms)):
            if msg.transforms[i].child_frame_id == 'tool0_controller'\
                    and msg.transforms[0].header.frame_id == 'base':
                self.tool_position = np.array([msg.transforms[i].transform.translation.x,
                                               msg.transforms[i].transform.translation.y,
                                               msg.transforms[i].transform.translation.z])

    def safetyModeCallback(self, msg):
        self.safety_mode = msg.mode

    def waitUntilNotMoving(self):
        while True:
            if self.is_not_moving(sleep_time=0.2):
                break

    def waitUntilSlowMoving(self):
        while True:
            if self.is_not_moving(sleep_time=0.1):
                break

    def is_not_moving(self, threshold=1e-3, sleep_time=0.2):
        prev_joint_position = self.joint_values.copy()
        rospy.sleep(sleep_time)
        return np.allclose(prev_joint_position, self.joint_values, atol=threshold)

    def is_position_arrived(self, x, y, z, threshold=1e-3):
        return np.allclose(np.asarray([x, y, z]), self.tool_position, atol=threshold)

    def moveToJ(self, joint_pos, v=2):
        for _ in range(1):
            s = 'movej({}, v={})'.format(joint_pos, v)
            rospy.sleep(0.2)
            self.pub.publish(s)
            self.waitUntilNotMoving()
            if np.allclose(joint_pos, self.joint_values, atol=1e-2):
                return


    def moveToP(self, x, y, z, rx, ry, rz):

        rx, ry, rz = rpyToRotVector(rx, ry, rz)
        pose = [x, y, z, rx, ry, rz]
        for _ in range(1):
            s = 'movel(p{}, v=0.5)'.format(pose)
            # s = 'movel(p{}, v=0.25)'.format(pose)
            # rospy.sleep(0.1)
            self.pub.publish(s)
            self.waitUntilNotMoving()


    def moveToPBlend(self, x, y, z, rx, ry, rz, blend_r=0.05):

        rx, ry, rz = rpyToRotVector(rx, ry, rz)
        pose = [x, y, z, rx, ry, rz]
        for _ in range(1):
            s = 'movel(p{}, v=0.5)'.format(pose)
            # s = 'movel(p{}, v=0.25)'.format(pose)
            # rospy.sleep(0.1)
            self.pub.publish(s)
            rospy.sleep(0.1)
            self.waitUntilSlowMoving()


    def moveToPT(self, x, y, z, rx, ry, rz, t=5, t_wait_reducing=-0.1, with_collision_detection=False):
        a_with_cd = 0.3
        v_with_cd = 0.2
        self.collision_flag = False
        rx, ry, rz = rpyToRotVector(rx, ry, rz)
        if with_collision_detection:
            rospy.sleep(0.15)
            with self.collision_detection as cd:
                pose = [x, y, z, rx, ry, rz]
                is_sent = False
                while True:
                    s = 'movel(p{}, a={}, v={})'.format(pose, a_with_cd, v_with_cd)
                    if not is_sent:
                        self.pub.publish(s)
                        is_sent = True
                        # print('with z protection ', z)
                    rospy.sleep(0.1)

                    if self.is_position_arrived(x, y, z) or not cd.is_running:
                        self.collision_flag = self.collided = not cd.is_running
                        # print('arrive', self.is_position_arrived(x, y, z))
                        # print(cd.is_running)
                        break

            if self.safety_mode == 3:
                self.release_protective_stop()
                self.collision_flag = self.collided = True

        if self.collision_flag and with_collision_detection:
            # # If either collision or protecitve stop happened, lift z for 1 cm
            z = self.tool_position[2] + 0.01
            pose = [x, y, z, rx, ry, rz]
            s = 'movel(p{}, v=0.5, t={})'.format(pose, 0.5)
            # s = 'movel(p{}, v=0.25)'.format(pose)
            # rospy.sleep(0.1)
            self.pub.publish(s)
            # print('collision detected z = ', z)
            rospy.sleep(0.5)
            self.collision_flag = False

        if not with_collision_detection:
            pose = [x, y, z, rx, ry, rz]
            s = 'movel(p{}, t={})'.format(pose, t)
            # s = 'movel(p{}, v=0.25)'.format(pose)
            # rospy.sleep(0.1)
            self.pub.publish(s)
            rospy.sleep(t - t_wait_reducing)
            # self.waitUntilSlowMoving()


    def moveToHome(self):
        self.moveToJ(self.home_joint_values)


    def moveToBinCenter(self):
        self.moveToJ(self.bins_center_joint_values, v=5)


    def checkGripperState(self):
        if not self.gripper.hasObj(wait_speed0=True):
            self.gripper.openGripper()
            self.holding_state = 0
        else:
            self.holding_state = 1
        return self.holding_state

    def only_pick(self, x, y, z, r, check_gripper_close_when_pick=True):
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
        self.gripper.closeGripper(force=255)
        # rospy.sleep(0.5)
        self.holding_state = 1
        if check_gripper_close_when_pick:
            if self.gripper.hasObj():
                self.gripper.openGripper()
                self.holding_state = 0
        self.moveToP(*pre_pos, rx, ry, rz)
        if not check_gripper_close_when_pick:
            self.gripper.closeGripper()
            if self.gripper.hasObj():
                self.gripper.openGripper()
                self.holding_state = 0

    def only_pick_fast(self, x, y, z, r, check_gripper_close_when_pick=True):
        if self.holding_state:
            return
        rx, ry, rz = r
        # rz = np.pi/2 + rz
        T = transformation.euler_matrix(rx, ry, rz)
        pre_pos = np.array([x, y, z])
        # pre_pos += self.pick_offset * T[:3, 2]
        pre_pos[2] += self.pick_offset

        self.moveToPT(*pre_pos, rx, ry, rz, t=1)
        self.moveToPT(x, y, z, rx, ry, rz, with_collision_detection=True)
        with self.grasp_collision_detection as cd:
            self.gripper.closeGripper()
            # print('closing gripper')
            time_for_close = 8
            while True:
                for _ in range(time_for_close):
                    if not cd.is_running:
                        self.collision_flag = True

                    if self.safety_mode == 3:
                        self.gripper.openGripper()
                        print('protective stop triggered')
                        rospy.sleep(0.1)
                        self.release_protective_stop()
                        self.collision_flag = True
                        break

                    if not cd.is_running or self.safety_mode == 3:
                        break
                    rospy.sleep(0.1)
                # print('safety mode ', self.safety_mode)
                if self.collision_flag:
                    recovery_z = self.tool_position[2] + 0.015
                    # self.gripper.openGripper()
                    self.moveToPT(x, y, recovery_z, rx, ry, rz, t=0.2)
                    self.collision_flag = False
                    cd.is_running = True
                    time_for_close = 3
                else:
                    # print('grasp finished')
                    break

        self.holding_state = 1
        # if check_gripper_close_when_pick:
        #     if not self.gripper.hasObj():
        #         # self.gripper.openGripper()
        #         self.holding_state = 0
        self.moveToPT(*pre_pos, rx, ry, rz, t=0.4)
        # if not check_gripper_close_when_pick:
        #     self.gripper.closeGripper()
        #     if not self.gripper.hasObj():
        #         self.gripper.openGripper()
        #         self.holding_state = 0

    def pick(self, x, y, z, r):
        self.only_pick(x, y, z, r)
        self.moveToHome()

    def only_place(self, x, y, z, r, no_action_when_empty=True, move2_prepose=True):
        if (not self.holding_state) and no_action_when_empty:
            return
        rx, ry, rz = r
        # rz = np.pi/2 + rz
        T = transformation.euler_matrix(rx, ry, rz)
        pre_pos = np.array([x, y, z])
        # pre_pos += self.pick_offset * T[:3, 2]
        pre_pos[2] += self.place_offset
        if move2_prepose:
            self.moveToP(*pre_pos, rx, ry, rz)
        self.moveToP(x, y, z, rx, ry, rz)
        self.gripper.openGripper(position=self.place_open_pos)
        rospy.sleep(0.5)
        self.holding_state = 0
        if move2_prepose:
            self.moveToP(*pre_pos, rx, ry, rz)
        self.gripper.openGripper()

    def only_place_fast(self, x, y, z, r, no_action_when_empty=True, move2_prepose=True):
        if (not self.holding_state) and no_action_when_empty:
            return
        rx, ry, rz = r
        # rz = np.pi/2 + rz
        T = transformation.euler_matrix(rx, ry, rz)
        pre_pos = np.array([x, y, z])
        # pre_pos += self.pick_offset * T[:3, 2]
        pre_pos[2] += self.place_offset
        if move2_prepose:
            self.moveToP(*pre_pos, rx, ry, rz)
        self.moveToPT(x, y, z, rx, ry, rz, t=0.8, t_wait_reducing=0.1)
        # self.gripper.openGripper(position=self.place_open_pos)
        self.gripper.openGripper()
        rospy.sleep(0.3)
        self.holding_state = 0
        if move2_prepose:
            self.moveToP(*pre_pos, rx, ry, rz)
        # self.gripper.openGripper()

    def place(self, x, y, z, r):
        self.only_place(x, y, z, r)
        self.moveToHome()


if __name__ == '__main__':
    rospy.init_node('ur5')
    ur5 = UR5()
    ur5.moveToHome()
    ur5.gripper.closeGripper()
    ur5.moveToPT(-0.451975, 0.273, 0.04, 0, 0, 0.7853981852531433, t=2, t_wait_reducing=0.1)
    while True:
        ur5.moveToPT(-0.451975, 0.273, 0.125, 0, 0, 0.7853981852531433, t=1, t_wait_reducing=-0.1)
        # ur5.moveToPT(-0.451975, 0.273, -0.02, 0, 0, 0.7853981852531433, t=0.2, t_wait_reducing=0.05)
        print('prepose')
        ur5.only_pick_fast(-0.451975, 0.273, -0.075, (0, 0, 0.7853981852531433), check_gripper_close_when_pick=True)
        ur5.moveToPT(-0.451975, 0.273, -0.075, 0, 0, 0.7853981852531433, with_collision_detection=True)
    pick_place_offset = 0.1
    pick_place_height = 0.25

    # while True: # 10.74s
    #     time_mark = time.time()
    #     # pick
    #     ur5.moveToP(-0.434, -0.18, -0.1 + pick_place_offset, 0, 0, 0)
    #     ur5.moveToP(-0.434, -0.18, -0.1, 0, 0, 0)
    #     ur5.gripper.closeGripper()
    #     rospy.sleep(0.5)
    #     ur5.moveToP(-0.434, -0.18, -0.1 + pick_place_offset, 0, 0, 0)
    #     # move
    #     ur5.moveToP(-0.434, 0.045, 0.15, 0, 0, 0)
    #     # place
    #     ur5.moveToP(-0.434, 0.275, 0.05, 0, 0, 1.57)
    #     ur5.gripper.openGripper()
    #     rospy.sleep(0.5)
    #     # ready pose
    #     ur5.moveToP(-0.434, 0.045, 0.15, 0, 0, 0)
    #     print('time used {:.2f}s'.format(time.time() - time_mark))
    #     time_mark = time.time()

    # while True: # 8.94s frequent checking speed and then sending next waypoint
    #     time_mark = time.time()
    #     # pick
    #     ur5.moveToPBlend(-0.434, -0.18, -0.1 + pick_place_offset, 0, 0, 0)
    #     ur5.moveToP(-0.434, -0.18, -0.1, 0, 0, 0)
    #     ur5.gripper.closeGripper()
    #     rospy.sleep(0.5)
    #     ur5.moveToPBlend(-0.434, -0.18, -0.1 + pick_place_offset, 0, 0, 0)
    #     # move
    #     ur5.moveToPBlend(-0.434, 0.045, 0.15, 0, 0, 0)
    #     # place
    #     ur5.moveToP(-0.434, 0.275, 0.05, 0, 0, 1.57)
    #     ur5.gripper.openGripper()
    #     rospy.sleep(0.2)
    #     # ready pose
    #     ur5.moveToP(-0.434, 0.045, 0.15, 0, 0, 0)
    #     print('time used {:.2f}s'.format(time.time() - time_mark))
    #     time_mark = time.time()

    while True: # 7.11s time control
        time_mark = time.time()
        # pick
        ur5.moveToPT(-0.434, -0.18, -0.1 + pick_place_offset, 0, 0, 0, t=1.2)
        # ur5.moveToP(-0.434, -0.18, -0.1, 0, 0, 0)
        ur5.moveToPT(-0.434, -0.18, -0.1, 0, 0, 0, t=0.8, t_wait_reducing=0.3)
        ur5.gripper.closeGripper()
        rospy.sleep(0.5)
        ur5.moveToPT(-0.434, -0.18, -0.1 + pick_place_offset, 0, 0, 0, t=0.8)
        # move
        ur5.moveToPT(-0.434, 0.045, 0.15, 0, 0, 0, t=1.2)
        # place
        ur5.moveToPT(-0.434, 0.275, 0.05, 0, 0, 1.57, t=1, t_wait_reducing=0.6)
        ur5.gripper.openGripper()
        rospy.sleep(0.6)
        # ready pose
        ur5.moveToPT(-0.434, 0.045, 0.15, 0, 0, 0, t=1)
        print('time used {:.2f}s'.format(time.time() - time_mark))
        time_mark = time.time()

    # while True: # 9.93s sudden stops
    #     time_mark = time.time()
    #     # pick
    #     ur5.moveToP(-0.434, -0.18, -0.1 + pick_place_height, 0, 0, 0)
    #     ur5.moveToP(-0.434, -0.18, -0.1, 0, 0, 0)
    #     ur5.gripper.closeGripper()
    #     rospy.sleep(0.5)
    #     ur5.moveToP(-0.434, -0.18, -0.1 + pick_place_height, 0, 0, 0)
    #     # # move
    #     # ur5.moveToP(-0.434, 0.045, 0.15, 0, 0, 0)
    #     # place
    #     ur5.moveToP(-0.434, 0.275, 0.05, 0, 0, 1.57)
    #     ur5.gripper.openGripper()
    #     rospy.sleep(0.5)
    #     # ready pose
    #     ur5.moveToP(-0.434, 0.045, 0.15, 0, 0, 0)
    #     print('time used {:.2f}s'.format(time.time() - time_mark))
    #     time_mark = time.time()

    # while True: #18s
    #     time_mark = time.time()
    #     # pick
    #     ur5.moveToPBlend(-0.434, -0.18, -0.1 + pick_place_offset, 0, 0, 0)
    #     ur5.moveToP(-0.434, -0.18, -0.1, 0, 0, 0)
    #     ur5.gripper.closeGripper()
    #     rospy.sleep(0.5)
    #     ur5.moveToPBlend(-0.434, -0.18, -0.1 + pick_place_offset, 0, 0, 0)
    #     # move
    #     ur5.moveToPBlend(-0.434, 0.045, 0.15, 0, 0, 0)
    #     # place
    #     ur5.moveToP(-0.434, 0.275, 0.05, 0, 0, 1.57)
    #     ur5.gripper.openGripper()
    #     rospy.sleep(0.5)
    #     # ready pose
    #     ur5.moveToP(-0.434, 0.045, 0.15, 0, 0, 0)
    #     print('time used {:.2f}s'.format(time.time() - time_mark))
    #     time_mark = time.time()

    # while True: #18s
    #     time_mark = time.time()
    #     # pick
    #     ur5.moveThruPBlend(-0.434, -0.18, -0.1 + pick_place_offset, 0, 0, 0)
    #     ur5.moveToP(-0.434, -0.18, -0.1, 0, 0, 0)
    #     ur5.gripper.closeGripper()
    #     rospy.sleep(0.5)
    #     ur5.moveThruPBlend(-0.434, -0.18, -0.1 + pick_place_offset, 0, 0, 0)
    #     # move
    #     ur5.moveThruPBlend(-0.434, 0.045, 0.15, 0, 0, 0)
    #     # place
    #     ur5.moveToP(-0.434, 0.275, 0.05, 0, 0, 1.57)
    #     ur5.gripper.openGripper()
    #     rospy.sleep(0.5)
    #     # ready pose
    #     ur5.moveToP(-0.434, 0.045, 0.15, 0, 0, 0)
    #     print('time used {:.2f}s'.format(time.time() - time_mark))
    #     time_mark = time.time()

