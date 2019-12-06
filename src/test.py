import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

joint_names_speedj = ['shoulder_pan_joint', 'shoulder_lift_joint', \
                       'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


# class test:
#     def __init__(self):
#         self.jointValues = []
#
#     def publishVelocities(self, velocities_dict):
#         a = 0.1
#         t = 0.1
#         vels = [velocities_dict['shoulder_pan_joint'],
#                 velocities_dict['shoulder_lift_joint'],
#                 velocities_dict['elbow_joint'],
#                 velocities_dict['wrist_1_joint'],
#                 velocities_dict['wrist_2_joint'],
#                 velocities_dict['wrist_3_joint']]
#         self.pub.publish('speedj(' + str(vels) + ', ' + str(a) \
#                          + ', ' + str(t) + ')')
#
#     def createJointVelocitiesDict(self, velocities):
#         dict_out = {}
#         for i in range(len(velocities)):
#             dict_out[self.joint_names_speedj[i]] = velocities[i]
#         return dict_out
#
#     def jointStateToDict(self, msg):
#         '''Convert JointState message to dictionary.'''
#         dict_out = {}
#         for i in range(len(msg.position)):
#             dict_out[msg.name[i]] = msg.position[i]
#
#         return dict_out
#
#     def jointsCallback(self, msg):
#         global joint_values
#         '''Callback function for the joint_states ROS topic.'''
#         positions_dict = self.jointStateToDict(msg)
#         joint_positions_dict = positions_dict
#         joint_values = [positions_dict[i] for i in joint_names_speedj]
#
#     def followTrajectory(self, traj, gain, gamma, breakGamma=0.75, maxErrorMag=0.80, maxDistToTarget=0.02,
#                          isBraking=True):
#         '''Simple leaky integrator with error scaling and breaking.'''
#
#         leakySum = 0.0
#         command = zeros(6)
#
#         print("Moving arm...")
#         avg_vels = []
#         max_vels = []
#         for i, target in enumerate(traj):
#             error = target - self.jointValues
#
#             while max(abs(error)) > maxDistToTarget:
#                 t0 = time()
#
#                 # scale to maximum error
#                 errorMag = norm(error)
#                 if errorMag > maxErrorMag:
#                     scale = maxErrorMag / errorMag
#                     error = scale * error
#
#                 # integrate error
#                 leakySum = gamma * leakySum + (1.0 - gamma) * error
#                 command = gain * leakySum
#
#                 self.publishVelocities(self.createJointVelocitiesDict(command))
#
#                 avg_vels.append(numpy.mean(command))
#                 max_vels.append(numpy.max(numpy.abs(command)))
#
#                 # sleepTime = 0.008 - (time() - t0)
#                 sleepTime = 0.008
#                 # print('  sleepTime:', sleepTime)
#                 rospy.sleep(sleepTime)
#                 # rospy.sleep(0.008)
#
#                 # compute error
#                 error = target - self.jointValues
#
#             print('[followTrajectory] Reached viapoint #', i, 'with error:', max(abs(error)), abs(error))
#
#         with open('/home/ur5/avg_vels.txt', 'w') as f:
#             for v in avg_vels:
#                 f.write('%.8f\n' % v)
#
#         with open('/home/ur5/max_vels.txt', 'w') as f:
#             for v in max_vels:
#                 f.write('%.8f\n' % v)
#
#         # Reduce velocity to 0.
#         if isBraking:
#             print("Braking...")
#             self.brake(command, breakGamma)
#
#             # set speed to 0.0
#             self.publishVelocities(self.createJointVelocitiesDict(zeros(6)))



rospy.init_node('test_ur_script')
# joints_sub = rospy.Subscriber("/joint_states", JointState, jointsCallback)
pub = rospy.Publisher('/ur_hardware_interface/script_command', String, queue_size=10)

pose = [-0.6, -0.1, 0.3, 0, 3.14, 0]
joint_pose_home = [1.28, -1.25, 1.13, -1.47, -1.5, -1.3]
# s = 'movel(p{},v=0.01)'.format(pose)
s = 'movej({}, v=0.01)'.format(joint_pose_home)
pub.publish(s)
rospy.sleep(0.008)
print(1)