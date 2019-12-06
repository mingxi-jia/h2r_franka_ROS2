'''
This module represents a Universal Robots robot arm.

The ordering of the joints is: shoulder pan, shoulder lift, elbow, wrist-1,
wrist-2, wrist-3. This is the physical ordering when starting from the robot's
base and following along the links toward the robot hand. This order might be
different from ordering on ROS topics, e.g., joint_states is alphabetical.
'''

# python
from copy import copy
from time import time
# scipy
import numpy
from numpy.linalg import norm
from numpy import abs, arange, array, asarray, ascontiguousarray, ceil, concatenate, dot, isnan, logical_and, max, pi, sum, where, zeros
# ros
import rospy
import roslib; roslib.load_manifest('ur_driver')
import sensor_msgs.msg as sensor_msgs
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import geometry_msgs.msg as geometry_msgs
# openrave
import openravepy
from trajoptpy import make_kinbodies
# ur_kinematics (IK)
import sys
sys.path.append("/home/ur5/ur5_ws/devel/lib/python2.7/dist-packages/ur_kinematics/")
#import ur_kin_py
from ur_kinematics import ur_kin_py
# self
import point_cloud

class Arm:

  def __init__(self, env, robot, manip, endEffector, viewEffector, isMoving, useURScript):
    '''Initializes arm parameters, joint limits, and controller and robot state topics.'''

    # basic assignments
    self.env = env
    self.robot = robot
    self.manip = manip
    self.endEffector = endEffector
    self.viewEffector = viewEffector
    self.isMoving = isMoving

    # joint values and joint limits
    self.jointValues = [0] * 6
    self.joint_velocities = [0] * 6
    self.joint_names = [''] * 6
    self.joint_limits = robot.GetDOFLimits(manip.GetArmIndices())
    self.joint_limits = array(self.joint_limits)
    self.joint_limits[0,:] = -pi
    self.joint_limits[1,:] = pi
    # self.joint_limits[0][0] = -pi;     self.joint_limits[1][0] = pi
    # self.joint_limits[0][1] = -3*pi/2; self.joint_limits[1][1] = pi/2
    # self.joint_limits[0][2] = -2.8;    self.joint_limits[1][2] = 2.8
    # self.joint_limits[0][3] = -4.15;   self.joint_limits[1][3] = 1.0
    # self.joint_limits[0][4] = -2.18;   self.joint_limits[1][4] = 2.18
    # self.joint_limits[0][5] = -pi;     self.joint_limits[1][5] = pi
    self.robot.SetDOFLimits(self.joint_limits[0], self.joint_limits[1])

    # Ordering of joints.
    self.joint_names_speedj = ['shoulder_pan_joint', 'shoulder_lift_joint', \
       'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    # subscribers
    self.joints_sub = rospy.Subscriber("/joint_states", sensor_msgs.JointState, self.jointsCallback)
    self.joints_sub = rospy.Subscriber("/wrench", geometry_msgs.WrenchStamped, self.forceCallback)
    self.useURScript = useURScript
    if useURScript:
        self.pub = rospy.Publisher('/ur_hardware_interface/script_command', String, queue_size=10)
    else:
        self.pub = rospy.Publisher("/ur_driver/joint_speed", JointTrajectory, queue_size=10)

    # collision checking
    self.handLinkNames = ["gripper_base", "gripper_middle", "finger_0", "finger_1"]
    self.obstacleCloud = None

    # register callback for CTRL+C
    rospy.on_shutdown(self.shutdownCallback)

  def addCloudToEnvironment(self, cloud, cubeSize=0.01):
    '''Add a point cloud to the OpenRAVE environment as a collision obstacle.'''

    # remove existing cloud, if there is one
    self.removeCloudFromEnvironment()

    if cloud.shape[0] == 0:
      print("No points in cloud being added to openrave environment!")
      return

    # convert point cloud to expected format and downsample
    cloud = point_cloud.Voxelize(cubeSize, cloud)

    # add to environment
    self.obstacleCloud = make_kinbodies.create_boxes(self.env, cloud, cubeSize / 2.0)

  def addOldCloudToEnvironment(self):
    '''If a cloud previously added is still in memory, adds it to the environment.'''

    self.env.Add(self.obstacleCloud)

  def calcFK(self, config, endEffectorName=None):
    '''Calculate the Cartesian pose for a given collection of joint positions.'''

    if endEffectorName is None:
      endEffectorName = self.endEffector

    self.robot.SetDOFValues(config, self.manip.GetArmIndices())
    return self.robot.GetLink(endEffectorName).GetTransform()

  def calcIKForPQ(self, pos, quat, opts=openravepy.IkFilterOptions.CheckEnvCollisions):
    '''TODO'''
    targetMat = openravepy.matrixFromPose(numpy.r_[quat, pos])
    solutions = self.manip.FindIKSolutions(targetMat, opts)
    return solutions

  def calcIKForT(self, T, ignoreHandCollisions=False, forceOpenrave=False):
    '''Uses both IKfast and UR IK solvers to find solutions.'''

    if forceOpenrave:
      if ignoreHandCollisions:
        valid = []
        tmpValid = self.manip.FindIKSolutions(T, openravepy.IkFilterOptions.IgnoreEndEffectorCollisions)
        for sol in tmpValid:
          if not self.checkCollisionIgnoreHand(sol):
            valid.append(sol)
      else:
        valid = self.manip.FindIKSolutions(T, openravepy.IkFilterOptions.CheckEnvCollisions)
      print("IK fast found {} solutions.".format(len(valid)))
      return valid

    # Call the IK solver in universal_robot/ur_kinematics
    sols = ur_kin_py.inverse(ascontiguousarray(T), 0.0)

    # Check for joint limits and collisions
    valid = []
    for i, sol in enumerate(sols):

      if isnan(sol[0]): continue

      # Change joint positions such that they are within the limits.
      larger2Pi = logical_and(sol > self.joint_limits[1, :], sol - 2 * pi > self.joint_limits[0, :])
      sol[where(larger2Pi)] -= 2 * pi

      smaller2Pi = logical_and(sol < self.joint_limits[0, :], sol + 2 * pi < self.joint_limits[1, :])
      sol[where(smaller2Pi)] += 2 * pi

      # Check if the config is within the joint limits.
      isInLimits = self.isInJointLimits(sol)

      # Check if the config is in collision.
      if ignoreHandCollisions:
        isInCollision = self.checkCollisionIgnoreHand(sol)
      else:
        isInCollision = self.isInCollision(sol)

      if isInLimits and not isInCollision:
        valid.append(sol)

    # Try IKfast if necessary.
    if len(valid) == 0:
      if ignoreHandCollisions:
        valid = []
        tmpValid = self.manip.FindIKSolutions(T, openravepy.IkFilterOptions.IgnoreEndEffectorCollisions)
        for sol in tmpValid:
          if not self.checkCollisionIgnoreHand(sol):
            valid.append(sol)
      else:
        valid = self.manip.FindIKSolutions(T, openravepy.IkFilterOptions.CheckEnvCollisions)
      print("IK fast found {} solutions.".format(len(valid)))
      if len(valid) > 0: raw_input("IKfast found solutions when ur_kinematics did not.")

    # Return results, if any.
    return valid

  def calcIKWithNoise(self, T, sigma, nAttempts, opts=openravepy.IkFilterOptions.CheckEnvCollisions):
    '''TODO'''

    positionNoise = numpy.random.normal(0, sigma, (nAttempts, 3)).tolist()
    positionNoise[0] = zeros(3)
    for noise in positionNoise:
      NT = copy(T)
      NT[0:3, 3] = NT[0:3, 3] + noise
      configs = self.calcIKForT(NT, opts)
      if len(configs) > 0: break

    return configs

  def checkCollisionIgnoreHand(self, config):
    '''Checks collisions with everything, pretending the hand is missing.'''

    self.robot.SetDOFValues(config, self.manip.GetArmIndices())
    self.env.UpdatePublishedBodies()
    report = openravepy.CollisionReport()
    inCollision = self.env.CheckCollision(self.robot, report)

    if not inCollision: return False

    inCollision = False
    for linkPair in report.vLinkColliding:
      if not (linkPair[0].GetName() in self.handLinkNames or linkPair[1].GetName() in self.handLinkNames):
        inCollision = True
        break

    return inCollision

  def findClosestIK(self, solutions):
    '''Find the joint positions closest to the current joint positions.'''

    return self.findClosestIKToGiven(solutions, self.jointValues)

  def findClosestIKToGiven(self, solutions, jointsStart):
    '''Find the joint positions closest to the given joint positions.'''

    closestSolution = None
    minDist = float('inf')
    for solution in solutions:
      diff = jointsStart - solution
      dist = dot(diff, diff)
      if dist < minDist:
        minDist = dist
        closestSolution = solution
    return closestSolution

  def findClosestIKDistance(self, solutions):
    '''Find the joint positions closest to the current joint positions, and return its distance.'''

    closestSolution = None
    minDist = 1000000
    for solution in solutions:
      dist = norm(self.jointValues - solution)
      if dist < minDist:
        minDist = dist
        closestSolution = solution
    print "Closest IK solution:", closestSolution
    print "Current joint values:", self.jointValues
    return minDist

  def findFarthestIKToGiven(self, solutions, jointsStart):
    '''
    Find the joint positions furthest to the given joint positions.

    @type solutions: list of 1x7 vectors
    @param solutions: the list of joint positions
    '''
    farthestSolution = None
    maxDist = -float('inf')
    for solution in solutions:
      diff = jointsStart - solution
      dist = dot(diff, diff)
      if dist > maxDist:
        maxDist = dist
        farthestSolution = solution
    return farthestSolution

  def publishVelocities(self, velocities_dict):
    if self.useURScript:
        a = 0.1
        t = 0.1
        vels = [velocities_dict['shoulder_pan_joint'],
                velocities_dict['shoulder_lift_joint'],
                velocities_dict['elbow_joint'],
                velocities_dict['wrist_1_joint'],
                velocities_dict['wrist_2_joint'],
                velocities_dict['wrist_3_joint']]
        self.pub.publish('speedj(' + str(vels) + ', ' + str(a) \
                         + ', ' + str(t) + ')')
    else:
      jointTraj = JointTrajectory()
      jointPos = JointTrajectoryPoint(velocities=velocities_dict.values())
      jointTraj.points = [jointPos]
      jointTraj.names = velocities_dict.keys()
      self.pub.publish(jointTraj)

  def followTrajectory(self, traj, gain, gamma, breakGamma=0.75, maxErrorMag=0.80, maxDistToTarget=0.02, isBraking=True):
    '''Simple leaky integrator with error scaling and breaking.'''

    if not self.isMoving:
      print("Skipping followTrajectory since isMoving=False.")
      return

    leakySum = 0.0
    command = zeros(6)

    print("Moving arm...")
    avg_vels = []
    max_vels = []
    for i, target in enumerate(traj):
      error = target - self.jointValues

      while max(abs(error)) > maxDistToTarget:
        t0 = time()

        # scale to maximum error
        errorMag = norm(error)
        if errorMag > maxErrorMag:
          scale = maxErrorMag / errorMag
          error = scale * error

        # integrate error
        leakySum = gamma*leakySum + (1.0-gamma)*error
        command = gain*leakySum

        self.publishVelocities(self.createJointVelocitiesDict(command))

        avg_vels.append(numpy.mean(command))
        max_vels.append(numpy.max(numpy.abs(command)))

        # sleepTime = 0.008 - (time() - t0)
        sleepTime = 0.008
        # print('  sleepTime:', sleepTime)
        rospy.sleep(sleepTime)
        # rospy.sleep(0.008)

        # compute error
        error = target - self.jointValues

      print('[followTrajectory] Reached viapoint #', i, 'with error:', max(abs(error)), abs(error))

    with open('/home/ur5/avg_vels.txt', 'w') as f:
      for v in avg_vels:
        f.write('%.8f\n' % v)

    with open('/home/ur5/max_vels.txt', 'w') as f:
      for v in max_vels:
        f.write('%.8f\n' % v)

    # Reduce velocity to 0.
    if isBraking:
        print("Braking...")
        self.brake(command, breakGamma)

        # set speed to 0.0
        self.publishVelocities(self.createJointVelocitiesDict(zeros(6)))

  def brakeSmooth(self, command, thresh=1e-6):
    diff = numpy.sum(command)

    while max(abs(command)) > thresh:
      t0 = time()
      command = gamma*command
      self.publishVelocities(self.createJointVelocitiesDict(command))
      # sleepTime = 0.008 - (time() - t0)
      sleepTime = 0.008
      rospy.sleep(sleepTime)
      # print('  sleepTime:', sleepTime, 'max(abs(command)):', max(abs(command)))
      # print('  command:', command)

  def brake(self, command, gamma, thresh=1e-6):
    while max(abs(command)) > thresh:
      t0 = time()
      command = gamma*command
      self.publishVelocities(self.createJointVelocitiesDict(command))
      # sleepTime = 0.008 - (time() - t0)
      sleepTime = 0.008
      rospy.sleep(sleepTime)
      # print('  sleepTime:', sleepTime, 'max(abs(command)):', max(abs(command)))
      # print('  command:', command)

  def forceCallback(self, msg):
    '''Callback function for the joint_states ROS topic.'''

    self.force_mag = msg.wrench.force.x ** 2 + msg.wrench.force.y ** 2 + msg.wrench.force.z ** 2

  def isInCollision(self, joint_positions):
    '''Check whether the robot is in collision for the given joint positions.'''

    self.robot.SetDOFValues(joint_positions, self.manip.GetArmIndices())
    self.env.UpdatePublishedBodies()
    return self.env.CheckCollision(self.robot)

  def isInJointLimits(self, joint_positions):
    '''Returns true only if the given arm position is within the joint limits.'''

    return (joint_positions >= self.joint_limits[0, :]).all() and (joint_positions <= self.joint_limits[1, :]).all()

  def jointsCallback(self, msg):
    '''Callback function for the joint_states ROS topic.'''
    positions_dict = jointStateToDict(msg)
    self.joint_positions_dict = positions_dict
    self.jointValues = array([positions_dict[i] for i in self.joint_names_speedj])

  def removeCloudFromEnvironment(self):
    '''Remove any existing point cloud obstacle from the environment.'''

    if self.obstacleCloud != None:
      self.env.Remove(self.obstacleCloud)

  def shutdownCallback(self):
    '''Gradually reduces the joint velocities to zero when the program is trying to shut down.'''

    print("Received shutdown signal ...")

    v = asarray(self.joint_velocities)
    print 'arm velocities:', v

    if norm(v) >= 0.01:
      print("Arm is moving. Slowing down to avoid hard braking.")
      max_iterations = 100
      vel = v
      step = v / max_iterations

      for i in range(0, max_iterations):
        vel -= step
        self.publishVelocities(self.createJointVelocitiesDict(vel))
        # print 'iter:', i, 'velocities:', vel
        rospy.sleep(0.01)

      # Reduce speed to zero.
      self.publishVelocities(self.createJointVelocitiesDict(zeros(6)))
      rospy.sleep(0.01)

    print("Exiting ...")

  def update(self):
    '''Set the OpenRave robot to the current joint values and update the viewer.'''

    self.robot.SetDOFValues(self.jointValues, self.manip.GetArmIndices())
    self.env.UpdatePublishedBodies()
    rospy.sleep(0.1)

  def createJointVelocitiesDict(self, velocities):
      dict_out = {}
      for i in range(len(velocities)):
        dict_out[self.joint_names_speedj[i]] = velocities[i]
      return dict_out

# =============================================================================
# Helper functions

def jointStateToDict(msg):
  '''Convert JointState message to dictionary.'''
  dict_out = {}
  for i in range(len(msg.position)):
      dict_out[msg.name[i]] = msg.position[i]

  return dict_out
