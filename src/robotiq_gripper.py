'''
This module represents a robot gripper. It uses the gripper driver from
https://github.com/UTNuclearRoboticsPublic/robotiq/tree/kinetic-devel.
'''

import rospy
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output as GripperCmd
from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_input as GripperStat

class Gripper:

  def __init__(self, isMoving):
    '''Constructor.'''

    self.gripperSub = rospy.Subscriber('/Robotiq2FGripperRobotInput', GripperStat, self.updateGripperStat)
    self.gripperPub = rospy.Publisher('/Robotiq2FGripperRobotOutput', GripperCmd, queue_size=1)

    self.status = None
    self.isMoving = isMoving

    print('Waiting for gripper driver to connect ...')
    while self.gripperPub.get_num_connections() == 0 or self.status is None:
      rospy.sleep(0.01)

    if self.status.gACT == 0:
      self.reset()
      self.activate()

  def updateGripperStat(self, msg):
    '''Obtain the status of the gripper.'''

    self.status = msg

  def reset(self):
    '''Reset the gripper.'''

    print('Resetting gripper ...')

    cmd = GripperCmd()
    cmd.rACT = 0
    self.gripperPub.publish(cmd)
    rospy.sleep(0.5)

  def activate(self):
    '''Activate the gripper.'''

    print('Activating gripper ...')

    cmd = GripperCmd()
    cmd.rACT = 1
    cmd.rGTO = 1
    cmd.rSP  = 255
    cmd.rFR  = 150
    self.gripperPub.publish(cmd)
    rospy.sleep(0.5)

  def closeGripper(self, speed=255, force=255):
    '''Close the gripper. Default values for optional arguments are set to their max.'''

    if not self.isMoving: return

    # print('Closing gripper ...')
    cmd = GripperCmd()
    cmd.rACT = 1
    cmd.rGTO = 1
    cmd.rPR = 255 # position
    cmd.rFR = force
    cmd.rSP = speed
    self.gripperPub.publish(cmd)
    # rospy.sleep(0.5)

  def openGripper(self, speed=255, force=255, position=0):
    '''Open the gripper. Default values for optional arguments are set to their max.'''

    if not self.isMoving: return

    # print('Opening gripper ...')
    cmd = GripperCmd()
    cmd.rACT = 1
    cmd.rGTO = 1
    cmd.rPR = position # position
    cmd.rFR = force
    cmd.rSP = speed
    self.gripperPub.publish(cmd)
    # rospy.sleep(0.5)

  def isClosed(self):
    # return self.status.gCU < 10
    print('checking gripper')
    return self.status.gPO > 204
