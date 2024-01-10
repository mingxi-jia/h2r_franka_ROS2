import sys
import time
import signal
import rospy
from robotiq_c_model_control.msg import CModel_robot_output as GripperCmd
from robotiq_c_model_control.msg import CModel_robot_input as GripperStat
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


def main():

    rospy.init_node("publish_finger_state")

    data = [0.]

    def update_gripper_stat(msg):
        data[0] = msg.gPO / 255.

    gripper_sub = rospy.Subscriber("/CModelRobotInput", GripperStat, update_gripper_stat)
    joint_state_pub = rospy.Publisher("joint_states", JointState, queue_size=10)
    rate = rospy.Rate(10) # 10hz

    while not rospy.is_shutdown():
        state_msg = JointState()
        state_msg.header = Header()
        state_msg.header.stamp = rospy.Time.now()
        state_msg.name = ["robotiq_85_left_knuckle_joint"]
        state_msg.position = [data[0]]
        state_msg.velocity = []
        state_msg.effort = []
        joint_state_pub.publish(state_msg)
        rate.sleep()

    gripper_sub.unregister()
    joint_state_pub.unregister()
    sys.exit(0)


main()
