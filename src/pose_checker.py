from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
import rospy
import numpy as np

class PoseChecker:
    def __init__(self, max_z=20, stop_acc=20, n_readings=1):
        self.tf = []
        self.command_pub = rospy.Publisher('/ur_hardware_interface/script_command', String, queue_size=10)
        self.wrench_sub = rospy.Subscriber("/tf", TFMessage, self.tfCallback)

    def tfCallback(self, msg):
        '''Callback function for the wrench ROS topic.'''
        for i in range(len(msg.transforms)):
            if msg.transforms[i].child_frame_id == 'tool0_controller' and msg.transforms[0].header.frame_id == 'base':
                print(msg.transforms[i].transform)

if __name__ == '__main__':
    rospy.init_node('ur5')
    a = PoseChecker()
    while True:
        rospy.sleep(0.5)
