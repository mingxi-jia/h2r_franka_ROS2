#!/usr/bin/env python3
'''Broadcasts the transform for the sensor.

Credits:
- http://wiki.ros.org/tf/Tutorials/Adding%20a%20frame%20%28Python%29

Assembled: Northeastern University, 2015
'''

import rospy
import tf

if __name__ == '__main__':
  
  rospy.init_node('add_sensor_frame')
  br = tf.TransformBroadcaster()
  rate = rospy.Rate(100.0)
  
  while not rospy.is_shutdown():
    # measured
    # br.sendTransform((0.092, 0.062, 0.044), (0, 0, 0, 1), rospy.Time.now(), "camera_link", "ee_link")
    # block calibrated
    # br.sendTransform((0.094, 0.061, 0.050), (0, 0, 0, 1), rospy.Time.now(), "camera_link", "ee_link")
    # br.sendTransform((0.096, 0.061, 0.046), (0, 0, 0, 1), rospy.Time.now(), "camera_link", "ee_link")
    # br.sendTransform((0.08, 0.061, 0.046), (0, 0, 0, 1), rospy.Time.now(), "camera_link", "ee_link")
    
    # br.sendTransform((0.533, 0.04, 0.8), (0, 0.7068252, 0, 0.7073883), rospy.Time.now(), "camera_link", "base_link")
    br.sendTransform((0,0,0), (0,0,0, 1), rospy.Time.now(), "base_link", "virtual_base_link")
    br.sendTransform((0.290, -0.402, 0.280), (-0.232, 0.236, 0.672, 0.662), rospy.Time.now(), "cam_3_link", "virtual_base_link")
    # br.sendTransform((0.412, 0.028, 0.962), (-0.504, 0.504, 0.484, 0.508), rospy.Time.now(), "cam_2_link", "virtual_base_link")
    # br.sendTransform((0.425, 0.066, 0.889), (-0.514, 0.514, 0.477, 0.494), rospy.Time.now(), "camera_base", "virtual_base_link")
    br.sendTransform((0.425, 0.058, 0.91), (-0.506, 0.507, 0.485, 0.502), rospy.Time.now(), "camera_base", "virtual_base_link")
    br.sendTransform((0.492, 0.652, 0.320), (0.269, 0.265, -0.674, 0.635), rospy.Time.now(), "cam_1_link", "virtual_base_link")
    # br.sendTransform((0.4334319986892161, 0.655284778074079, 0.3213047165316557), (-0.02146651730867777, -0.9212888143273752, 0.38786679283955533, 0.01804051668474774), rospy.Time.now(), "cam_1_color_optical_frame", "base_link")
    # br.sendTransform((0.4569025746249762, 0.06498678898163027, 0.8937676989977422), (0.9994043841503968, -0.01021054957586394, 0.008269457088688694, 0.031909837006529954), rospy.Time.now(), "rgb_camera_link", "base_link")
    # br.sendTransform((0.34921490169263303, -0.4024229613310762, 0.279550692469885), (-0.9007312427647147, -0.00417783214204831, 0.006725739403190029, 0.4343046608725331), rospy.Time.now(), "cam_3_color_optical_frame", "base_link")
    # br.sendTransform((-0.416, 0.019, 0.954), (1.000, -0.010, 0.010, -0.009), rospy.Time.now(), "base_link", "cam_2_color_optical_frame")
    # br.sendTransform((0.392, -0.239, 0.714), (0.021, 0.921, -0.388, 0.018), rospy.Time.now(), "base_link", "cam_1_color_optical_frame")
    # br.sendTransform((-0.471, 0.018, 0.889), (0.999, -0.010, 0.008, -0.032), rospy.Time.now(), "base_link", "rgb_camera_link")

    rate.sleep()
