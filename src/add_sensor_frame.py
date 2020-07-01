#!/usr/bin/env python
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
    br.sendTransform((0.096, 0.061, 0.046), (0, 0, 0, 1), rospy.Time.now(), "camera_link", "ee_link")
    
    rate.sleep()
