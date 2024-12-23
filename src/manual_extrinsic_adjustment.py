#!/usr/bin/env python

# sudo pip install keyboard
from math import radians
import pynput.keyboard as keyboard
import rclpy
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, TransformStamped, Vector3
import argparse

"""
Clone from Dian Wang's helping_hands_ur5 env at Northeastern University
A single file Python-only replacement for tf_keyboard_cal,
does NOT need to have the terminal on focus.

It needs to be run as root as explained in the keyboard library https://pypi.org/project/keyboard/

'To avoid depending on X, the Linux parts reads raw device files (/dev/input/input*) but this requries root.'

It's ugly, but it works. Remember to source your ROS installation
and maybe point your ROS_MASTER_URI correctly.

Author: Sammy Pfeiffer <Sammy.Pfeiffer at student.uts.edu.au>
License: BSD

usage: keyboard_py.py [-h]
                      from_frame child_frame_id x y z rpy r p y --unit [deg, rad] rate
                      from_frame child_frame_id x y z quaternion x y z w rate

1.27683 -0.0480804 0.583822 -0.287011 -0.00599455 0.957905 -0.00244425 

Set up a TF transformation with your keyboard.

positional arguments:
  from_frame        from which frame to be a child of
  child_frame_id    to which frame, child_frame_id
  x                 initial x coordinate
  y                 initial y coordinate
  z                 initial z coordinate
  {rpy,quaternion}
  rate              rate to publish the transform

rpy positional arguments:
  r            initial roll rotation
  p            initial pitch rotation
  yaw          initial yaw rotation

quaternion positional arguments:
  qx          initial x quaternion unit
  qy          initial y quaternion unit
  qz          initial z quaternion unit
  qw          initial w quaternion unit


optional arguments:
  --unit UNIT  specifying rotation of rpy in degrees 'deg' or radians 'rad'

optional arguments:
  -h, --help        show this help message and exit

"""


class KeyboardTFCalibration(Node):
    FINE = 0.001
    COARSE = 0.01
    VERY_COARSE = 0.1

    def __init__(self, from_frame, child_frame_id,
                 pose, rate):
        super().__init__("KeyboardTFCalibration")
        self.from_frame = from_frame
        self.child_frame_id = child_frame_id
        self.last_pose = PoseStamped()
        self.last_pose.header.frame_id = child_frame_id
        self.last_pose.pose = pose
        self.delta = self.COARSE
        self.rate = rate

        self.timer = self.create_timer(1.0/self.rate, self.timer_callback)

        print("Starting transform:")
        print("From frame: " + str(from_frame))
        print("To child_frame_id: " + str(child_frame_id))
        print("With pose: " + str(pose))
        print("Publishing at rate: " + str(self.rate))

        self.tf_br = TransformBroadcaster(self)
        self.pub = self.create_publisher(PoseStamped, 'keyboardtfcal_ps', 1)

        # Set hooks for all keys

        key_listeners = {}
        key_listeners['q'] = keyboard.Listener(self.keys_cb, None)
        # key_listeners['a'] = keyboard.Listener(self.keys_cb, None)
        # key_listeners['w'] = keyboard.Listener(self.keys_cb, None)
        # key_listeners['s'] = keyboard.Listener(self.keys_cb, None)
        # key_listeners['e'] = keyboard.Listener(self.keys_cb, None)
        # key_listeners['d'] = keyboard.Listener(self.keys_cb, None)
        # key_listeners['r'] = keyboard.Listener(self.keys_cb, None)
        # key_listeners['f'] = keyboard.Listener(self.keys_cb, None)
        # key_listeners['t'] = keyboard.Listener(self.keys_cb, None)
        # key_listeners['g'] = keyboard.Listener(self.keys_cb, None)
        # key_listeners['y'] = keyboard.Listener(self.keys_cb, None)
        # key_listeners['h'] = keyboard.Listener('h', self.keys_cb, None)
        # key_listeners['u'] = keyboard.Listener('u', self.keys_cb, None)
        # key_listeners['i'] = keyboard.Listener('i', self.keys_cb, None)
        # key_listeners['o'] = keyboard.Listener('o', self.keys_cb, None)
        # key_listeners['p']= keyboard.Listener('p', self.keys_cb, None)

        for val in key_listeners.values():
            val.start()

        self.usage()

# def print_(arg1, arg2):
#     print(arg1, arg2)

    def usage(self):
        usage_str = """Manual alignment of {} to {}:
=======================================
MOVE:  X  Y  Z  R  P  YAW
-------------------------
up     q  w  e  r  t  y
down   a  s  d  f  g  h
Fast:  u
Med:   i
Slow:  o
Print: p""".format(self.from_frame, self.child_frame_id)

        print("\n" + usage_str)

    def keys_cb(self, key):
        trigger = key.char
        print(("trigger: " + str(trigger)))
        o = self.last_pose.pose.orientation
        r, p, y = euler_from_quaternion([o.x, o.y, o.z, o.w])
        if trigger == 'q':
            self.last_pose.pose.position.x += self.delta
        if trigger == 'a':
            self.last_pose.pose.position.x -= self.delta
        if trigger == 'w':
            self.last_pose.pose.position.y += self.delta
        if trigger == 's':
            self.last_pose.pose.position.y -= self.delta
        if trigger == 'e':
            self.last_pose.pose.position.z += self.delta
        if trigger == 'd':
            self.last_pose.pose.position.z -= self.delta
        if trigger == 'r':
            r += self.delta
        if trigger == 'f':
            r -= self.delta
        if trigger == 't':
            p += self.delta
        if trigger == 'g':
            p -= self.delta
        if trigger == 'y':
            y += self.delta
        if trigger == 'h':
            y -= self.delta
        if trigger in 'rftgyh':
            q = quaternion_from_euler(r, p, y)
            self.last_pose.pose.orientation = Quaternion()
            self.last_pose.pose.orientation.x = q[0]
            self.last_pose.pose.orientation.y = q[1]
            self.last_pose.pose.orientation.z = q[2]
            self.last_pose.pose.orientation.w = q[3]
        if trigger == 'u':
            self.delta = self.VERY_COARSE
        if trigger == 'i':
            self.delta = self.COARSE
        if trigger == 'o':
            self.delta = self.FINE
        if trigger == 'p':
            self.print_commands(self.last_pose)
        self.usage()
        print("")
        print(("Delta: " + str(self.delta)))
        self.print_pose(self.last_pose)

    def print_pose(self, pose, decimals=4):
        p = pose.pose.position
        o = pose.pose.orientation
        pos_part = str(round(p.x, decimals)) + " " + \
            str(round(p.y, decimals)) + " " + str(round(p.z, decimals))
        r, p, y = euler_from_quaternion([o.x, o.y, o.z, o.w])
        ori_part = str(round(r, decimals)) + " " + \
            str(round(p, decimals)) + " " + str(round(y, decimals))
        print((pos_part + " " + ori_part))

    def print_commands(self, pose, decimals=4):
        p = pose.pose.position
        o = pose.pose.orientation

        print("\n---------------------------")
        print("Static transform publisher command (with roll pitch yaw):")
        common_part = "rosrun tf static_transform_publisher "
        pos_part = str(round(p.x, decimals)) + " " + \
            str(round(p.y, decimals)) + " " + str(round(p.z, decimals))
        r, p, y = euler_from_quaternion([o.w, o.x, o.y, o.z])
        ori_part = str(round(r, decimals)) + " " + \
            str(round(p, decimals)) + " " + str(round(y, decimals))
        static_tf_cmd = common_part + pos_part + " " + ori_part + \
            " " + self.from_frame + " " + self.child_frame_id + " 50"
        print("  " + static_tf_cmd)
        print()
        print("Static transform publisher command (with quaternion):")
        ori_q = str(round(o.x, decimals)) + " " + str(round(o.y, decimals)) + \
            " " + str(round(o.z, decimals)) + " " + str(round(o.w, decimals))
        static_tf_cmd = common_part + pos_part + " " + ori_q + \
            " " + self.from_frame + " " + self.child_frame_id + " 50"
        print("  " + static_tf_cmd)
        print()

        print("Roslaunch line of static transform publisher (rpy):")
        node_name = "from_" + self.from_frame + \
            "_to_" + self.child_frame_id + "_static_tf"
        roslaunch_part = '  <node name="' + node_name + '" pkg="tf" type="static_transform_publisher" args="' +\
                         pos_part + " " + ori_part + " " + self.from_frame + \
            " " + self.child_frame_id + " 50" + '" />'
        print(roslaunch_part)
        print()

        print("URDF format:")  # <origin xyz="0.04149 -0.01221 0.001" rpy="0 0 0" />
        print('  <origin xyz="' + pos_part + '" rpy="' + ori_part + '" />')
        print("\n---------------------------")

        print("Run again this program:")
        print("keyboard_py.py " + self.from_frame + " " + self.child_frame_id + " " + pos_part + " rpy " + ori_part + " --unit rad " + str(self.rate))
        print()

    def timer_callback(self):
        if self.tf_br is not None:
            pos = self.last_pose.pose.position
            ori = self.last_pose.pose.orientation
            ts = TransformStamped()
            ts.header.stamp =  self.get_clock().now().to_msg()
            ts.header.frame_id = self.from_frame
            ts.child_frame_id = self.child_frame_id
            trans = Vector3()
            trans.x = pos.x
            trans.y = pos.y
            trans.z = pos.z
            ts.transform.translation = trans 
            ts.transform.rotation = ori
            self.tf_br.sendTransform(ts)
        ps = PoseStamped()
        ps.pose = self.last_pose.pose
        ps.header.frame_id = self.from_frame
        ps.header.stamp =  self.get_clock().now().to_msg()
        self.pub.publish(ps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Set up a TF transformation with your keyboard.')
    parser.add_argument('from_frame', type=str,
                        help='from which frame to be a child of')
    parser.add_argument('child_frame_id', type=str,
                        help='to which frame, child_frame_id')
    parser.add_argument('x', type=float,
                        help='initial x coordinate')
    parser.add_argument('y', type=float,
                        help='initial y coordinate')
    parser.add_argument('z', type=float,
                        help='initial z coordinate')

    subparsers = parser.add_subparsers(dest='subcommand')
    parser_rpy = subparsers.add_parser('rpy')

    parser_rpy.add_argument('r', type=float,
                            help='initial roll rotation')
    parser_rpy.add_argument('p', type=float,
                            help='initial pitch rotation')
    parser_rpy.add_argument('yaw', type=float,
                            help='initial yaw rotation')
    parser_rpy.add_argument('--unit', type=str,
                            default='deg',
                            required=False,
                            help="specifying rotation of rpy in degrees 'deg' or radians 'rad'")

    parser_quaternion = subparsers.add_parser('quaternion')
    parser_quaternion.add_argument('qx', type=float,
                                   help='initial x quaternion unit')
    parser_quaternion.add_argument('qy', type=float,
                                   help='initial y quaternion unit')
    parser_quaternion.add_argument('qz', type=float,
                                   help='initial z quaternion unit')
    parser_quaternion.add_argument('qw', type=float,
                                   help='initial w quaternion unit')

    parser.add_argument('rate', type=float,
                        default=20,
                        help='rate to publish the transform')

    a = parser.parse_args()

    ps = Pose()
    ps.position.x = a.x
    ps.position.y = a.y
    ps.position.z = a.z
    if hasattr(a, 'qw'):
        ps.orientation.x = a.qx
        ps.orientation.y = a.qy
        ps.orientation.z = a.qz
        ps.orientation.w = a.qw
    else:
        if a.unit == 'deg':
            q = quaternion_from_euler(radians(a.r),
                                      radians(a.p),
                                      radians(a.yaw))
        else:
            q = quaternion_from_euler(a.r, a.p, a.yaw)

        ps.orientation = Quaternion()
        ps.orientation.x = q[0]
        ps.orientation.y = q[1]
        ps.orientation.z = q[2]
        ps.orientation.w = q[3]

    rclpy.init()
    try:
        ktfc = KeyboardTFCalibration(a.from_frame,
                                    a.child_frame_id,
                                    ps,
                                    a.rate)
        rclpy.spin(ktfc)

    except KeyboardInterrupt:
        pass
    finally:
        # this is needed because of ROS2 mechanism.
        # without destroy_node(), it somehow wont work if you restart the program
        ktfc.destroy_node()
        rclpy.shutdown()
