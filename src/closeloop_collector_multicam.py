import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.task import Future

from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_srvs.srv import Trigger  # Import for pause service
from sensor_msgs.msg import Image, JointState, Joy
from geometry_msgs.msg import Pose, Twist, PoseStamped, TwistStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume, RobotState, JointConstraint

import os
import git
import time
import yaml
import threading
import numpy as np
from pynput import keyboard  # Import pynput.keyboard
from datetime import datetime

from panda_utils.configs import JOINT_HOME


def save_dict_to_yaml(data, filename):
    with open(filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

class SynchronizedDatasetCollector(Node):
    def __init__(self):
        super().__init__('synchronized_dataset_collector')

        self.collect_freq = 10
        self.time_diff = 1. / self.collect_freq

        # Subscribers
        camera_name = 'dave'
        self.rgb_sub1 = Subscriber(self, Image, f'/{camera_name}/color/image_raw')
        self.depth_sub1 = Subscriber(self, Image, f'/{camera_name}/aligned_depth_to_color/image_raw')
        camera_name = 'stuart'
        self.rgb_sub2 = Subscriber(self, Image, f'/{camera_name}/color/image_raw')
        self.depth_sub2 = Subscriber(self, Image, f'/{camera_name}/aligned_depth_to_color/image_raw')
        camera_name = 'mel'
        self.rgb_sub3 = Subscriber(self, Image, f'/{camera_name}/color/image_raw')
        self.depth_sub3 = Subscriber(self, Image, f'/{camera_name}/aligned_depth_to_color/image_raw')

        self.action_sub = Subscriber(self, TwistStamped, '/servo_node/delta_twist_cmds')
        self.pose_sub = Subscriber(self, PoseStamped, '/franka_robot_state_broadcaster/current_pose')
        self.franka_joint_sub = Subscriber(self, JointState, '/joint_states')
        self.gripper_action_sub = Subscriber(self, Joy, '/spacemouse/grasp_signals')
        self.get_logger().info("All data subscriber ready.")

        # Synchronizer
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub1, self.rgb_sub2, self.rgb_sub3, self.depth_sub1, self.depth_sub2, self.depth_sub3, self.action_sub, self.pose_sub, self.franka_joint_sub, self.gripper_action_sub],
            queue_size=20,          
            slop=self.time_diff
        )
        self.ts.registerCallback(self.sync_callback)
        self.get_logger().info("Data Synchronizer ready.")

        self.base_link = "fr3_link0"
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.move_group_client = ActionClient(self, MoveGroup, 'move_action')
        self.move_group_client.wait_for_server()
        self.get_logger().info("move_group_client ready.")
        
        self.current_joint_state: JointState = None
        while self.current_joint_state is None:
            self.get_logger().warn("Waiting for current_joint_state...")
            time.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=1.0)
        # initialize gripper state because the init open width is always different, causing problems for motion planning
        gripper_init_state = self.current_joint_state.position[-1]
        JOINT_HOME['fr3_finger_joint1'] = JOINT_HOME['fr3_finger_joint2'] = gripper_init_state

        # Service client for pausing servo node
        self.pause_servo_client = self.create_client(Trigger, '/servo_node/pause_servo')
        self.start_servo_client = self.create_client(Trigger, '/servo_node/start_servo')

        # prepare dataset meta
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        ymdhm = datetime.now().strftime("%Y-%m-%d-%H-%M")

        task_name = input('what is the task name for this collection: ')
        self.dataset_path = f"./raw_datasets/{task_name}_{ymdhm}"
        os.makedirs(self.dataset_path)

        language_instruction = input('what is the language instruction for this task: ')
        comments = input('leave some meta comments for this dataset: ')
        self.meta_data = {'collection_time': ymdhm,
                     'Robot': 'fr3',
                     'task_name': task_name,
                     'git': sha,
                     'camera_intrinsics': None, 
                     'camera_extrinsics': None,
                     'number_of_episodes': 0, 
                     'language_instruction': language_instruction,
                     'meta_comments': comments,
                     'collect_freq': self.collect_freq
                     }
        # RLDS-style Dataset
        self.episode = []
        self.episode_idx = 0
        self.saving_flag = False

        # Thread for keyboard input
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()

        self.get_logger().warn("Waiting 3 seconds for initialization...")
        time.sleep(3.0)

        self.get_logger().info("New episode started.")
        self.start_new_episode(proceed=False)

    def keyboard_listener(self):
        def on_press(key):
            try:
                if key.char == 'a': # next episode
                    self.saving_flag = False
                    self.get_logger().info("a key pressed. abandon and continue new episode...")
                    self.start_new_episode(proceed=False)
                elif key.char == 's': # save and continue
                    self.saving_flag = False
                    self.get_logger().info("s key pressed. saved and continue new episode...")
                    self.start_new_episode(proceed=True)
            except AttributeError:
                # Handle special keys (e.g., shift, ctrl)
                pass

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    def sync_callback(self, rgb1: Image, rgb2: Image, rgb3: Image, depth1: Image, depth2: Image, depth3: Image, action_stamped:TwistStamped, pose_stamped:PoseStamped, franka_joint_states:JointState, gripper_action: Joy):
        # self.get_logger().info(f"RGB timestamp: {rgb.header.stamp.sec}.{rgb.header.stamp.nanosec}")
        # self.get_logger().info(f"Action timestamp: {action_stamped.header.stamp.sec}.{action_stamped.header.stamp.nanosec}")
        # self.get_logger().info(f"pose_stamped timestamp: {pose_stamped.header.stamp.sec}.{pose_stamped.header.stamp.nanosec}")
        # self.get_logger().info(f"franka_joint_states timestamp: {franka_joint_states.header.stamp.sec}.{franka_joint_states.header.stamp.nanosec}")
        # self.get_logger().info(f"gripper_action timestamp: {gripper_action.header.stamp.sec}.{gripper_action.header.stamp.nanosec}")
        # self.get_logger().info("sync_callback triggered.")
        # Collect synchronized data
        rgb_data_dave = np.array(rgb1.data).reshape((rgb1.height, rgb1.width, -1))
        depth_data_dave = np.array(depth1.data).reshape((depth1.height, depth1.width, -1))
        rgb_data_stuart = np.array(rgb2.data).reshape((rgb2.height, rgb2.width, -1))
        depth_data_stuart = np.array(depth2.data).reshape((depth2.height, depth2.width, -1))
        rgb_data_mel = np.array(rgb3.data).reshape((rgb3.height, rgb3.width, -1))
        depth_data_mel = np.array(depth3.data).reshape((depth3.height, depth3.width, -1))
        twist_action = action_stamped.twist
        twist_action_data = np.array([twist_action.linear.x, twist_action.linear.y, twist_action.linear.z,
                                twist_action.angular.x, twist_action.angular.y, twist_action.angular.z])
        pose = pose_stamped.pose
        pose_data = np.array([pose.position.x, pose.position.y, pose.position.z,
                              pose.orientation.x, pose.orientation.y,
                              pose.orientation.z, pose.orientation.w])
        franka_joint_data = np.array(franka_joint_states.position)
        gripper_joint_data = np.array(franka_joint_states.position[-2:])
        gripper_action_data = np.array(gripper_action.buttons[0])

        # start saving only when there are 
        if (twist_action_data[:2] > 0.).any():
            # only check x and y because (if four dof) other twist command have auto compensation
            self.saving_flag = True
            # self.get_logger().info("saving_flag triggered by valid action inputs.")

        if self.saving_flag:
            # Add synchronized step to RLDS
            step = dict()
            step["seconds"] = rgb1.header.stamp.sec
            step["nanoseconds"] = rgb1.header.stamp.nanosec
            step["rgb_dave"] = rgb_data_dave
            step["depth_dave"] = depth_data_dave
            step["rgb_stuart"] = rgb_data_stuart
            step["depth_stuart"] = depth_data_stuart
            step["rgb_mel"] = rgb_data_mel
            step["depth_mel"] = depth_data_mel
            step["action_rel"] = twist_action_data
            step["franka_ee_pose_states"] = pose_data
            step["franka_joint_states"] = franka_joint_data
            step['gripper_joint_states'] = gripper_joint_data
            step['gripper_joint_states'] = gripper_joint_data
            step['grasp_action'] = gripper_action_data
            self.episode.append(step)
            self.get_logger().info("Synchronized data step added.")
    
    def joint_state_callback(self, msg: JointState):
        """Callback to update current joint states."""
        self.current_joint_state = msg
        # self.get_logger().info(f'Updated current joint state: {msg.name}')

    def start_new_episode(self, proceed=True):
        """Starts a new episode after confirming the robot reset."""
        if proceed:
            self.get_logger().info(f"Saved a new episode with {len(self.episode)}.")
            idx = str(self.episode_idx).zfill(4)
            self.save_episode(f'episode_{idx}', self.episode)
            time.sleep(0.2)
            # self.dataset[f'episode_{idx}'] = self.episode
            self.episode_idx += 1

        # Start a new episode
        self.episode = []
        # Wait for robot reset and user confirmation
        self.pause_servo()
        if self.reset_robot():
            self.start_servo()
        else:
            self.get_logger().error("Failed to reset the robot. Exiting initialization.")
        self.get_logger().info("Press Enter to start a new episode.")

        self.get_logger().info("New episode started. Move to start. Press S to save or A to abandon. Ctrl-C to finish.")

    def pause_servo(self):
        """Pauses the servo node by calling the pause service."""
        self.get_logger().info("pause servo")
        if not self.pause_servo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Pause servo service not available.")
            return

        request = Trigger.Request()
        future = self.pause_servo_client.call_async(request)
        future.add_done_callback(self.pause_servo_response)
        time.sleep(1)

    def pause_servo_response(self, future):
        """Callback for handling the pause service response."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Servo node successfully paused.")
            else:
                self.get_logger().error(f"Failed to pause servo node: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Exception while pausing servo: {e}")

    def start_servo(self):
        """Pauses the servo node by calling the pause service."""
        if not self.start_servo_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Pause servo service not available.")
            return

        request = Trigger.Request()
        future = self.start_servo_client.call_async(request)
        future.add_done_callback(self.start_servo_response)
        time.sleep(1)

    def start_servo_response(self, future):
        """Callback for handling the pause service response."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Servo node successfully started.")
            else:
                self.get_logger().error(f"Failed to start servo node: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Exception while starting servo: {e}")

    def reset_robot(self):
        
        while self.current_joint_state is None:
            self.get_logger().warn("Waiting for current_joint_state...")
            time.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=1.0)

        goal_msg = MoveGroup.Goal()
        # Fill goal_msg with relevant goal constraints from PoseStamped
        goal_msg.request.workspace_parameters.header.frame_id = self.base_link
        goal_msg.request.workspace_parameters.min_corner.x = -1.0
        goal_msg.request.workspace_parameters.min_corner.y = -1.0
        goal_msg.request.workspace_parameters.min_corner.z = -1.0
        goal_msg.request.workspace_parameters.max_corner.x = 1.0
        goal_msg.request.workspace_parameters.max_corner.y = 1.0
        goal_msg.request.workspace_parameters.max_corner.z = 1.0

        start_state = RobotState()
        start_state.joint_state = self.current_joint_state
        goal_msg.request.start_state.is_diff = False
        goal_msg.request.start_state = start_state

        # Set goal constraints using the IK solution
        constraints = Constraints()
        for joint, position in JOINT_HOME.items():
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = joint
            joint_constraint.position = position
            joint_constraint.tolerance_above = 0.0001
            joint_constraint.tolerance_below = 0.0001
            joint_constraint.weight = 1.0
            constraints.joint_constraints.append(joint_constraint)

        # print(constraints)
        goal_msg.request.goal_constraints.append(constraints)
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.group_name = 'fr3_arm'
        goal_msg.request.pipeline_id = 'move_group'
        goal_msg.request.max_velocity_scaling_factor = 0.3
        goal_msg.request.max_acceleration_scaling_factor = 0.3
        # goal_msg.request.planner_id = 'RRTConnectkConfigDefault'

        # print(goal_msg)

        self.get_logger().info("Sending reset goal... Approximately 5s.")
        future = self.move_group_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)
        time.sleep(5.0)
        return True  # Return early to avoid blocking

    def goal_response_callback(self, future):
        """Callback for handling goal response."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Reset goal was rejected.")
            return

        self.get_logger().info("Reset goal accepted, waiting 5 seconds for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        """Callback for handling motion result."""
        result = future.result()
        if result.status == 4:  # Assuming 1 indicates success
            self.get_logger().info("Robot successfully reset.")
        else:
            self.get_logger().error(f"Failed to reset robot. Error code: {result.status} Please manually open the gripper and try again.")

    def save_episode(self, file_name, episode):
        path = os.path.join(self.dataset_path, file_name)
        np.save(path, episode)
        self.get_logger().info(f"{file_name} saved, which includes {len(episode)} state-action pairs.")

    def print_dataset_info(self):
        # if there are something else otherthan metadata, save
        if self.episode_idx > 0:
            self.meta_data['number_of_episodes'] = self.episode_idx
            save_dict_to_yaml(self.meta_data, os.path.join(self.dataset_path, 'meta_data.yaml'))
            self.get_logger().info(f"Dataset saved in {self.dataset_path}. {self.meta_data}")
        else:
            self.get_logger().info("Nothing saved.")

def main(args=None):
    rclpy.init(args=args)
    node = SynchronizedDatasetCollector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.print_dataset_info()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
