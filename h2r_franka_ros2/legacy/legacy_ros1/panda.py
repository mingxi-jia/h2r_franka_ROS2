import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import CollisionObject, PlanningSceneWorld, PlanningScene
from shape_msgs.msg import SolidPrimitive
import time

GRIPPER_OFFSET = 0.1
CAMERA_OFFSET = -0.12
HOME_POSE = [-0.1727104764639174, -0.5949685050443688, 1.5450852559912969, 
             0.33045405768433256, -0.9930651649135822, -2.5002492544036263, -0.47692708008218576]
TIM_TO_TCP = [0.048, -0.012, -0.075, 0.004, -0.003, -0.707, 0.707]
GRIPPER_OPEN_WIDTH = 0.08  # meters
GRIPPER_CLOSE_WIDTH = 0.01  # meters

class PandaArmControl(Node):
    def __init__(self):
        super().__init__('fr3_arm_control_custom')
        self.goal_pub = self.create_publisher(PlanningSceneWorld, '/planning_scene', 10)
        self.cli = self.create_client(ApplyPlanningScene, 'apply_planning_scene')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for apply_planning_scene service...')
        self.req = ApplyPlanningScene.Request()
        
        self.robot_frame = "fr3_link0"
        self.tcp_frame = "fr3_link8"
        self.get_logger().info("PandaArmControl initialized")

    def init_scene(self):
        """Add collision objects to the planning scene."""

        # add wall
        wall = self.create_collision(name='wall', size=[0.4, 1.8, 1.8], xyz_pose=[-0.5, 0.0, 0.0])
        
        # Add the collision object to the planning scene
        planning_scene = PlanningScene()
        planning_scene.world.collision_objects.append(wall)
        planning_scene.is_diff = True
        # Send the request
        self.req.scene = planning_scene
        self.future = self.cli.call_async(self.req)
        
        while rclpy.ok():
            rclpy.spin_once(self)
            if self.future.done():
                try:
                    response = self.future.result()
                    self.get_logger().info('Planning scene updated successfully')
                except Exception as e:
                    self.get_logger().error(f'Service call failed: {e}')
                break

    def create_collision(self, name, size: list, xyz_pose: list):
        """Add collision objects to the planning scene."""
        collision_object = CollisionObject()
        collision_object.header.frame_id = self.tcp_frame
        collision_object.id = name
        
        # Define wall
        obj = SolidPrimitive()
        obj.type = SolidPrimitive.BOX
        obj.dimensions = [size[0], size[1], size[2]]
        obj_pose = Pose()
        obj_pose.position.x = xyz_pose[0]
        obj_pose.position.y = xyz_pose[1]
        obj_pose.position.z = xyz_pose[2]
        obj_pose.orientation.w = 1.0
        collision_object.primitives.append(obj)
        collision_object.primitive_poses.append(obj_pose)
        collision_object.pose = obj_pose
        collision_object.operation = CollisionObject.ADD
        
        return collision_object

        
    
    def go_home(self):
        """Move the robot to the home position."""
        self.panda_arm.set_start_state_to_current_state()
        self.panda_arm.set_goal_state(configuration_name="ready")
        self.plan_and_execute()
        self.get_logger().info("Moved to home position")

    def move_to_pose(self, x, y, z, rx, ry, rz):
        """Move the robot to a specified pose."""
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "panda_link0"
        pose_goal.pose.position.x = x
        pose_goal.pose.position.y = y
        pose_goal.pose.position.z = z
        pose_goal.pose.orientation.x = rx
        pose_goal.pose.orientation.y = ry
        pose_goal.pose.orientation.z = rz
        pose_goal.pose.orientation.w = 0.0  # Adjust as needed

        self.panda_arm.set_start_state_to_current_state()
        self.panda_arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="panda_link8")
        self.plan_and_execute()
        self.get_logger().info(f"Moved to pose: x={x}, y={y}, z={z}, rx={rx}, ry={ry}, rz={rz}")

    def plan_and_execute(self):
        """Helper function to plan and execute motion."""
        plan_result = self.panda_arm.plan()
        if plan_result:
            self.get_logger().info("Executing plan")
            robot_trajectory = plan_result.trajectory
            self.panda.execute(robot_trajectory, controllers=[])
        else:
            self.get_logger().error("Planning failed")
        time.sleep(1)

    def control_gripper(self, width):
        """Send a command to control the gripper."""
        command = f"width:{width}"
        self.gripper_pub.publish(String(data=command))
        self.get_logger().info(f"Gripper command sent: {command}")

def main(args=None):
    rclpy.init(args=args)
    
    node = PandaArmControl()
    # node.init_scene()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
