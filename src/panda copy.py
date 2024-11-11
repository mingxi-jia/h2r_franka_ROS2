import rclpy
from rclpy.node import Node
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene, CollisionObject
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive

class PlanningSceneUpdater(Node):
    def __init__(self):
        super().__init__('planning_scene_updater')
        self.cli = self.create_client(ApplyPlanningScene, 'apply_planning_scene')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for apply_planning_scene service...')
        self.req = ApplyPlanningScene.Request()

    def send_request(self):
        # Initialize a PlanningScene message
        planning_scene = PlanningScene()
        planning_scene.is_diff = True

        # Add multiple objects
        objects = [
            {
                'id': 'box1',
                'type': SolidPrimitive.BOX,
                'dimensions': [0.1, 0.1, 0.1],
                'position': (0.5, 0.0, 0.5),
            },
            {
                'id': 'box2',
                'type': SolidPrimitive.BOX,
                'dimensions': [0.2, 0.2, 0.2],
                'position': (0.3, 0.3, 0.3),
            },
            {
                'id': 'cylinder1',
                'type': SolidPrimitive.CYLINDER,
                'dimensions': [0.5, 0.05],  # [height, radius]
                'position': (0.6, -0.2, 0.5),
            },
        ]

        for obj in objects:
            collision_object = CollisionObject()
            collision_object.id = obj['id']
            collision_object.header.frame_id = 'fr3_link0'

            primitive = SolidPrimitive()
            primitive.type = obj['type']
            primitive.dimensions = obj['dimensions']

            pose = Pose()
            pose.position.x = obj['position'][0]
            pose.position.y = obj['position'][1]
            pose.position.z = obj['position'][2]
            pose.orientation.w = 1.0

            collision_object.primitives.append(primitive)
            collision_object.primitive_poses.append(pose)
            collision_object.operation = CollisionObject.ADD

            planning_scene.world.collision_objects.append(collision_object)

        # Send the request
        self.req.scene = planning_scene
        self.future = self.cli.call_async(self.req)

def main(args=None):
    rclpy.init(args=args)
    node = PlanningSceneUpdater()
    node.send_request()

    while rclpy.ok():
        rclpy.spin_once(node)
        if node.future.done():
            try:
                response = node.future.result()
                node.get_logger().info('Planning scene updated successfully')
            except Exception as e:
                node.get_logger().error(f'Service call failed: {e}')
            break

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
