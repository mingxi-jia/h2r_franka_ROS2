
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from std_msgs.msg import Header
from shape_msgs.msg import Mesh, MeshTriangle

from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
)

from rclpy.node import Node
import rclpy


class CollisionMeshHandler(Node):
    def __init__(
        self,
        base_link_name,
    ):
        super().__init__('collision_mesh_handler')
        self.__base_link_name = base_link_name
        self.__collision_object_publisher = self.create_publisher(
            CollisionObject, "/collision_object", 10
        )
        self.__attached_collision_object_publisher = self.create_publisher(
            AttachedCollisionObject, "/attached_collision_object", 10
        )

    #adapted from pymoveit2
    def add_collision_mesh(
        self,
        filepath: Optional[str],
        id: str,
        pose: Optional[Union[PoseStamped, Pose]] = None,
        position: Optional[Union[Point, Tuple[float, float, float]]] = None,
        quat_xyzw: Optional[
            Union[Quaternion, Tuple[float, float, float, float]]
        ] = None,
        frame_id: Optional[str] = None,
        operation: int = CollisionObject.ADD,
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        mesh: Optional[Any] = None,
    ):
        """
        Add collision object with a mesh geometry. Either `filepath` must be
        specified or `mesh` must be provided.
        Note: This function required 'trimesh' Python module to be installed.
        """

        # Load the mesh
        try:
            import trimesh
        except ImportError as err:
            raise ImportError(
                "Python module 'trimesh' not found! Please install it manually in order "
                "to add collision objects into the MoveIt 2 planning scene."
            ) from err

        # Check the parameters
        if (pose is None) and (position is None or quat_xyzw is None):
            raise ValueError(
                "Either `pose` or `position` and `quat_xyzw` must be specified!"
            )
        if (filepath is None and mesh is None) or (
            filepath is not None and mesh is not None
        ):
            raise ValueError("Exactly one of `filepath` or `mesh` must be specified!")
        if mesh is not None and not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("`mesh` must be an instance of `trimesh.Trimesh`!")

        if isinstance(pose, PoseStamped):
            pose_stamped = pose
        elif isinstance(pose, Pose):
            pose_stamped = PoseStamped(
                header=Header(
                    stamp=self._node.get_clock().now().to_msg(),
                    frame_id=(
                        frame_id if frame_id is not None else self.__base_link_name
                    ),
                ),
                pose=pose,
            )
        else:
            if not isinstance(position, Point):
                position = Point(
                    x=float(position[0]), y=float(position[1]), z=float(position[2])
                )
            if not isinstance(quat_xyzw, Quaternion):
                quat_xyzw = Quaternion(
                    x=float(quat_xyzw[0]),
                    y=float(quat_xyzw[1]),
                    z=float(quat_xyzw[2]),
                    w=float(quat_xyzw[3]),
                )
            pose_stamped = PoseStamped(
                header=Header(
                    stamp=self.get_clock().now().to_msg(),
                    frame_id=(
                        frame_id if frame_id is not None else self.__base_link_name
                    ),
                ),
                pose=Pose(position=position, orientation=quat_xyzw),
            )

        msg = CollisionObject(
            header=pose_stamped.header,
            id=id,
            operation=operation,
            pose=pose_stamped.pose,
        )

        if filepath is not None:
            mesh = trimesh.load(filepath)

        # Scale the mesh
        if isinstance(scale, float):
            scale = (scale, scale, scale)
        if not (scale[0] == scale[1] == scale[2] == 1.0):
            # If the mesh was passed in as a parameter, make a copy of it to
            # avoid transforming the original.
            if filepath is None:
                mesh = mesh.copy()
            # Transform the mesh
            transform = np.eye(4)
            np.fill_diagonal(transform, scale)
            mesh.apply_transform(transform)
        mesh.show()

        msg.meshes.append(
            Mesh(
                triangles=[MeshTriangle(vertex_indices=face) for face in mesh.faces],
                vertices=[
                    Point(x=vert[0], y=vert[1], z=vert[2]) for vert in mesh.vertices
                ],
            )
        )

        self.__collision_object_publisher.publish(msg)

    #adapted from pymoveit2
    def remove_collision_mesh(self, id):
        """
        Remove collision mesh specified by its `id`.
        """

        msg = CollisionObject()
        msg.id = id
        msg.operation = CollisionObject.REMOVE
        msg.header.stamp = self.get_clock().now().to_msg()
        self.__collision_object_publisher.publish(msg)

    def attach_collision_mesh(
        self,
        id: str,
        link_name: Optional[str] = None,
        touch_links: List[str] = [],
        weight: float = 0.0,
    ):
        """
        Attach collision object to the robot.
        """

        if link_name is None:
            link_name = self.__end_effector_name

        msg = AttachedCollisionObject(
            object=CollisionObject(id=id, operation=CollisionObject.ADD)
        )
        msg.link_name = link_name
        msg.touch_links = touch_links
        msg.weight = weight

        self.__attached_collision_object_publisher.publish(msg)

    def detach_collision_mesh(self, id: int):
        """
        Detach collision object from the robot.
        """

        msg = AttachedCollisionObject(
            object=CollisionObject(id=id, operation=CollisionObject.REMOVE)
        )
        self.__attached_collision_object_publisher.publish(msg)


    def add_attached_collision_mesh(
        self,
        filepath: Optional[str],
        id: str,
        pose: Optional[Union[PoseStamped, Pose]] = None,
        position: Optional[Union[Point, Tuple[float, float, float]]] = None,
        quat_xyzw: Optional[
            Union[Quaternion, Tuple[float, float, float, float]]
        ] = None,
        frame_id: Optional[str] = None,
        operation: int = CollisionObject.ADD,
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        mesh: Optional[Any] = None,
        ):

        self.add_collision_mesh(filepath, id, pose, position, quat_xyzw, frame_id, operation, scale, mesh)
        self.attach_collision_mesh(id, link_name = 'fr3_hand_tcp')
        
if __name__=="__main__":
    parent_mesh_file = '/home/mingxi/code/h2r_franka_ROS2_skye/src/rack_parent.obj'
    child_mesh_file = '/home/mingxi/code/h2r_franka_ROS2_skye/src/mug_child.obj'
    rclpy.init()

    mesh_handler = CollisionMeshHandler('fr3_link0')
    try: 
        print("collision mesh load")
        mesh_handler.add_collision_mesh(parent_mesh_file,
                              'parent',
                              position = [0,0,0],
                              quat_xyzw = [0,0,0,1])
        # mesh_handler.add_collision_mesh(child_mesh_file,
        #                       'child',
        #                       position = [0,0,0],
        #                       quat_xyzw = [0,0,0,1])
    except KeyboardInterrupt:
        pass
    finally:
        # this is needed because of ROS2 mechanism.
        # without destroy_node(), it somehow wont work if you restart the program
        mesh_handler.destroy_node()
        rclpy.shutdown()