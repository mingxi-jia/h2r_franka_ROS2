

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform',   required=True, type=list, help='Final end effector transform')
    parser.add_argument('--parent_mesh',  required=True, type=str, help="filename of parent object mesh")
    parser.add_argument('--child_mesh',  required=True, type=str, help="filename of child object mesh")

    panda = MoveItPy(node_name="moveit_py_planning_scene")
    panda_arm = panda.get_planning_component("panda_arm")
    planning_scene_monitor = panda.get_planning_scene_monitor()

    #load planning scene urdf 
    with planning_scene_monitor.read_write() as scene:
        parent_collision_object = CollisionObject()
        parent_collision_object.header.frame_id = "panda_link0"
        parent_collision_object.id = "parent"

        parent_pose = Pose()
        mesh_pose_in_world = #Get the origin of the mesh in the world frame
        parent_pose.position.x = mesh_pose_in_world[0]
        parent_pose.position.y = mesh_pose_in_world[1]
        parent_pose.position.z = mesh_pose_in_world[2]
        parent_pose.orientation.x = mesh_pose_in_world[3]
        parent_pose.orientation.y = mesh_pose_in_world[4]
        parent_pose.orientation.z = mesh_pose_in_world[5]
        parent_pose.orientation.w = mesh_pose_in_world[6]

        box = Mesh()
        #parameters?

        parent_collision_object.primitives.append(box)
        parent_collision_object.primitive_poses.append(box_pose)
        parent_collision_object.operation = CollisionObject.ADD

        scene.apply_collision_object(collision_object)

        child_collision_object = CollisionObject()
        child_collision_object.header.frame_id = "panda_link0"
        child_collision_object.id = "child"

        child_pose = Pose()
        mesh_pose_in_world = #Get the origin of the mesh in the world frame
        child_pose.position.x = mesh_pose_in_world[0]
        child_pose.position.y = mesh_pose_in_world[1]
        child_pose.position.z = mesh_pose_in_world[2]
        child_pose.orientation.x = mesh_pose_in_world[3]
        child_pose.orientation.y = mesh_pose_in_world[4]
        child_pose.orientation.z = mesh_pose_in_world[5]
        child_pose.orientation.w = mesh_pose_in_world[6]

        #add to the robot's end effector

        #add another one 
        scene.current_state.update()  # Important to ensure the scene is updated

    #Visualize the planning scene

    input("Check the planning scene - press enter to continue")

    #Visualize the plan

    input("Check the plan - press enter to execute on the real robot")
    move_robot(self, transform[:3], transform[3:], 0)

    input("Record final state - press enter to finish")




