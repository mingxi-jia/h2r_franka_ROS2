from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    camera_name = "dave"
    # Declare launch arguments
    namespace_prefix_arg = DeclareLaunchArgument("namespace_prefix", default_value="panda_eob_calib")
    marker_id_arg = DeclareLaunchArgument("markerId", default_value="582")
    marker_size_arg = DeclareLaunchArgument("markerSize", default_value="0.1")
    eye_arg = DeclareLaunchArgument("eye", default_value="left")
    marker_frame_arg = DeclareLaunchArgument("marker_frame", default_value="aruco_marker_frame")
    ref_frame_arg = DeclareLaunchArgument("ref_frame", default_value="")
    corner_refinement_arg = DeclareLaunchArgument("corner_refinement", default_value="LINES")
    camera_frame_arg = DeclareLaunchArgument("camera_frame", default_value=f"{camera_name}_color_optical_frame")
    camera_image_topic_arg = DeclareLaunchArgument("camera_image_topic", default_value=f"/{camera_name}/color/image_raw")
    camera_info_topic_arg = DeclareLaunchArgument("camera_info_topic", default_value=f"/{camera_name}/color/camera_info")

    # Define the aruco_ros node
    aruco_node = Node(
        package="aruco_ros",
        executable="single",
        name="aruco_single",
        remappings=[
            ("/camera_info", LaunchConfiguration("camera_info_topic")),
            ("/image", LaunchConfiguration("camera_image_topic")),
        ],
        parameters=[
            {"image_is_rectified": True},
            {"marker_size": LaunchConfiguration("markerSize")},
            {"marker_id": LaunchConfiguration("markerId")},
            {"reference_frame": LaunchConfiguration("ref_frame")},
            {"camera_frame": LaunchConfiguration("camera_frame")},
            {"marker_frame": LaunchConfiguration("marker_frame")},
            {"corner_refinement": LaunchConfiguration("corner_refinement")},
        ],
        output="screen",
    )

    # Include the easy_handeye2 calibration launch file
    calibrate_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("easy_handeye2"), "launch", "calibrate.launch.py"
            )
        ),
        launch_arguments={
            "eye_on_hand": "false",
            "namespace_prefix": LaunchConfiguration("namespace_prefix"),
            "move_group": "fr3_manipulator",
            "freehand_robot_movement": "false",
            "robot_base_frame": "fr3_link0",
            "robot_effector_frame": "fr3_hand_tcp",
            "tracking_base_frame": f"{camera_name}_link",
            "tracking_marker_frame": "aruco_marker_frame",
            "calibration_type": 'eye_in_hand',
            "name": LaunchConfiguration("namespace_prefix"),
        }.items(),
    )

    # Return the LaunchDescription
    return LaunchDescription([
        namespace_prefix_arg,
        marker_id_arg,
        marker_size_arg,
        eye_arg,
        marker_frame_arg,
        ref_frame_arg,
        corner_refinement_arg,
        camera_frame_arg,
        camera_image_topic_arg,
        camera_info_topic_arg,
        aruco_node,
        calibrate_launch,
    ])
