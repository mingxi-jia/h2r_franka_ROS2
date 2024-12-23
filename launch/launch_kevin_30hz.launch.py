import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch_ros.actions import Node

def generate_launch_description():
    img_width, img_height = 640, 480
    hz = 30
    return LaunchDescription([
        # Camera nodes
        GroupAction([Node(
                package='realsense2_camera',
                executable='realsense2_camera_node',
                name='kevin_camera',
                namespace='kevin',
                parameters=[{
                    'serial_no': 'f1420123',
                    'depth_module.profile': f'{img_width}x{img_height}x{hz}',
                    'rgb_camera.profile': f'{img_width}x{img_height}x{hz}',
                    'enable_infra': False, 
                    'enable_infra1': False,
                    'enable_infra2': False,
                    'enable_gyro': False,
                    'enable_accel': False,
                    'rgb_camera.enable_auto_exposure': False,
                    'depth_module.enable_auto_exposure': False,
                    'rgb_camera.enable_auto_white_balance': False,
                    'rgb_camera.white_balance':3071.0,
                    'rgb_camera.exposure':300,
                    'align_depth.enable': True,
                    'depth_width': img_width,
                    'depth_height': img_height,
                    'depth_fps': hz,
                    'color_width': img_width,
                    'color_height': img_height,
                    'color_fps': hz,
                    'camera_name': 'kevin',
                }]
            ),
        ]),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='kevin_link_broadcaster',
            arguments=['0.665721', '-0.0293555', '1.00848', '0.504637', '0.490822', '-0.487034', '0.516946', 'fr3_link0', 'kevin_link']
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='workspace_link_broadcaster',
            arguments=['0.5', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', 'fr3_link0', 'workspace_link']
        ),
    ])