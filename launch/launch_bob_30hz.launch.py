import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch_ros.actions import Node

def generate_launch_description():
    img_width, img_height = 640, 480
    hz = 30
    return LaunchDescription([
        # Camera nodes
        GroupAction([ Node(
                package='realsense2_camera',
                executable='realsense2_camera_node',
                name='bob_camera',
                namespace='bob',
                parameters=[{
                    'serial_no': '234322306820',
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
                    'rgb_camera.white_balance':3293.0,
                    'rgb_camera.exposure':100,
                    'enable_infra': False, 
                    'align_depth.enable': True,
                    'depth_width': img_width,
                    'depth_height': img_height,
                    'depth_fps': hz,
                    'color_width': img_width,
                    'color_height': img_height,
                    'color_fps': hz,
                    'camera_name': 'bob',
                }]
            ),
        ]),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='bob_link_broadcaster',
        #arguments=['0.55482', '0.488489', '0.638426', '0.321738', '0.30324', '-0.611973', '0.655759', 'fr3_link0', 'bob_link']
            arguments =[ '0.5618', '0.4885', '0.6324', '0.3121', '0.3116', '-0.6226', '0.6464', 'fr3_link0', 'bob_link']
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='workspace_link_broadcaster',
            arguments=['0.5', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', 'fr3_link0', 'workspace_link']
        ),
    ])
