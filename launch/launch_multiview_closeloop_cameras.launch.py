import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch_ros.actions import Node

def generate_launch_description():
    img_width, img_height = 640, 480
    fps = 15
    return LaunchDescription([
        # Camera nodes
        GroupAction([
            Node(
                package='realsense2_camera',
                executable='realsense2_camera_node',
                name='stuart_camera',
                namespace='stuart',
                parameters=[{
                    'serial_no': '234222301686',
                    'depth_module.profile': f'{img_width}x{img_height}x{fps}',
                    'rgb_camera.profile': f'{img_width}x{img_height}x{fps}',
                    'enable_infra': False, 
                    'enable_infra1': False,
                    'enable_infra2': False,
                    'enable_gyro': False,
                    'enable_accel': False,
                    'rgb_camera.enable_auto_exposure': False,
                    'depth_module.enable_auto_exposure': False,
                    'rgb_camera.enable_auto_white_balance': False,
                    'rgb_camera.white_balance':3330.0,
                    'rgb_camera.exposure':130,
                    'enable_infra': False, 
                    'align_depth.enable': True,
                    'depth_width': img_width,
                    'depth_height': img_height,
                    'depth_fps': fps,
                    'color_width': img_width,
                    'color_height': img_height,
                    'color_fps': fps,
                    'camera_name': 'stuart',
                }]
            ),
            Node(
                package='realsense2_camera',
                executable='realsense2_camera_node',
                name='mel_camera',
                namespace='mel',
                parameters=[{
                    'serial_no': '239222303414',
                    'depth_module.profile': f'{img_width}x{img_height}x{fps}',
                    'rgb_camera.profile': f'{img_width}x{img_height}x{fps}',
                    'enable_infra': False, 
                    'enable_infra1': False,
                    'enable_infra2': False,
                    'enable_gyro': False,
                    'enable_accel': False,
                    'rgb_camera.enable_auto_exposure': False,
                    'depth_module.enable_auto_exposure': False,
                    'rgb_camera.enable_auto_white_balance': False,
                    'rgb_camera.white_balance':3330.0,
                    'rgb_camera.exposure':100,
                    'enable_infra': False, 
                    'align_depth.enable': True,
                    'depth_width': img_width,
                    'depth_height': img_height,
                    'depth_fps': fps,
                    'color_width': img_width,
                    'color_height': img_height,
                    'color_fps': fps,
                    'camera_name': 'mel',
                }]
            ),
            Node(
                package='realsense2_camera',
                executable='realsense2_camera_node',
                name='dave_camera',
                namespace='dave',
                parameters=[{
                    'serial_no': '239222303046',
                    'depth_module.profile': f'{img_width}x{img_height}x{fps}',
                    'rgb_camera.profile': f'{img_width}x{img_height}x{fps}',
                    'enable_infra': False, 
                    'enable_infra1': False,
                    'enable_infra2': False,
                    'enable_gyro': False,
                    'enable_accel': False,
                    'rgb_camera.enable_auto_exposure': False,
                    'depth_module.enable_auto_exposure': False,
                    'rgb_camera.enable_auto_white_balance': False,
                    'rgb_camera.white_balance':3182.0,
                    'depth_module.exposure':14725,
                    'rgb_camera.exposure':100,
                    'enable_infra': False, 
                    'align_depth.enable': True,
                    'depth_width': img_width,
                    'depth_height': img_height,
                    'depth_fps': fps,
                    'color_width': img_width,
                    'color_height': img_height,
                    'color_fps': fps,
                    'camera_name': 'dave',
                }]
            )
        ]),

        # Static transforms for each camera
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='stuart_link_broadcaster',
            arguments=['0.0197533', '0.370377', '0.613283', '0.10964', '0.397813', '-0.248267', '0.876406', 'fr3_link0', 'stuart_link']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='mel_link_broadcaster',
            arguments=['0.509716', '-0.521368', '0.688352', '-0.256824', '0.252428', '0.667639', '0.651597', 'fr3_link0', 'mel_link']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='dave_link_broadcaster',
            arguments=['1.27683', '-0.0480804', '0.583822', '-0.287011', '-0.00599455', '0.957905', '-0.00244425', 'fr3_link0', 'dave_link']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='workspace_link_broadcaster',
            arguments=['0.5', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0', 'fr3_link0', 'workspace_link']
        ),
    ])
