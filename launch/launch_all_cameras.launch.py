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
                name='kevin_camera',
                namespace='kevin',
                parameters=[{
                    'serial_no': 'f1420123',
                    'depth_module.profile': 'fps',
                    'rgb_camera.profile': f'{img_width}x{img_height}x{fps}',
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
                    'depth_fps': fps,
                    'color_width': img_width,
                    'color_height': img_height,
                    'color_fps': fps,
                    'camera_name': 'kevin',
                }]
            ),
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
                name='tim_camera',
                namespace='tim',
                parameters=[{
                    'serial_no': '218722271574',
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
                    'enable_infra': False, 
                    'align_depth.enable': True,
                    'depth_width': img_width,
                    'depth_height': img_height,
                    'depth_fps': fps,
                    'color_width': img_width,
                    'color_height': img_height,
                    'color_fps': fps,
                    'camera_name': 'tim',
                }]
            ),
            Node(
                package='realsense2_camera',
                executable='realsense2_camera_node',
                name='bob_camera',
                namespace='bob',
                parameters=[{
                    'serial_no': '234322306820',
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
                    'rgb_camera.white_balance':3293.0,
                    'rgb_camera.exposure':100,
                    'enable_infra': False, 
                    'align_depth.enable': True,
                    'depth_width': img_width,
                    'depth_height': img_height,
                    'depth_fps': fps,
                    'color_width': img_width,
                    'color_height': img_height,
                    'color_fps': fps,
                    'camera_name': 'bob',
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
            name='kevin_link_broadcaster',
            arguments=['0.665721', '-0.0293555', '1.00848', '0.504637', '0.490822', '-0.487034', '0.516946', 'fr3_link0', 'kevin_link']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tim_link_broadcaster',
            arguments=['0.0804052', '0.00460167', '-0.121265', '-0.0058877', '-0.70644', '-0.00265498', '0.707743', 'fr3_hand_tcp', 'tim_link']
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
            name='bob_link_broadcaster',
            arguments=['0.55482', '0.493489', '0.638426', '0.321738', '0.30324', '-0.611973', '0.655759', 'fr3_link0', 'bob_link']
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
