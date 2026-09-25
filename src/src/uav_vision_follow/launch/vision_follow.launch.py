from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'params',
            default_value=PathJoinSubstitution(
                [FindPackageShare('uav_vision_follow'), 'config', 'follower.yaml'])),
        DeclareLaunchArgument('auto_arm', default_value='true'),
        DeclareLaunchArgument(
            'model_path',
            default_value=PathJoinSubstitution(
                [FindPackageShare('uav_vision_follow'), 'models', 'drone_yolo.pt'])),
        Node(
            package='uav_vision_follow', executable='vision_follower',
            name='vision_follower', output='screen',
            parameters=[LaunchConfiguration('params'), {
                'auto_arm': LaunchConfiguration('auto_arm'),
                'model_path': LaunchConfiguration('model_path'),
            }],
        ),
    ])
