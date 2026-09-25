from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=PathJoinSubstitution(
            [FindPackageShare("uav_vision_detect"), "models", "target_yolo.onnx"]
        ),
    )
    image_topic_arg = DeclareLaunchArgument("image_topic", default_value="/camera/image_raw")
    camera_info_topic_arg = DeclareLaunchArgument(
        "camera_info_topic", default_value="/camera/camera_info"
    )
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=PathJoinSubstitution(
            [FindPackageShare("uav_vision_detect"), "config", "params.yaml"]
        ),
    )

    node = Node(
        package="uav_vision_detect",
        executable="uav_vision_detect",
        name="uav_vision_detect",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {"model_path": LaunchConfiguration("model_path")},
        ],
        remappings=[
            ("camera/image_raw", LaunchConfiguration("image_topic")),
            ("camera/camera_info", LaunchConfiguration("camera_info_topic")),
        ],
    )

    return LaunchDescription([
        model_path_arg,
        image_topic_arg,
        camera_info_topic_arg,
        params_file_arg,
        node,
    ])
