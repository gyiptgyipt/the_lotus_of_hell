from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=PathJoinSubstitution(
            [FindPackageShare("uav_control"), "models", "policy.onnx"]
        ),
        description="Path to the exported ONNX policy",
    )

    offboard_control_node = Node(
        package="uav_control",
        executable="offboard_control_node",
        name="offboard_control",
        output="screen",
        parameters=[{"takeoff_altitude_m": 3.0}],
    )

    rl_inference_node = Node(
        package="uav_control",
        executable="rl_inference_node",
        name="rl_inference_node",
        output="screen",
        parameters=[{
            "model_path": LaunchConfiguration("model_path"),
            "pos_norm": 50.0,
            "vel_norm": 20.0,
            "max_speed_mps": 8.0,
            "max_yaw_rate_rps": 1.0,
            "watchdog_timeout_sec": 0.3,
            "capture_radius_m": 1.0,
            "control_rate_hz": 20.0,
        }],
    )

    return LaunchDescription([
        model_path_arg,
        offboard_control_node,
        rl_inference_node,
    ])
