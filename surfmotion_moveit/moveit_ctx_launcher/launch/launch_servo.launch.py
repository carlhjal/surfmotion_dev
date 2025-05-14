import yaml
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from ament_index_python.packages import get_package_share_directory


def load_yaml(file_path):
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        print("error loading config yaml")
        return None


def load_servo_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path) as file:
            return yaml.safe_load(file)
    except OSError:
        return None


def launch_setup(context):
    # Get the resolved path to the meta config YAML
    meta_config_rel = LaunchConfiguration("meta_config_name").perform(context)
    package_path = get_package_share_directory("moveit_ctx_launcher")
    meta_config_path = os.path.join(package_path, "config", meta_config_rel)
    print(meta_config_path)

    # Load the meta YAML
    meta = load_yaml(meta_config_path)
    if "moveit_config_name" not in meta or "moveit_config_package" not in meta or "launch_servo" not in meta:
        raise RuntimeError("Meta config must contain 'moveit_config_name', 'moveit_config_package' and 'launch_servo'")

    # MoveIt description etc.
    moveit_config = (
        MoveItConfigsBuilder(meta["moveit_config_name"], package_name=meta["moveit_config_package"])
        .to_moveit_configs()
    )

    servo_yaml = load_servo_yaml(meta["moveit_config_package"], "config/servo_config.yaml")
    servo_params = {"moveit_servo": servo_yaml}

    servo_node = Node(
        package="moveit_servo",
        executable="servo_node",
        parameters=[
            moveit_config.to_dict(),
            servo_params,
        ],
        output="screen",
    )

    return [servo_node]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "meta_config_name",
            default_value="kuka_meta.yaml",
            description="Name of meta info YAML"
        ),
        OpaqueFunction(function=launch_setup)
    ])