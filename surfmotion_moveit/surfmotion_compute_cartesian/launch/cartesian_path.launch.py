import yaml
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory

def load_yaml(file_path):
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        print("error loading config yaml")
        return None

def launch_setup(context):
    # Get the resolved path to the meta config YAML
    meta_config_rel = LaunchConfiguration("meta_config_name").perform(context)
    package_path = get_package_share_directory("moveit_ctx_launcher")
    meta_config_path = os.path.join(package_path, "config", meta_config_rel)
    print(meta_config_path)

    # Load the meta YAML
    meta = load_yaml(meta_config_path)
    if "moveit_config_name" not in meta or "moveit_config_package" not in meta or "move_group_name" not in meta:
        raise RuntimeError("Meta config must contain 'moveit_config_name', 'moveit_config_package' and 'move_group_name'")

    # Build the moveit config
    moveit_config = (
        MoveItConfigsBuilder(meta["moveit_config_name"], package_name=meta["moveit_config_package"])
        .to_moveit_configs()
    )
    print("MOVEIT CONFIG PARAMS:", moveit_config.to_dict().keys())
    return [
        Node(
            package="surfmotion_compute_cartesian",
            executable="cartesian_path_node",
            name="cartesian_path_node",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {  # If we use moveit configs builder we have to override these parameters
                    "publish_planning_scene":   False,
                    "publish_state_updates":    False,
                    "publish_geometry_updates": False,
                },
                { # And manually pass this parameter
                    "move_group": meta["move_group_name"]
                }

            ]
        )
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "meta_config_name",
            default_value="fanuc_meta.yaml",
            description="Name of meta info YAML"
        ),
        OpaqueFunction(function=launch_setup)
    ])