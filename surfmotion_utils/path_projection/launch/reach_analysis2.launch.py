import os
import yaml
import math
import shutil
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler, OpaqueFunction
from launch.event_handlers import OnProcessIO
from launch.substitutions import LaunchConfiguration, FindExecutable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def load_yaml(file_path):
    print(f"Attempting to load YAML from: {file_path}")
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        print("Error loading config YAML")
        return None


def get_package_file(file: dict) -> str:
    if isinstance(file, dict) and "package" in file and "file" in file:
        return os.path.join(get_package_share_directory(file["package"]), file["file"])
    return ""


def shutdown_if_done(event):
    output_text = event.text.decode("utf-8")
    if "Press enter to quit" in output_text:
        print("Reach finished. Shutting it down")
        from launch.events import Shutdown
        from launch.actions import EmitEvent
        return [EmitEvent(event=Shutdown())]


def launch_setup(context):
    use_config_yaml = True
    autoclose = False

    # Get the resolved path to the meta config YAML
    config_file = LaunchConfiguration("config_file").perform(context)
    package_path = get_package_share_directory("path_projection")
    meta_config_path = os.path.join(package_path, "config", config_file)
    print(meta_config_path)
    config = load_yaml(meta_config_path)

    if config is None:
        return []

    package_path = get_package_share_directory("path_projection")
    reach_ros_path = get_package_share_directory("reach_ros")
    reach_custom_setup_path = get_package_share_directory("path_projection")

    setup_launch_file = os.path.join(reach_custom_setup_path, "launch", "setup_reach.launch.py")
    start_launch_file = os.path.join(reach_ros_path, "launch", "start_reach.launch.py")

    results_dir = os.path.join(package_path, "output", "results")
    if os.path.exists(results_dir):
        print(f"Removing existing results directory: {results_dir}")
        shutil.rmtree(results_dir)

    robot_rotation = str(math.pi)
    robot_base_frame = config["robot_base_frame"]
    robot_translation = config["robot_pose"]["translation"]
    robot_rotation_rpy = config["robot_pose"]["rotation_rpy"]

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_world_to_base",
        arguments=[str(entry) for entry in robot_translation + robot_rotation_rpy + ["world", robot_base_frame]]
    )

    if use_config_yaml:
        robot_description_file = get_package_file(config["robot_description_file"])
        robot_description_semantic_file = get_package_file(config["robot_description_semantic_file"])
        robot_description_kinematics_file = get_package_file(config["robot_description_kinematics_file"])
        robot_description_joint_limits_file = get_package_file(config["robot_description_joint_limits_file"])
        config_file = get_package_file(config["config_file"])
        config_name = config["config_name"]
        results_dir = get_package_file(config["results_dir"])
    # else:
        # fallback hardcoded paths
        # robot_description_file = os.path.join(get_package_share_directory("ur_custom_description"), "urdf", "custom_ur_rotated.urdf.xacro")
        # robot_description_semantic_file = os.path.join(get_package_share_directory("ur_custom_moveit_config"), "config", "custom_ur.srdf")
        # robot_description_kinematics_file = os.path.join(get_package_share_directory("ur_custom_moveit_config"), "config", "kinematics.yaml")
        # robot_description_joint_limits_file = os.path.join(get_package_share_directory("ur_custom_moveit_config"), "config", "joint_limits.yaml")
        # config_file = os.path.join(package_path, "config", "reach_config.yaml")
        # config_name = "test"
        # results_dir = os.path.join(package_path, "output", "results")

    setup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(setup_launch_file),
        launch_arguments={
            "robot_description_file": robot_description_file,
            "robot_description_semantic_file": robot_description_semantic_file,
            "use_rviz": "True",
            "robot_rotation": robot_rotation
        }.items()
    )

    reach_start_launch = ExecuteProcess(
        cmd=[
            FindExecutable(name="ros2"),
            "launch",
            "reach_ros",
            "start.launch.py",
            f"robot_description_file:={robot_description_file}",
            f"robot_description_semantic_file:={robot_description_semantic_file}",
            f"robot_description_kinematics_file:={robot_description_kinematics_file}",
            f"robot_description_joint_limits_file:={robot_description_joint_limits_file}",
            f"config_file:={config_file}",
            f"config_name:={config_name}",
            f"results_dir:={results_dir}"
        ]
    )

    monitor_start_output = RegisterEventHandler(
        event_handler=OnProcessIO(
            target_action=reach_start_launch,
            on_stdout=lambda event: shutdown_if_done(event)
        )
    )

    actions = [static_tf, setup_launch, reach_start_launch]
    # actions = [setup_launch, reach_start_launch]
    if autoclose:
        actions.append(monitor_start_output)

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value="config_kuka.yaml",
            description='Path to the configuration YAML file'
        ),
        OpaqueFunction(function=launch_setup)
    ])
