import os
import yaml
import launch
import shutil
import math
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, EmitEvent, ExecuteProcess, DeclareLaunchArgument
from launch.event_handlers import OnProcessIO, OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import FindExecutable, PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory

parameters = [
  {'name': 'config', 'description':'Path to the configuration yaml', 'default': os.path.join(get_package_share_directory('path_projection'), 'config', 'config_ur20.yaml')},
]

# def declare_launch_arguments():
#     return [DeclareLaunchArgument(
#                 entry['name'],
#                 description=entry['description'],
#                 default_value=entry['default']
#             ) for entry in parameters]

def load_yaml(file_path):
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        print("error loading config yaml")
        return None
    
def get_package_file(file: dict) -> os.path:
    """
    Takes a dict containing a package and filename, returns global path 
    """
    if isinstance(file, dict) and "package" in file and "file" in file:
        print(f"full path: {os.path.join(get_package_share_directory(file["package"]), file["file"])}")
        return os.path.join(get_package_share_directory(file["package"]), file["file"])

def generate_launch_description():
    autoclose = True
    use_config_yaml = True
    
    config_test = LaunchConfiguration("config")
    print(config_test)

    package_name = "path_projection"
    package_path = get_package_share_directory(package_name)

    reach_ros_package_name = "reach_ros"
    reach_ros_path = get_package_share_directory(reach_ros_package_name)
    
    reach_custom_setup_path = get_package_share_directory("path_projection")

    setup_launch_file = os.path.join(reach_custom_setup_path, "launch", "setup_reach.launch.py")
    start_launch_file = os.path.join(reach_ros_path, "launch", "start_reach.launch.py")

    results_dir = os.path.join(package_path, "output", "results")

    # Delete directory if it exists
    if os.path.exists(results_dir):
        print(f"Removing existing results directory: {results_dir}")
        shutil.rmtree(results_dir)
    
    config_file = os.path.join(package_path, "config", "config_ur20.yaml")
    config = load_yaml(config_file)
    
    robot_rotation = str(math.pi)

    robot_base_frame = config["robot_base_frame"]
    robot_translation = config["robot_pose"]["translation"]
    robot_rotation_rpy = config["robot_pose"]["rotation_rpy"]

    print([str(entry) for entry in robot_translation + robot_rotation_rpy + ["world"] + [robot_base_frame]])

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_world_to_base",
        arguments=[str(entry) for entry in robot_translation + robot_rotation_rpy + ["world"] + [robot_base_frame]]
    )

    if use_config_yaml:
        robot_description_file = get_package_file(config["robot_description_file"])
        robot_description_semantic_file = get_package_file(config["robot_description_semantic_file"])
        robot_description_kinematics_file = get_package_file(config["robot_description_kinematics_file"])
        robot_description_joint_limits_file = get_package_file(config["robot_description_joint_limits_file"])
        config_file = get_package_file(config["config_file"])
        config_name = config["config_name"]
        results_dir = get_package_file(config["results_dir"])
    else:
        robot_description_file = os.path.join(get_package_share_directory("ur_custom_description"), "urdf", "custom_ur_rotated.urdf.xacro")
        robot_description_semantic_file = os.path.join(get_package_share_directory("ur_custom_moveit_config"), "config", "custom_ur.srdf")
        robot_description_kinematics_file = os.path.join(get_package_share_directory("ur_custom_moveit_config"), "config", "kinematics.yaml")
        robot_description_joint_limits_file = os.path.join(get_package_share_directory("ur_custom_moveit_config"), "config", "joint_limits.yaml")
        config_file = os.path.join(package_path, "config", "reach_config.yaml")
        config_name = "test"
        results_dir = os.path.join(package_path, "output", "results")
    
    setup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(setup_launch_file),
        launch_arguments = {
            "robot_description_file": robot_description_file,
            "robot_description_semantic_file": robot_description_semantic_file,
            "use_rviz": "True",
            "robot_rotation": robot_rotation
        }.items()
    )

    reach_start_launch = ExecuteProcess(
        cmd=[FindExecutable(name="ros2"),
             "launch",
             reach_ros_package_name,
             "start.launch.py",
             f"robot_description_file:={robot_description_file}",
             f"robot_description_semantic_file:={robot_description_semantic_file}",
             f"robot_description_kinematics_file:={robot_description_kinematics_file}",
             f"robot_description_joint_limits_file:={robot_description_joint_limits_file}",
             f"config_file:={config_file}",
             f"config_name:={config_name}",
             f"results_dir:={results_dir}"]
    )

    monitor_start_output = RegisterEventHandler(
        event_handler=OnProcessIO(
            target_action=reach_start_launch,
            on_stdout=lambda event: shutdown_if_done(event)
        )
    )

    if autoclose:
        return launch.LaunchDescription([
            static_tf,
            setup_launch,
            reach_start_launch,
            monitor_start_output
        ])
    else:
        return launch.LaunchDescription([
            static_tf,
            setup_launch,
            reach_start_launch,
        ])
    

def shutdown_if_done(event: RegisterEventHandler):
    output_text = event.text.decode("utf-8")
    if "Press enter to quit" in output_text:
        print("Reach finished. Shutting it down")
        return [EmitEvent(event=launch.events.shutdown())]