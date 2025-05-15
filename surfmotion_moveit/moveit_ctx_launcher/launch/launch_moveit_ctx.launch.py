import yaml
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils.launch_utils import DeclareBooleanLaunchArg
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import TimerAction, ExecuteProcess
from launch.conditions import IfCondition

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
    if "moveit_config_name" not in meta or "moveit_config_package" not in meta or "launch_servo" not in meta:
        raise RuntimeError("Meta config must contain 'moveit_config_name', 'moveit_config_package' and 'launch_servo'")

    # Build the moveit config
    moveit_config = (
        MoveItConfigsBuilder(meta["moveit_config_name"], package_name=meta["moveit_config_package"])
        .to_moveit_configs()
    )

    # ld = LaunchDescription()
    actions = []
    actions.append(
        DeclareLaunchArgument("use_sim_time", default_value="false")
    )

    actions.append(DeclareBooleanLaunchArg("use_rviz", default_value=True))
    virtual_joints_launch = (
        moveit_config.package_path / "launch/static_virtual_joint_tfs.launch.py"
    )

    if virtual_joints_launch.exists():
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(virtual_joints_launch)),
            )
        )

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(moveit_config.package_path / "launch/rsp.launch.py")
            ),
            # to get 100 hz tf messages
            launch_arguments={"publish_frequency": "400.0"}.items(),
        )
    )
    
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "use_sim_time": False,
                "publish_robot_description": True,
                "publish_robot_description_semantic": True,
            },
        ],
    )
    actions.append(move_group_node)

    actions.append(
        TimerAction(
            period=2.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        str(moveit_config.package_path / "launch/moveit_rviz.launch.py")
                    ),
                    condition=IfCondition(LaunchConfiguration("use_rviz")),
                )
            ],
        )
    )

    # Fake joint driver
    actions.append(
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[
                str(moveit_config.package_path / "config/ros2_controllers.yaml"),
            ],
            remappings=[
                ("/controller_manager/robot_description", "/robot_description"),
            ],
        )
    )

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(moveit_config.package_path / "launch/spawn_controllers.launch.py")
            ),
        )
    )

    if meta["launch_servo"]:
        actions.append(
            TimerAction(
                period=4.0,
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            os.path.join(package_path, "launch/launch_servo.launch.py")
                        ),
                        condition=IfCondition(LaunchConfiguration("use_rviz")),
                    )
                ],
            )
        )
 
        actions.append(
            TimerAction(
                period=8.0, # wait for servo to set up
                actions=[
                    ExecuteProcess(
                        cmd=[
                            "ros2", "service", "call",
                            "/servo_node/switch_command_type",
                            "moveit_msgs/srv/ServoCommandType",
                            "{command_type: 1}"
                        ],
                        output="screen"
                    )
                ]
            )
        )

    return actions

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "meta_config_name",
            default_value="fanuc_meta.yaml",
            description="Name of meta info YAML"
        ),
        OpaqueFunction(function=launch_setup)
    ])