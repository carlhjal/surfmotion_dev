from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    workspace_install_path = os.getenv("COLCON_PREFIX_PATH")
    workspace_base_path = os.path.join(workspace_install_path, "..")
    venv_python_exe = os.path.join(workspace_base_path, ".venv", "bin", "python3")

    package_name = "path_projection"
    package_path = get_package_share_directory(package_name)
    pcl_processing_path = os.path.join(package_path, "scripts", "path_projection.py")

    return LaunchDescription([
        ExecuteProcess(
            cmd=[venv_python_exe, pcl_processing_path],
            output="screen",
            shell=True
        )
    ])
