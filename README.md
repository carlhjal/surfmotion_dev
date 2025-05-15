# SurfMotion

A robot-agnostic package for robotic pipe cutting and welding trajectory trialing

## Installation

This section goes through all the setup that needs to be done on a clean ROS2 Jazzy install on Ubuntu 24.04 LTS.

### ROS setup

Setup a ROS workspace:

``` bash
cd ~
mkdir -p /ws_folder/src
cd ~/ws_folder
git clone https://github.com/carlhjal/surfmotion_dev.git src/surfmotion
```

Install dependencies:

``` bash
vcs import src < src/surfmotion/dependencies.repos
rosdep install --from-paths src --ignore-src -r -y
```

Build the repository

``` bash
colcon build
```

### Python Environment

It is recommended to install additional python dependencies in a virtual environment.  
This can be done either with the automated script:

``` bash
cd ~/ws_folder
python3 src/surfmotion/setup_venv.py
```

Or manually:

``` bash
cd ~/ws_folder
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/surfmotion/requirements.txt
```

## Running the path planning script

Make sure you have sourced the workspace build files

``` bash
source install/setup.bash
```

Launch the path planner, the launch file will look for the virtual environment by itself..

``` bash
ros2 launch path_projection path_projection.launch.py
```

Output paths will be saved under path_projection_share/output.

### Reach analysis

In order to run reach reachability analysis, you will need to define two yamls in the path_projection package, as well as point the path_projection.launch.py launch file to them.

config_ur20.yaml:

```yaml
# reach needs some information about the robot
robot_description_file: 
  package: "ur20_custom_description"
  file: "urdf/custom_ur.urdf.xacro"
robot_description_semantic_file: 
  package: "ur20_custom_moveit_config"
  file: "config/custom_ur.srdf"
robot_description_kinematics_file:
  package: "ur20_custom_moveit_config"
  file: config/kinematics.yaml
robot_description_joint_limits_file:
  package: "ur20_custom_moveit_config"
  file: "config/joint_limits.yaml"
config_file: 
  package: "path_projection"
  file: "config/reach_config_ur20.yaml"
results_dir:
  package: "path_projection"
  file: "output/results"
pointcloud:
  package: "path_projection"
  file: "output/test_output.pcd"
config_name: "ur20"

# Specifying the base link name of the robot model is used for creating a virtual joint between world and robot_base_frame
# Don't use this if you already know what you're doing 
robot_base_frame: "base_link"
robot_pose: 
  translation: [0.0, 0.0, 0.0]
  rotation_rpy: [0.0, 0.0, 0.0]
```

### Running the motion planner back-ends

To run either of the back-ends, a moveit context must first be initialized:

```bash
ros2 launch moveit_ctx_launcher launch_moveit_ctx.launch.py 
```

An optional parameter pointing to a meta-info yaml in the form of:

```yaml
moveit_config_package: "ur20_custom_moveit_config"
moveit_config_name: "custom_ur"
move_group_name: "ur_arm"
launch_servo: false
```

located in moveit_ctx_launcher/config can be provided. 
In order to lanch a specific robot:

```bash
ros2 launch moveit_ctx_launcher launch_moveit_ctx.launch.py meta_config_name:=ur20_meta.yaml
```

In the same spirit, the backends respectively can be launched with:

```bash
ros2 launch surfmotion_compute_cartesian cartesian_path.launch.py
ros2 launch surfmotion_servo servo.launch.py
ros2 launch surfmotion_pilz pilz.launch.py
```

The meta information yaml is expected to be located in the moveit_ctx_launcher package.

#### Collision objects

The collision object can be added to the MoveIt planning scene with:

```bash
ros2 launch object_spawner spawn_mesh.launch.py
```


### Servo notes

All provided example configs come packaged with a servo_config.yaml file, this has to exist in any additional moveit_configs. The launch files expect the name "servo_config.yaml".
