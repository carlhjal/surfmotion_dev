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

### Running the path planning script

Make sure you have sourced the workspace build files

``` bash
source install/setup.bash
```

Launch the path planner

``` bash
ros2 launch reach_planner path_projection.launch.py
```

Output paths will be saved under pkg_name_share/output.

### Running the motion planner back-ends

To run either of the back-ends, a moveit context must first be initialized:

```bash
ros2 launch moveit_ctx_launcher launch_moveit_ctx.launch.py 
```

An optional parameter pointing to a meta-info yaml in the form of:

```yaml
moveit_config_package: "ur20_custom_moveit_config"
moveit_config_name: "custom_ur"
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

The meta information yaml follows the same structure.

### Servo notes

All provided example configs come packaged with a servo_config.yaml file, this has to exist in any additional moveit_configs. The launch files expect the name "servo_config.yaml".
