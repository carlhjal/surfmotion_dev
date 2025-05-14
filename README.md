# robotic_pipecutting

A robot-agnostic package for robotic pipe cutting and welding trajectory trialing

## Installation

This section goes through all the setup that needs to be done on a clean ROS2 Jazzy install on Ubuntu 24.04 LTS.

### ROS setup

Setup a ROS workspace:

``` bash
cd ~
mkdir -p /ws_folder/src
cd ~/ws_folder
git clone https://github.com/carlhjal/robotic_pipecutting.git src/robotic_pipecutting
```

Install dependencies:

``` bash
vcs import src < src/robotic_pipecutting/dependencies.repos
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
python3 src/robotic_pipecutting/setup_venv.py
```

Or manually:

``` bash
cd ~/ws_folder
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/robotic_pipecutting/requirements.txt
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
