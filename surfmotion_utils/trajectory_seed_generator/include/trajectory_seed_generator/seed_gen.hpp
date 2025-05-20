/**
 * @file seed_gen.hpp
 * @author Carl Hjalmar Love Hult
 * @brief Attempts to find a viable seed joint state for a cartesian trajectory
 * @version 0.1
 * @date 2025-05-09
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <moveit/kinematics_base/kinematics_base.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <vector>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/planning_interface/planning_interface.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit/kinematic_constraints/kinematic_constraint.hpp>
#include <moveit/kinematic_constraints/utils.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>


/**
 * @brief Returns a vector of joint states for a seed state that will allow execution of a cartesian trajectory
 * 
 * @param current_pose 
 * @param waypoints 
 * @param scene 
 * @param model 
 * @param move_group 
 * @param group_name 
 * @param eef_step 
 * @param num_trials 
 * @param ik_timeout 
 * @return std::vector<double> 
 */
std::vector<double> get_viable_seed_state(
    const rclcpp::Node::SharedPtr &node,
    const std::vector<geometry_msgs::msg::Pose> &waypoints,
    const std::string &group_name,
    double eef_step,
    uint32_t num_trials,
    double ik_timeout
);