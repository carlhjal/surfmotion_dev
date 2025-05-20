#include "trajectory_seed_generator/seed_gen.hpp"
#include <memory>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_monitor/planning_scene_monitor.hpp>
#include <moveit/robot_model_loader/robot_model_loader.hpp>
#include <moveit/robot_state/cartesian_interpolator.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/moveit_cpp/moveit_cpp.hpp>
#include <moveit/planning_scene_monitor/planning_scene_monitor.hpp>

// MAKE SURE THAT 
// {  
//     'publish_planning_scene':   False,
//     'publish_state_updates':    False,
//     'publish_geometry_updates': False,
// }
// HAVE BEEN SET AS NODE PARAMETERS

std::vector<double> get_viable_seed_state(
    const rclcpp::Node::SharedPtr &node,
    const std::vector<geometry_msgs::msg::Pose> &waypoints,
    const std::string &group_name,
    double eef_step,
    uint32_t num_trials,
    double ik_timeout
) {
    auto move_group = moveit::planning_interface::MoveGroupInterface(node, group_name);
    auto model = move_group.getRobotModel();
    // auto state = *move_group.getCurrentState();
    auto joint_model_group = model->getJointModelGroup(group_name);
    const auto current_pose = move_group.getCurrentPose().pose;

    auto psm = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(node, "robot_description");

    psm->startSceneMonitor("/monitored_planning_scene");
    psm->stopPublishingPlanningScene(); 
    psm->waitForCurrentRobotState(rclcpp::Time(0), 1.0); 
    planning_scene_monitor::LockedPlanningSceneRO scene(psm);

    std::string plugin_name;
    if (!node->get_parameter("robot_description_kinematics." + group_name + ".kinematics_solver", plugin_name))
    RCLCPP_ERROR(node->get_logger(), "No IK plugin declared for %s", group_name.c_str());

    if (!joint_model_group->getSolverInstance()) {
    RCLCPP_ERROR(node->get_logger(),
                "The kinematics plugin for group '%s' did not load.",
                group_name.c_str());
    return {}; // calling setFromIK() would seg‑fault
    }
    auto is_valid_quat = [](const geometry_msgs::msg::Quaternion& q) {
    return std::abs(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z - 1.0) < 1e-3;
    };

    const auto& pose = waypoints[0];
    if (!is_valid_quat(pose.orientation)) {
    RCLCPP_ERROR(node->get_logger(), "Target pose has an invalid quaternion");
    return {};
    }
    
    const auto &target_pose = waypoints[0];
    for (size_t i = 0; i < num_trials; i++) {
        moveit::core::RobotState state(model);
        state.setToRandomPositions(joint_model_group);
        if (!state.setFromIK(joint_model_group, target_pose, ik_timeout))  {
                RCLCPP_WARN(node->get_logger(),"setfromik failed");
            continue;
        }

        
        if (scene->isStateColliding(state, group_name)) {
            RCLCPP_WARN(node->get_logger(),"isstatecolliding failed");
            continue;
        } 

        std::vector<double> seed;
        state.copyJointGroupPositions(joint_model_group, seed);

        state.update();
        move_group.setStartState(state);

        moveit_msgs::msg::RobotTrajectory traj;
        double fraction = move_group.computeCartesianPath(waypoints, eef_step, traj);
        RCLCPP_WARN(node->get_logger(),"fraction: %f", fraction);

        if (fraction == 1.0) {
            RCLCPP_WARN(node->get_logger(),"Found a suitable seed state");
            return seed;
        }
    }

    return {};
}
