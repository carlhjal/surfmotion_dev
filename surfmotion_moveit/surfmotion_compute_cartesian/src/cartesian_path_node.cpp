#include <chrono>
#include <fstream>
#include <cstdlib>
#include <memory>
#include <moveit/planning_scene_monitor/planning_scene_monitor.hpp>
#include <rclcpp/executors.hpp>
#include <rclcpp/future_return_code.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <unordered_map>
#include <vector>
#include <jsoncpp/json/json.h>
#include "geometry_msgs/msg/pose.hpp"
#include "moveit/move_group_interface/move_group_interface.h"
#include "trajectory_seed_generator/seed_gen.hpp"

std::vector<geometry_msgs::msg::Pose> poses_from_json(const std::string& filename) {
    std::vector<geometry_msgs::msg::Pose> poses;
    std::ifstream file(filename, std::ifstream::binary);

    if (!file.is_open()) {
        std::cerr << "Error opening file" << filename << std::endl;
        return poses;
    }

    Json::Value root;
    file >> root;

    for (const auto& pose_data : root) {
        geometry_msgs::msg::Pose pose;
        pose.position.x = pose_data["position"]["x"].asDouble();
        pose.position.y = pose_data["position"]["y"].asDouble();
        pose.position.z = pose_data["position"]["z"].asDouble();
        pose.orientation.x = pose_data["orientation"]["x"].asDouble();
        pose.orientation.y = pose_data["orientation"]["y"].asDouble();
        pose.orientation.z = pose_data["orientation"]["z"].asDouble();
        pose.orientation.w = pose_data["orientation"]["w"].asDouble();
        
        poses.push_back(pose);
    }
    return poses;
}

std::unordered_map<std::string, double> joint_state_from_json(const std::string& filename) {
    std::unordered_map<std::string, double> joint_states;
    std::ifstream file(filename, std::ifstream::binary);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return joint_states;
    }

    Json::Value root;
    file >> root;

    for (const auto& joint_data : root) {
        const Json::Value& joint_state = joint_data["joint_state"];
        for (const auto& joint : joint_state.getMemberNames()) {
            joint_states[joint] = joint_state[joint].asDouble();
        }
    }
    return joint_states;
}

using namespace std::chrono_literals;

int main(int argc, char * argv[]) { 
    std::string poses_filename = "/home/carl/thesis/install/reach_planner/share/reach_planner/output/poses.json";
    // std::string joint_state_filename = "/home/carl/thesis/thesis_ws/install/reach_planner/share/reach_planner/output/joint_state.json";
    std::vector<geometry_msgs::msg::Pose> poses = poses_from_json(poses_filename);
    // std::unordered_map<std::string, double> seed_state = joint_state_from_json(joint_state_filename);

    rclcpp::init(argc, argv);
    auto const node = std::make_shared<rclcpp::Node>("cartesian_path", rclcpp::NodeOptions());
    auto const logger = node->get_logger();
    rclcpp::executors::SingleThreadedExecutor executor;
    
    executor.add_node(node);
    auto spinner = std::thread([&executor]() { executor.spin(); });
    auto move_group_interface = moveit::planning_interface::MoveGroupInterface(node, "ur_arm");

    move_group_interface.setPlanningPipelineId("ompl");
    move_group_interface.setPlannerId("RRTConnectkConfigDefault");  
    move_group_interface.setPlanningTime(15.0);
    move_group_interface.setMaxVelocityScalingFactor(0.8);
    move_group_interface.setMaxAccelerationScalingFactor(0.8);

//   std::vector<double> joint_positions;
//   for (const auto& joint_name : move_group_interface.getJointNames()) {
//     joint_positions.push_back(seed_state[joint_name]);
//   }

    // set a seed state, hope that it works?
    // geometry_msgs::msg::PoseStamped target_pose;
    // target_pose.header.frame_id = "ur20_base_link";
    // target_pose.pose.position.x = poses[0].position.x;
    // target_pose.pose.position.y = poses[0].position.y;
    // target_pose.pose.position.z = poses[0].position.z;
    // target_pose.pose.orientation.x = poses[0].orientation.x;
    // target_pose.pose.orientation.y = poses[0].orientation.y;
    // target_pose.pose.orientation.z = poses[0].orientation.z;
    // target_pose.pose.orientation.w = poses[0].orientation.w;
    // move_group_interface.setPoseTarget(target_pose);
    // auto const [success, plan] = [&move_group_interface] {
    //   moveit::planning_interface::MoveGroupInterface::Plan msg;
    //   auto const ok = static_cast<bool>(move_group_interface.plan(msg));
    //   return std::make_pair(ok, msg);
    // }();
    // if (success)
    // {
    //   RCLCPP_INFO(logger, "Planning successful! Executing plan...");
    //   move_group_interface.execute(plan);
    // }
    // else
    // {
    //   RCLCPP_ERROR(logger, "Planning failed!");
    //   rclcpp::shutdown();
    //   return -1;
    // }

    // get a working seed state and set it
    std::vector<double> seed_state = get_viable_seed_state(node, poses, "ur_arm", 100, 100, 10);
    if (seed_state.empty()) {
        RCLCPP_WARN(logger,"failed generating a viable seed state");
        rclcpp::shutdown();
        return -1;
    }

    move_group_interface.setJointValueTarget(seed_state);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group_interface.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    RCLCPP_INFO(logger, "successfully planned a plan to get to the seed state");

    if (success) {
        move_group_interface.execute(plan);
    } else {
        RCLCPP_ERROR(logger, "Planning to seed state failed");
        rclcpp::shutdown();
        return -1;
    }


    move_group_interface.setMaxVelocityScalingFactor(0.1);
    move_group_interface.setMaxAccelerationScalingFactor(0.1);
    const double jump_threshold = 0.0;
    const double eef_step = 0.02;

    // for (double eef_step = 0.001; eef_step < 0.2; eef_step = eef_step+0.001) {
      // for (double jump_threshold = 0.0; jump_threshold < 0.1; jump_threshold = jump_threshold+0.01) {
    
    moveit_msgs::msg::RobotTrajectory trajectory;
    double fraction = move_group_interface.computeCartesianPath(poses, eef_step, trajectory, true);
    RCLCPP_INFO(logger, "Visualizing Cartesian path plan (%.2f%% achieved), eef_step: %f", fraction * 100.0, eef_step);

    if(fraction == 1){
        move_group_interface.execute(trajectory);
    }

    
    rclcpp::shutdown();
    spinner.join();
    return 0;
}
