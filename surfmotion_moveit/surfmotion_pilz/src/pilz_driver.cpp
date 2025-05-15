#include <chrono>
#include <fstream>
#include <cstdlib>
#include <memory>
#include <moveit/kinematic_constraints/utils.hpp>
#include <moveit/planning_scene_monitor/planning_scene_monitor.hpp>
#include <rclcpp/executors.hpp>
#include <rclcpp/future_return_code.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <unordered_map>
#include <vector>
#include <jsoncpp/json/json.h>
#include "geometry_msgs/msg/pose.hpp"
#include "moveit/move_group_interface/move_group_interface.hpp"
#include "trajectory_seed_generator/seed_gen.hpp"
#include "std_msgs/msg/bool.hpp"

#include <moveit_msgs/action/move_group_sequence.hpp>
#include <moveit_msgs/msg/motion_sequence_item.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <cmath>
#include <filesystem>
#include "ament_index_cpp/get_package_share_directory.hpp"

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
    std::string poses_filename = std::filesystem::path(ament_index_cpp::get_package_share_directory("path_projection")) / "output" / "poses.json";
    std::vector<geometry_msgs::msg::Pose> poses = poses_from_json(poses_filename);

    rclcpp::init(argc, argv);
    auto const node = std::make_shared<rclcpp::Node>("cartesian_path", rclcpp::NodeOptions());
    auto const logger = node->get_logger();
    rclcpp::executors::SingleThreadedExecutor executor;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr logging_trigger_pub_;
    logging_trigger_pub_ = node->create_publisher<std_msgs::msg::Bool>("/start_logging", 10);
    executor.add_node(node);
    auto spinner = std::thread([&executor]() { executor.spin(); });
    auto group_name = node->declare_parameter<std::string>("move_group", "");
    if (group_name.empty()) {
    throw std::runtime_error("Missing required parameter: 'move_group'");
    }
    auto move_group = moveit::planning_interface::MoveGroupInterface(node, group_name);

    move_group.setPlanningPipelineId("ompl");
    move_group.setPlannerId("RRTConnectkConfigDefault");  
    move_group.setPlanningTime(15.0);
    move_group.setMaxVelocityScalingFactor(0.8);
    move_group.setMaxAccelerationScalingFactor(0.8);

    // get a working seed state and set it
    std::vector<double> seed_state = get_viable_seed_state(node, poses, group_name, 100, 100, 10);
    if (seed_state.empty()) {
        RCLCPP_WARN(logger,"failed generating a viable seed state");
        rclcpp::shutdown();
        return -1;
    }

    move_group.setJointValueTarget(seed_state);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    
    if (success) {
        RCLCPP_INFO(logger, "successfully planned a plan to get to the seed state");
        move_group.execute(plan);
    } else {
        RCLCPP_ERROR(logger, "Planning to seed state failed");
        rclcpp::shutdown();
        return -1;
    }

    move_group.setPlanningPipelineId("pilz_industrial_motion_planner");   // NEW
    move_group.setMaxVelocityScalingFactor(0.1);
    move_group.setMaxAccelerationScalingFactor(0.1);

    // -------- Build MotionSequenceRequest --------
    const std::string tool_link = move_group.getEndEffectorLink();
    const std::string planning_frame = move_group.getPlanningFrame();
    const auto now = node->get_clock()->now();
    
    auto distance = [](const geometry_msgs::msg::Pose& a,
                      const geometry_msgs::msg::Pose& b)
    {
      double dx = a.position.x - b.position.x;
      double dy = a.position.y - b.position.y;
      double dz = a.position.z - b.position.z;
      return std::sqrt(dx*dx + dy*dy + dz*dz);
    };
    // for (size_t i = 0; i < poses.size(); ++i) {
    //   RCLCPP_WARN(logger, "dist between points: %f", distance(poses[i], poses[i + 1]));
    // }

    moveit_msgs::msg::MotionSequenceRequest seq;
    seq.items.reserve(poses.size());

    for (size_t i = 0; i < poses.size(); ++i)
    {
        geometry_msgs::msg::PoseStamped ps;
        ps.header.frame_id = planning_frame;
        ps.header.stamp    = now;
        ps.pose            = poses[i];
        
        moveit_msgs::msg::MotionSequenceItem item;
        item.req.group_name  = group_name;
        item.req.planner_id  = "LIN";
        item.req.pipeline_id = "pilz_industrial_motion_planner";
        item.req.max_velocity_scaling_factor     = 0.07;
        item.req.max_acceleration_scaling_factor = 0.07;

        item.req.goal_constraints.emplace_back(kinematic_constraints::constructGoalConstraints(tool_link, ps));

        item.blend_radius = 0.0;
        if (i > 0 && i + 1 < poses.size() - 1) {
            item.blend_radius = distance(poses[i], poses[i + 1]) / 4.0;
        }

        seq.items.emplace_back(item);
    }
    
    auto client = rclcpp_action::create_client<moveit_msgs::action::MoveGroupSequence>(node, "/sequence_move_group");

    if (!client->wait_for_action_server(5s))
    {
      RCLCPP_ERROR(logger, "Sequence action server not available");
      return -1;
    }


    // Tell the logging node to start logging joint_states
    auto msg = std_msgs::msg::Bool();
    msg.data = true;
    logging_trigger_pub_->publish(msg);
    rclcpp::sleep_for(std::chrono::milliseconds(200));  


    moveit_msgs::action::MoveGroupSequence::Goal goal;
    goal.request           = seq;
    goal.planning_options.plan_only = false;
    auto gh     = client->async_send_goal(goal).get();          // wait for accept
    auto result = client->async_get_result(gh).get();           // wait for done

    if (result.result->response.error_code.val == result.result->response.error_code.SUCCESS)
      RCLCPP_INFO(logger, "sequence finished");
    else
      RCLCPP_ERROR(logger, "sequence failed");
    
    msg.data = false;
    logging_trigger_pub_->publish(msg);
    rclcpp::sleep_for(std::chrono::milliseconds(200));  

    rclcpp::shutdown();
    spinner.join();
    return 0;
}
