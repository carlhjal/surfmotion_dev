#include <cmath>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/parameter.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors.hpp>
#include <rclcpp/logging.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit_servo/moveit_servo/servo.hpp>
#include <moveit_servo/moveit_servo/utils/common.hpp>
#include <chrono>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <memory>
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "std_msgs/msg/bool.hpp"
#include "trajectory_seed_generator/seed_gen.hpp"
#include <filesystem>
#include "ament_index_cpp/get_package_share_directory.hpp"

using namespace std::chrono_literals;

bool close_enough(
  const geometry_msgs::msg::Pose &pose, 
  const geometry_msgs::msg::Pose &goal, 
  double threshold_distance=0.005) 
{
  double dx = pose.position.x - goal.position.x;
  double dy = pose.position.y - goal.position.y;
  double dz = pose.position.z - goal.position.z;
  return std::sqrt(dx*dx + dy*dy + dz*dz) < threshold_distance;
}

geometry_msgs::msg::Vector3 quaternionToRPY(const geometry_msgs::msg::Quaternion& q_msg) {
    tf2::Quaternion q(q_msg.x, q_msg.y, q_msg.z, q_msg.w);
    tf2::Matrix3x3 m(q);

    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    geometry_msgs::msg::Vector3 rpy;
    rpy.x = roll;
    rpy.y = pitch;
    rpy.z = yaw;
    return rpy;
}

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

std::string base_link;

struct SmoothedTwist {
  geometry_msgs::msg::Twist prev_cmd{};
  double alpha = 0.2;  // [0..1], higher = snappier, lower = smoother

  geometry_msgs::msg::Twist filter(const geometry_msgs::msg::Twist& input) {
    geometry_msgs::msg::Twist output;

    output.linear.x  = alpha * input.linear.x  + (1 - alpha) * prev_cmd.linear.x;
    output.linear.y  = alpha * input.linear.y  + (1 - alpha) * prev_cmd.linear.y;
    output.linear.z  = alpha * input.linear.z  + (1 - alpha) * prev_cmd.linear.z;
    output.angular.x = alpha * input.angular.x + (1 - alpha) * prev_cmd.angular.x;
    output.angular.y = alpha * input.angular.y + (1 - alpha) * prev_cmd.angular.y;
    output.angular.z = alpha * input.angular.z + (1 - alpha) * prev_cmd.angular.z;

    prev_cmd = output;
    return output;
  }
};

void move_with_servo(
  const std::vector<geometry_msgs::msg::Pose> &target_poses,
  moveit::planning_interface::MoveGroupInterface &move_group,
  rclcpp::Node::SharedPtr node)
{
  auto twist_pub = node->create_publisher<geometry_msgs::msg::TwistStamped>("/servo_node/delta_twist_cmds", 10);
  rclcpp::Rate rate(100); // Hz
  const double max_linear_speed = 0.025;  // m/s
  const double max_angular_speed = 0.3; // rad/s
  const double linear_thresh = 0.1;
  const double angular_thresh = 0.5;
  SmoothedTwist smoother;
  int pose_idx = 0;
  for (const auto &target : target_poses) {
    while (rclcpp::ok()) {
      geometry_msgs::msg::Pose current = move_group.getCurrentPose().pose;

      double dx = target.position.x - current.position.x;
      double dy = target.position.y - current.position.y;
      double dz = target.position.z - current.position.z;
      double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

      tf2::Quaternion q1, q2;
      tf2::fromMsg(current.orientation, q1);
      tf2::fromMsg(target.orientation, q2);
      if (q1.dot(q2) < 0.0) {
          q2 = tf2::Quaternion(-q2.x(), -q2.y(), -q2.z(), -q2.w());
      }
      double angle = q1.angleShortestPath(q2);

      // Check if close enough
      if (dist < linear_thresh && angle < angular_thresh) {
        RCLCPP_INFO(node->get_logger(), "Reached waypoint %d", pose_idx);
        pose_idx++;
        break;
      }

      geometry_msgs::msg::TwistStamped cmd;
      
      // Linear velocity
      if (dist > 1e-4) {
        cmd.twist.linear.x = dx / dist * max_linear_speed;
        cmd.twist.linear.y = dy / dist * max_linear_speed;
        cmd.twist.linear.z = dz / dist * max_linear_speed;
      }
      
      // Angular velocity
      tf2::Quaternion q_delta = q2 * q1.inverse();
      q_delta.normalize();
      tf2::Vector3 axis = q_delta.getAxis();
      
      if (angle > 1e-3) {
        // tf2::Vector3 omega = axis.normalized() * std::min(angle / 0.01, max_angular_speed);
        tf2::Vector3 omega = axis.normalized() * max_angular_speed;
        cmd.twist.angular.x = omega.x();
        cmd.twist.angular.y = omega.y();
        cmd.twist.angular.z = omega.z();
      }
      RCLCPP_INFO(node->get_logger(), "Angular cmd: %.2f %.2f %.2f", cmd.twist.angular.x, cmd.twist.angular.y, cmd.twist.angular.z);
      // Low pass filter in case..
      // geometry_msgs::msg::Twist smoothed = smoother.filter(cmd.twist);
      // geometry_msgs::msg::TwistStamped
      cmd.twist = smoother.filter(cmd.twist);
      cmd.header.stamp = node->get_clock()->now();
      cmd.header.frame_id = base_link; // Match your servo config
      twist_pub->publish(cmd);
      rate.sleep();
    }
  }

  // Stop at the end
  geometry_msgs::msg::TwistStamped stop;
  stop.header.stamp = node->get_clock()->now();
  stop.header.frame_id = base_link;
  twist_pub->publish(stop);
}


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  std::string poses_filename = std::filesystem::path(ament_index_cpp::get_package_share_directory("path_projection")) / "output" / "poses.json";
  std::vector<geometry_msgs::msg::Pose> poses = poses_from_json(poses_filename); // load or generate them
  auto const logger = rclcpp::get_logger("servo_logger");
  rclcpp::NodeOptions options;
  options.parameter_overrides({rclcpp::Parameter("use_sim_time", false)});
  auto node = std::make_shared<rclcpp::Node>("servo_controller", options);

  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr logging_trigger_pub_;
  logging_trigger_pub_ = node->create_publisher<std_msgs::msg::Bool>("/start_logging", 10);
  bool use_sim_time = node->get_parameter("use_sim_time").as_bool();
  RCLCPP_INFO(logger, "use_sim_time is set to: %s", use_sim_time ? "true" : "false");
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() { executor.spin(); });

  auto group_name = node->declare_parameter<std::string>("move_group", "");
  if (group_name.empty()) {
  throw std::runtime_error("Missing required parameter: 'move_group'");
  }
  auto move_group = moveit::planning_interface::MoveGroupInterface(node, group_name);

  base_link = move_group.getPlanningFrame();

  rclcpp::sleep_for(std::chrono::milliseconds(2000));
  move_group.setPlanningPipelineId("ompl");
  move_group.setPlannerId("RRTConnectkConfigDefault");  
  move_group.setPlanningTime(15.0);
  move_group.setMaxVelocityScalingFactor(0.8);
  move_group.setMaxAccelerationScalingFactor(0.8);

  std::vector<double> seed_state = get_viable_seed_state(node, poses, group_name, 100, 100, 10);
  if (seed_state.empty()) {
      RCLCPP_WARN(logger,"failed generating a viable seed state");
      rclcpp::shutdown();
      return -1;
  }

  move_group.setJointValueTarget(seed_state);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool success = (move_group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  RCLCPP_INFO(logger, "successfully planned a plan to get to the seed state");

  if (success) {
      move_group.execute(plan);
  } else {
      RCLCPP_ERROR(logger, "Planning to seed state failed");
      rclcpp::shutdown();
      return -1;
  }


  auto test = move_group.getEndEffectorLink();
  RCLCPP_INFO(logger, "This is an INFO message!: %s", test.c_str());
  rclcpp::sleep_for(std::chrono::milliseconds(100));
  while (!move_group.getCurrentState(1.0)) {
    RCLCPP_WARN(logger, "Waiting for current robot state...");
    rclcpp::sleep_for(std::chrono::milliseconds(500));
  }
  geometry_msgs::msg::Pose current = move_group.getCurrentPose().pose;
  RCLCPP_INFO(logger, "Got the pose over here just fine");
  
  // Tell the logging node to start logging joint_states
  auto msg = std_msgs::msg::Bool();
  msg.data = true;
  logging_trigger_pub_->publish(msg);
  rclcpp::sleep_for(std::chrono::milliseconds(200));  

  move_group.setMaxVelocityScalingFactor(0.1);
  move_group.setMaxAccelerationScalingFactor(0.1);
  move_with_servo(poses, move_group, node);

  // tell the logging node to stop logging
  msg.data = false;
  logging_trigger_pub_->publish(msg);
  rclcpp::sleep_for(std::chrono::milliseconds(200));  

  rclcpp::shutdown();
  spinner.join();
  return 0;
}
