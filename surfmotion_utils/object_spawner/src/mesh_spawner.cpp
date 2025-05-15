#include <memory>
#include <rclcpp/logger.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include "collision_object_spawner.hpp"

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto const node = std::make_shared<rclcpp::Node>("cylinder_spawner");
    auto const logger = rclcpp::get_logger("cylinder_spawner");

    node->declare_parameter<std::string>("mesh_file", "");
    std::string mesh_file;
    if (!node->get_parameter("mesh_file", mesh_file) || mesh_file.empty()) {
        RCLCPP_ERROR(node->get_logger(), "Mesh file path not provided!");
        return -1;
    }

    if (mesh_file.find("file://") != 0) {
        mesh_file = "file://" + mesh_file;
    }
    
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    auto spinner = std::thread([&executor]() { executor.spin(); });
    
    CollisionObjectSpawner spawner("world");
    geometry_msgs::msg::Pose cylinder_pose;
    cylinder_pose.position.x = 1.0;
    cylinder_pose.position.y = 0.2;
    cylinder_pose.position.z = 0.56;

    tf2::Quaternion q;
    q.setRPY(1.43, 1.57, -1.35);
    cylinder_pose.orientation.x = q.x();
    cylinder_pose.orientation.y = q.y();
    cylinder_pose.orientation.x = q.z();
    cylinder_pose.orientation.w = q.w();

    rosidl_runtime_cpp::BoundedVector<double, 3> dimensions = {0.80, 0.45, 0.80};
    // std::string mesh_file = "package://reach_config/config/cylinder_lower_away.ply";

    if (spawner.spawnMeshObj("cylinder_0", mesh_file, dimensions, cylinder_pose) == 0) {
        RCLCPP_INFO(node->get_logger(), "Successfully spawned the cylinder!");
    } else {
        RCLCPP_ERROR(node->get_logger(), "Failed to spawn the cylinder!");
    }

    rclcpp::shutdown();
    spinner.join();
    return 0;
}