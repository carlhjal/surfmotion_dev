#include "collision_object_spawner.hpp"
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>

CollisionObjectSpawner::CollisionObjectSpawner(const std::string &frame_id) : frame_id_(frame_id) {}

int CollisionObjectSpawner::spawnPrimitiveObj(const std::string &id,
    const shape_msgs::msg::SolidPrimitive::_type_type &primitive_type,
    const rosidl_runtime_cpp::BoundedVector<double, 3> &dimensions,
    const geometry_msgs::msg::Pose &pose) {
    
    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.header.frame_id = frame_id_;
    collision_object.id = id;

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive_type;
    primitive.dimensions = dimensions;

    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(pose);
    collision_object.operation = collision_object.ADD;

    planning_scene_interface_.applyCollisionObject(collision_object);
    return 0;
}

int CollisionObjectSpawner::spawnMeshObj(const std::string &id,
        const std::string &file_name,
        const rosidl_runtime_cpp::BoundedVector<double, 3> &dimensions,
        const geometry_msgs::msg::Pose &pose) {
    
    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.header.frame_id = frame_id_;
    collision_object.id = id;

    shapes::Mesh *mesh = shapes::createMeshFromResource(file_name);
    if (!mesh) {
        RCLCPP_ERROR(rclcpp::get_logger("CollisionObjectSpawner"), 
        "Failed to load resource: %s", file_name.c_str());
        return -1;
    }

    shape_msgs::msg::Mesh mesh_msg;
    shapes::ShapeMsg shape_msg;
    shapes::constructMsgFromShape(mesh, shape_msg);
    mesh_msg = boost::get<shape_msgs::msg::Mesh>(shape_msg);

    geometry_msgs::msg::Pose mesh_pose;
    mesh_pose.position.x = 0;
    mesh_pose.position.y = 0;
    mesh_pose.position.z = 0;
    mesh_pose.orientation.x = 0;
    mesh_pose.orientation.y = 0;
    mesh_pose.orientation.z = 0;
    mesh_pose.orientation.w = 1;

    collision_object.meshes.push_back(mesh_msg);
    collision_object.mesh_poses. push_back(mesh_pose);
    collision_object.operation = collision_object.ADD;

    planning_scene_interface_.applyCollisionObject(collision_object);
    return 0;
}