#include <shape_msgs/msg/solid_primitive.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <string>
#include <geometric_shapes/shape_operations.h>
#include <shape_msgs/shape_msgs/msg/mesh.hpp>
#include <shape_msgs/msg/mesh.hpp>
#include <geometric_shapes/shape_messages.h>
#include <geometric_shapes/shapes.h>
#include <geometric_shapes/mesh_operations.h>

class CollisionObjectSpawner{
public:
    explicit CollisionObjectSpawner(const std::string &frame_id);

    int spawnPrimitiveObj(
        const std::string &id,
        const shape_msgs::msg::SolidPrimitive::_type_type &primitive_type,
        const rosidl_runtime_cpp::BoundedVector<double, 3> &dimensions,
        const geometry_msgs::msg::Pose &pose
    );

    int spawnMeshObj(
        const std::string &id,
        const std::string &file_name,
        const rosidl_runtime_cpp::BoundedVector<double, 3> &dimensions,
        const geometry_msgs::msg::Pose &pose
    );
    
private:
    std::string frame_id_;
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
};