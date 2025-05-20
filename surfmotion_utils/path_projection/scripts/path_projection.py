import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import threading
import time
import os
import subprocess
import json
from enum import Enum
from scipy.spatial import KDTree
from ament_index_python.packages import get_package_share_directory
from transforms3d.quaternions import mat2quat, qmult, axangle2quat

package_name = "path_projection"
package_path = get_package_share_directory(package_name)
output_dir = os.path.join(package_path, "output")
pointcloud_path = os.path.join(output_dir, "pointcloud.pcd")
mesh_path = os.path.join(package_path, "meshes", "cylinder_lower_away.ply")


def normal_and_twist_to_quaternion(normal, position, centroid, tool_rotation=90):
    # Z-axis from surface normal
    z = -normal / np.linalg.norm(normal)
    
    radial = position - centroid
    radial_proj = radial - np.dot(radial, z) * z
    
    if np.linalg.norm(radial_proj) < 1e-6:
        # fallback if radial vector is somehow aligned with Z
        x = np.array([1, 0, 0])
    else:
        x = radial_proj / np.linalg.norm(radial_proj)

    y = np.cross(z, x)
    
    R = np.column_stack((x, y, z))

    q = mat2quat(R)

    tool_rotation_rad = np.deg2rad(tool_rotation)
    q_rot = axangle2quat([0, 0, 1], tool_rotation_rad)
    q_final = qmult(q, q_rot)
    return q_final  # [x, y, z, w]

def normal_to_quaternion(normal):
    x_axis = np.array([0, 0, 1])
    axis = np.cross(x_axis, normal)
    angle = np.arccos(np.dot(x_axis, normal) / (np.linalg.norm(x_axis) * np.linalg.norm(normal)))

    if np.linalg.norm(axis) < 1e-6:
        R = np.eye(3)
    else:
        axis = axis / np.linalg.norm(axis)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    return mat2quat(R)

def save_poses(pcd: o3d.geometry.PointCloud):
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    centroid = np.mean(points, axis=0)
    
    poses = []
    
    for i in range(len(points)):
        position = {
            "x": float(points[i][0]),
            "y": float(points[i][1]),
            "z": float(points[i][2])
        }
        quaternion = normal_to_quaternion(normals[i])
        np_pos = np.array([position["x"], position["y"], position["z"]])
        quaternion = normal_and_twist_to_quaternion(normals[i], np_pos, centroid, tool_rotation=180)
        orientation = {
            "x": quaternion[1],
            "y": quaternion[2],
            "z": quaternion[3],
            "w": quaternion[0]
        }
        poses.append({"position": position, "orientation": orientation})
    json_path = os.path.join(output_dir, "poses.json")
    with open(json_path, "w") as f:
        json.dump(poses, f, indent=4)

    print(f"Saved the path successfully at: {json_path}")
        

def save_trajectory(pcd: o3d.geometry.PointCloud):
    output_path = os.path.join(output_dir, "test_output.pcd")
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)

def run_reach_study():
    # ros2_launch_exe = FindExecutable(name="ros2")

    try:
        subprocess.run(["ros2",
                        "launch",
                        package_name,
                        "reach_analysis2.launch.py"
                        ])
    except:
        print(f"Error launch reach_analysis")

def create_circle(radius=0.1, segments=64, z=0.0):
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    points = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full_like(angles, z)], axis=-1)
    lines = [[i, (i + 1) % segments] for i in range(segments)]
    circle = o3d.geometry.LineSet()
    circle.points = o3d.utility.Vector3dVector(points)
    circle.lines = o3d.utility.Vector2iVector(lines)
    return circle

def generate_circle(radius, z_offset, y_offset, x_plane=0, num_points=50):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = np.full(num_points, x_plane)
    y = radius * np.cos(theta) + y_offset
    z = radius * np.sin(theta) + z_offset
    points = np.vstack((x, y, z)).T
    return points

class TrajectoryType(Enum):
    CIRCLE = 0
    OTHER = 0

class App:
    def __init__(self, mesh_path):
        # self.pointcloud_full = o3d.io.read_point_cloud(pointcloud_path)
        # print(type(self.pointcloud_full))
        self.auto_orient = True
        self.is_done = False
        self.visible = None
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.mesh.compute_vertex_normals()
        self.pcd = self.mesh.sample_points_uniformly(number_of_points=2000000)
        # move points 1cm outward)
        self.pcd.points = o3d.utility.Vector3dVector(np.asarray(self.pcd.points) + 0.02 * np.asarray(self.pcd.normals))  
        self.pcd_downsampled = self.pcd.voxel_down_sample(voxel_size=0.01)
        self.pcd_very_downsampled = self.pcd.voxel_down_sample(voxel_size=0.02)
        save_trajectory(self.pcd_downsampled)
        # o3d.io.write_point_cloud("whatsthis.pcd", self.pcd, write_ascii=True)
        self.visible_pcd = o3d.geometry.PointCloud(self.pcd_downsampled)
        self.visible_hd = None


        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("Trajectory casting", 1024, 768)        
        self.window.set_on_close(self.on_main_window_closing)

        # basic setup
        em = self.window.theme.font_size
        self.layout = gui.Horiz(0, gui.Margins(0.5*em, 0.5*em, 0.5*em, 0.5*em))
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0, 0, 0, 1])

        # mat
        self.mat_white = rendering.MaterialRecord()
        self.mat_white.shader = "defaultUnlit"
        self.mat_white.point_size = 3
        self.mat_white.base_color= ([1, 1, 1, 1])
        # self.mat.base_color = ([1, 0, 0, 1])

        self.mat_blue = rendering.MaterialRecord()
        self.mat_blue.shader = "defaultUnlit"
        self.mat_blue.point_size = 5
        self.mat_blue.base_color = ([0, 0, 1, 1])
        
        self.mat_red = rendering.MaterialRecord()
        self.mat_red.shader = "defaultUnlit"
        self.mat_red.point_size = 8
        self.mat_red.base_color = ([1, 0, 0, 1])

        self.mat_green = rendering.MaterialRecord()
        self.mat_green.shader = "defaultLit"
        self.mat_green.point_size = 5
        self.mat_green.base_color = ([0, 1, 0, 1])

        self.mat_selection = rendering.MaterialRecord()
        self.mat_selection.shader = "defaultLitTransparency"
        self.mat_selection.base_color = ([1, 0, 0, 0.5])

        params = {
            "radius": 0.2,
            "z_offset": 0.0,
            "y_offset": 0.0
        }

        self.circle = o3d.geometry.PointCloud()
        self.circle.points = o3d.utility.Vector3dVector(generate_circle(**params))
        self.circle.paint_uniform_color([1, 0, 0])
        self.circle_transform = np.eye(4)

        self.trajectory_type = TrajectoryType.CIRCLE

        # add point cloud and circle
        self.scene.scene.add_geometry("pcd", self.pcd_downsampled, self.mat_white)
        bounds = self.pcd.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60, bounds, bounds.get_center())
        # self.circle = create_circle(radius=0.2, z=1.0)
        self.scene.scene.add_geometry("circle", self.circle, self.mat_red)
        
        self.proj_circle = o3d.geometry.PointCloud()
        self.proj_circle.points = o3d.utility.Vector3dVector(generate_circle(**params))
        self.proj_circle.paint_uniform_color([1, 1, 0])
        self.scene.scene.add_geometry("projection", self.proj_circle, self.mat_red)
        
        pole = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01, height=1.0)
        pole.compute_vertex_normals()
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.compute_vertex_normals()
        self.scene.scene.add_geometry("pole", pole, self.mat_white)
        self.scene.scene.add_geometry("sphere", sphere, self.mat_red)
        self.scene.scene.add_geometry("visible", self.visible_pcd, self.mat_blue)
        
        self.arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.005,
            cone_radius=0.01,
            cylinder_height=0.2,
            cone_height=0.03
        )
        self.arrow.compute_vertex_normals()
        self.arrow.paint_uniform_color([0, 1, 0])
        self.scene.scene.add_geometry("arrow", self.arrow, self.mat_white)

        self.normal_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius = 0.005,
            cone_radius=0.01,
            cylinder_height=0.05,
            cone_height=0.03
        )
        self.normal_arrow.compute_vertex_normals()
        self.normal_arrow.paint_uniform_color([0, 1, 0])

        for i in range(50):
            self.scene.scene.add_geometry(f"normal_arrow{i}", self.normal_arrow, self.mat_green)

        self.box_transform = np.eye(4)
        self.selection_box = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.1)
        self.selection_box.compute_vertex_normals()
        self.selection_box.translate([-0.025, -0.025, -0.025])
        self.selection_box.paint_uniform_color([0.8, 0.3, 0.0])
        self.scene.scene.add_geometry("selection_box", self.selection_box, self.mat_selection)
        self.scene.scene.add_geometry("selected_points", self.selection_box, self.mat_selection)

        self.panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
        self.circle_sliders = [
            self.make_circle_slider("X", 0), 
            self.make_circle_slider("Y", 1), 
            self.make_circle_slider("Z", 2)
        ]

        self.box_sliders = [
            self.make_box_slider("Box X", 0),
            self.make_box_slider("Box Y", 1),
            self.make_box_slider("Box Z", 2),
        ]

        self.buttons = [
            self.make_button("Run high-def analysis", self.on_high_def_analysis),
            self.make_button("Run Reach study", self.on_run_reach_study),
            self.make_button("Toggle cutting/welding mode", self.on_toggle_mode),
            self.make_button("Save path", self.on_save_path)
        ]

        self.welding_mode = False

        self.window.add_child(self.scene)
        self.window.add_child(self.panel)

        def on_layout(ctx):
            content_rect = self.window.content_rect
            panel_width = 200
            self.panel.frame = gui.Rect(content_rect.get_right() - panel_width, content_rect.y,
                                        panel_width, content_rect.height)
            self.scene.frame = gui.Rect(content_rect.x, content_rect.y,
                                        content_rect.width - panel_width, content_rect.height)

        self.window.set_on_layout(on_layout)

    def run(self):
        self.app.run()
    
    def on_main_window_closing(self):
        self.is_done = True
        return True

    def make_button(self, label, callback):
        button = gui.Button(label)
        button.set_on_clicked(callback)
        self.panel.add_child(button)
        return button
    
    def on_save_path(self):
        save_poses(self.proj_circle)


    def on_high_def_analysis(self):
        self.slow_update_fov(self.pcd)
        self.project_on_surface(self.visible_hd)
        if self.welding_mode:
            self.update_welding_normals(self.proj_circle)

    def on_run_reach_study(self):
        save_trajectory(self.proj_circle)
        save_poses(self.proj_circle)
        run_reach_study()

    def on_toggle_mode(self):
        self.welding_mode = True

    def make_circle_slider(self, label, idx):
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(-2.0, 2.0)
        slider.set_on_value_changed(lambda val: self.update_circle_position(idx, val))
        self.panel.add_child(gui.Label(label))
        self.panel.add_child(slider)
        return slider
    
    def make_box_slider(self, label, axis):
        slider = gui.Slider(gui.Slider.DOUBLE)
        slider.set_limits(-1.0, 1.0)
        slider.set_on_value_changed(lambda val: self.update_box_position(axis, val))
        self.panel.add_child(gui.Label(label))
        self.panel.add_child(slider)
        return slider
    
    def update_box_position(self, axis, value):
        self.box_transform[axis, 3] = value
        self.scene.scene.set_geometry_transform("selection_box", self.box_transform)  
        self.get_points_in_selection_box() 

    def get_points_in_selection_box(self):
        # TODO Should be initialized some other way. look closer at documentation
        box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=self.box_transform[:3, 3] - 0.025, # box size 
            max_bound=self.box_transform[:3, 3] + 0.075
        )
        points = self.pcd_downsampled.crop(box)
        # if len(points.points) > 0: 
        #     points = self.detect_normal_jumps(points)
        
        self.scene.scene.remove_geometry("selected_points")
        self.scene.scene.add_geometry("selected_points", points, self.mat_red)

    def update_circle_position(self, axis, value):
        self.circle_transform[axis, 3] = value
        self.scene.scene.set_geometry_transform("circle", self.circle_transform)

        R = o3d.geometry.get_rotation_matrix_from_xyz([0, -np.pi/2, 0])  # rotates Z to X
        arrow_transform = self.circle_transform.copy()
        arrow_transform[:3, :3] = self.circle_transform[:3, :3] @ R
        self.scene.scene.set_geometry_transform("arrow", arrow_transform)
        self.fast_update_fov()
        if self.auto_orient == True:
            circle_pos = self.circle_transform[:3, 3]
            self.auto_orientation(circle_pos=circle_pos)
        self.project_on_surface(self.visible)
        
    def fast_update_fov(self):
        # fast visibility check if only part of the circle projection is on top of the object, the fov should "wrap around" the surface
        view_matrix = self.circle_transform
        pos = view_matrix[:3, 3]
        radius = 2
        _, pt_map = self.pcd_very_downsampled.hidden_point_removal(pos, radius)
        self.visible = self.pcd_very_downsampled.select_by_index(pt_map)

        self.scene.scene.remove_geometry("visible")
        self.scene.scene.add_geometry("visible", self.visible, self.mat_blue)
        self.update_timer = 0

    def slow_update_fov(self, pcd: o3d.geometry.PointCloud):
        view_matrix = self.circle_transform
        pos = view_matrix[:3, 3]
        radius = 2
        _, pt_map = pcd.hidden_point_removal(pos, radius)
        self.visible_hd = pcd.select_by_index(pt_map)

    def auto_orientation(self, circle_pos):
        pcd_tree = o3d.geometry.KDTreeFlann(self.pcd_downsampled)
        # _, idxs, _ = pcd_tree.search_knn_vector_3d(circle_pos, 30)
        _, idxs, _ = pcd_tree.search_radius_vector_3d(circle_pos, radius=0.5)
        normals = np.asarray(self.pcd_downsampled.normals)[idxs]
        avg_normal = np.mean(normals, axis=0)
        avg_normal /= np.linalg.norm(avg_normal)

        x_axis = np.array([1, 0, 0])
        axis = np.cross(x_axis, avg_normal)
        angle = np.arccos(np.dot(x_axis, avg_normal) / (np.linalg.norm(x_axis) * np.linalg.norm(avg_normal)))

        if np.linalg.norm(axis) < 1e-6:
            R = np.eye(3)
        else:
            axis = axis / np.linalg.norm(axis)
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = circle_pos
        self.circle_transform = T

    def project_on_surface(self, pcd: o3d.geometry.PointCloud):
        # get circle rotation and position
        R_circ = self.circle_transform[:3, :3]
        origin = self.circle_transform[:3, 3]

        # Get visible points and move them into the circle's local frame
        pcd_np = np.asarray(pcd.points)
        rel_points = pcd_np - origin
        pcd_in_circle_frame = (R_circ.T @ rel_points.T).T

        # Project into circle's x, y plane
        pcl_2d = pcd_in_circle_frame[:, 1:3]
        circle_local = np.asarray(self.circle.points)
        circle_2d = circle_local[:, 1:3]

        tree = KDTree(pcl_2d)
        dists, idx = tree.query(circle_2d, distance_upper_bound=0.02)
        
        valid_mask = np.isfinite(dists)
        valid_idx = idx[valid_mask]

        proj_points = np.asarray(pcd.points)[valid_idx]
        proj_normals = np.asarray(pcd.normals)[valid_idx]

        proj_pcd = o3d.geometry.PointCloud()
        proj_pcd.points = o3d.utility.Vector3dVector(proj_points)
        proj_pcd.normals = o3d.utility.Vector3dVector(proj_normals)

        # Visualization
        # self.proj_circle = self.visible.select_by_index(idx)
        self.proj_circle = proj_pcd

        if self.welding_mode:
            self.update_welding_normals(self.proj_circle)

        self.scene.scene.remove_geometry("projection")
        self.scene.scene.add_geometry("projection", self.proj_circle, self.mat_red)

    def update_welding_normals(self, pcd: o3d.geometry.PointCloud, trajectory_type: TrajectoryType = TrajectoryType.CIRCLE):
        if trajectory_type == TrajectoryType.CIRCLE:
            print("Using a circular trajectory\n Updating normals...")
            pcd.normals = self.calculate_welding_normals_circle(pcd)
            self.draw_normal_arrows(pcd)
        else:
            print("Non-circlular trajectory not supported")

    def calculate_welding_normals_circle(self, pcd: o3d.geometry.PointCloud, tilt_angle=20) -> o3d.geometry.PointCloud.normals:
        normals = np.asarray(pcd.normals)
        points = np.asarray(pcd.points)
        tilt_angle = np.radians(tilt_angle)
        tilted_normals = []
        centroid = np.mean(points, axis=0)
        for i in range(len((normals))):
            point = points[i]
            normal = normals[i]

            centroid_direction = -(centroid - point)
            centroid_direction = centroid_direction / np.linalg.norm(centroid_direction)

            normal_tilted = np.cos(tilt_angle) * normal + np.sin(tilt_angle) * centroid_direction
            normal_tilted = normal_tilted / np.linalg.norm(normal_tilted)

            # check if its pointing outwards
            if np.dot(normal_tilted, centroid_direction) < 0:
                normal_tilted = -normal_tilted

            tilted_normals.append(normal_tilted)

        return o3d.utility.Vector3dVector(np.asarray(tilted_normals))

    def draw_normal_arrows(self, pcd):
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        arrows = []

        for i in range(len(points)):
            point = points[i]
            normal = normals[i]

            arrow_copy = o3d.geometry.TriangleMesh()
            arrow_copy.vertices = self.normal_arrow.vertices
            arrow_copy.triangles = self.normal_arrow.triangles
            arrow_copy.vertex_normals = self.normal_arrow.vertex_normals
            
            normal = normal / np.linalg.norm(normal)
            arrow_copy.translate(point)
            self.align_arrow_to_normal(arrow_copy, normal, point)

            arrows.append(arrow_copy)
            self.scene.scene.remove_geometry(f"normal_arrow{i}")
            self.scene.scene.add_geometry(f"normal_arrow{i}", arrow_copy, self.mat_green)

    def align_arrow_to_normal(self, arrow, normal, point):
        normal = normal / np.linalg.norm(normal)
        up_vector = np.array([0, 0, 1])
        rotation_axis = np.cross(up_vector, normal)
        rotation_angle = np.arccos(np.dot(up_vector, normal))

        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis*rotation_angle)
        arrow.rotate(rotation_matrix, center=point)

def main():
    app = App(mesh_path)
    app.run()

if __name__ == "__main__":
    main()
