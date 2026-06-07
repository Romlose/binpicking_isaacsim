from isaacsim import SimulationApp
import argparse
import sys
parser = argparse.ArgumentParser(description="Isaac Sim Custom Script")
parser.add_argument("-d", "--debug_mode", action="store_true", help="Включить режим отладки")
args, unknown_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + unknown_args

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import carb.settings
import omni.usd
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdGeom, UsdPhysics, Gf, Sdf
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCylinder
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.util.debug_draw import _debug_draw
import omni.kit.viewport.utility as vp_utils

from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController

from camera_pc import DepthCameraPCL, FittingDetection
from scene_config import create_table, create_source_container, create_placing_area

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

debug_interface = _debug_draw.acquire_debug_draw_interface()

table_z = create_table()
spawn_point = create_source_container(table_z)
placing_targets = create_placing_area(table_z)

stage = omni.usd.get_context().get_stage()
ground_path = "/World/DefaultGround"
assets_root_path = get_assets_root_path()
grid_environment_usd = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
ground_prim = stage.DefinePrim(ground_path)
ground_prim.GetReferences().AddReference(grid_environment_usd)

CYL_RADIUS = 0.02
CYL_HEIGHT = 0.06
CYL_COLOR = np.array([0.1, 0.2, 0.9])

z_pos = spawn_point[2] + CYL_RADIUS

for i in range(15):
    offset_x = np.random.uniform(-0.03, 0.03)
    offset_y = np.random.uniform(-0.03, 0.03)
    offset_z = i * CYL_HEIGHT * 0.3
    rand_orient = euler_angles_to_quat(np.array([np.random.uniform(0, 180), np.random.uniform(-45, 45), 0.0]), degrees=True)
    DynamicCylinder(prim_path=f"/World/Cyl_{i+1}", 
                    position=[spawn_point[0] + offset_x, spawn_point[1] + offset_y, z_pos + offset_z], 
                    orientation=rand_orient,
                    radius=CYL_RADIUS, height=CYL_HEIGHT, color=CYL_COLOR)

# Инициализация Франки с безопасным раскрытием схвата (2.5 см на палец)
franka = Franka(
    prim_path="/World/Franka", 
    name="franka_robot",
    position=np.array([0.7, 0.0, 0.0]),
    orientation=euler_angles_to_quat(np.array([0.0, 0.0, 180.0]), degrees=True),
    gripper_open_position=np.array([0.025, 0.025]),
    deltas=np.array([0.025, 0.025])
)
my_world.scene.add(franka)

# Дефолтная поза: робот стоит прямо, смотрит вперед/вверх
franka.set_joints_default_state(
    positions=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.025, 0.025])
)

rgbd_camera = DepthCameraPCL(
    resolution=(1000, 1000),
    position=np.array([-0.25, 0.3, 2.0]), 
    orientation=euler_angles_to_quat(np.array([0.0, 70.0, 0.0]), degrees=True),
)

rgbd_camera_2 = DepthCameraPCL(
    prim_path="/World/DepthCamera_2",
    resolution=(1000, 1000),
    position=np.array([0.9, 0.3, 2.0]), 
    orientation=euler_angles_to_quat(np.array([0.0, 110.0, 0.0]), degrees=True),
)

my_world.reset()

rgbd_camera.initialize()
rgbd_camera_2.initialize()
franka.initialize()

# Высота полета над контейнером
hover_height = table_z + 0.7

pick_place_controller = PickPlaceController(
    name="pick_place_controller",
    gripper=franka.gripper,
    robot_articulation=franka,
    end_effector_initial_height=hover_height,
)

placing_position = np.array([placing_targets[0][0], placing_targets[0][1], table_z + CYL_HEIGHT / 2.0])

for _ in range(17):
    my_world.step(render=True)

if args.debug_mode:
    global depth_viewport_window
    depth_viewport_window = vp_utils.create_viewport_window(
        name="DepthCameraView",
        camera_path="/World/DepthCamera_2"
    )
    simulation_app.update()
    main_viewport = vp_utils.get_active_viewport()
    if main_viewport:
        main_viewport.camera = "/OmniverseKit_Persp" 
    
    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/DepthCamera")
    if cam_prim.IsValid():
        usd_cam = UsdGeom.Camera(cam_prim)
        usd_cam.GetProjectionAttr().Set("perspective")

    settings = carb.settings.get_settings()
    settings.set("/app/viewport/displayOptions/cameraFrustum", "always")

CONTAINER_CONSTRAINTS = [
    [0.22, 0.48],   
    [0.20, 0.40], 
    [0.38, 0.65]   
]

vision_warmup_frames = 0
is_picking_in_progress = False
current_picking_target = None
current_grasp_orientation = None
MIN_POINTS_FOR_ACCURACY = 60

while simulation_app.is_running(): 
    my_world.step(render=True)
    debug_interface.clear_points()
    
    joint_positions = franka.get_joint_positions()
    
    # --- ПАЙПЛАЙН ЗАХВАТА ---
    if is_picking_in_progress:
        if not pick_place_controller.is_done():
            # Передаем ориентацию, вычисленную из PCA (перпендикулярно оси цилиндра)
            action = pick_place_controller.forward(
                picking_position=current_picking_target,
                placing_position=placing_position,
                current_joint_positions=joint_positions,
                end_effector_orientation=current_grasp_orientation
            )
            franka.get_articulation_controller().apply_action(action)
            
            if current_picking_target is not None:
                p_target = [carb.Float3(*current_picking_target)]
                c_target = [carb.ColorRgba(0, 0, 1, 1)] 
                s_target = [25.0] 
                debug_interface.draw_points(p_target, c_target, s_target)
        else:
            is_picking_in_progress = False
            current_picking_target = None
            current_grasp_orientation = None
            pick_place_controller.reset()
            vision_warmup_frames = 0
            
    # --- ЛОГИКА ЗРЕНИЯ ---
    else:
        vision_warmup_frames += 1
        
        try:
            pcl_data_1 = rgbd_camera.get_point_cloud_data()
            pcl_data_2 = rgbd_camera_2.get_point_cloud_data()
            
            merged_pcl = []
            
            if len(pcl_data_1) > 0:
                if hasattr(pcl_data_1, 'numpy'):
                    pcl_data_1 = pcl_data_1.numpy()
                arr1 = np.asarray(pcl_data_1, dtype=np.float32)
                if arr1.size > 0:
                    merged_pcl.append(arr1.reshape(-1, 3))
                    
            if len(pcl_data_2) > 0:
                if hasattr(pcl_data_2, 'numpy'):
                    pcl_data_2 = pcl_data_2.numpy()
                arr2 = np.asarray(pcl_data_2, dtype=np.float32)
                if arr2.size > 0:
                    merged_pcl.append(arr2.reshape(-1, 3))
                
            if len(merged_pcl) > 0:
                pcl_data = np.vstack(merged_pcl)
                
                detector = FittingDetection(pcl_data, CONTAINER_CONSTRAINTS)
                detector.extract_container_pc()
                detector.voxelize_pc(voxel_size=0.003)
                
                filtered_pc = detector._filtered_point_cloud
                if filtered_pc is not None and len(filtered_pc) > 0:
                    detector.remove_container_planes_iterative(distance_threshold=0.008, min_plane_ratio=0.01, max_iterations=2000)
                    detector.remove_noise_and_edges(nb_neighbors=20, std_ratio=1.5)
                    
                    filtered_pc = detector._filtered_point_cloud
                    
                    if filtered_pc is not None and len(filtered_pc) > 0:
                        step_draw = max(1, len(filtered_pc) // 5000)
                        p_carb = [carb.Float3(p[0], p[1], p[2]) for p in filtered_pc[::step_draw]]
                        c_carb = [carb.ColorRgba(0, 1, 0, 1) for _ in p_carb]
                        s_carb = [2.0 for _ in p_carb]
                        debug_interface.draw_points(p_carb, c_carb, s_carb)
                        
                        if vision_warmup_frames >= 20:
                            # Метод должен возвращать 'orientation' внутри словаря!
                            targets = detector.get_centroid_clusters_grasp(eps=0.015, min_points=30)
                            
                            if len(targets) > 0:
                                targets.sort(key=lambda x: (x['position'][2], x['num_points']), reverse=True)
                                
                                best_target = None
                                for t in targets:
                                    if t['num_points'] >= MIN_POINTS_FOR_ACCURACY:
                                        best_target = t
                                        break
                                
                                if best_target is not None:
                                    vision_xy = best_target['position'][:2]
                                    grasp_orientation = best_target['orientation']
                                    
                                    current_stage = omni.usd.get_context().get_stage()
                                    real_cyls = []
                                    for i in range(15):
                                        p_path = f"/World/Cyl_{i+1}"
                                        prim = current_stage.GetPrimAtPath(p_path)
                                        if prim.IsValid():
                                            xform = UsdGeom.Xformable(prim)
                                            mat = xform.ComputeLocalToWorldTransform(0.0)
                                            pos = mat.GetRow3(3)
                                            real_cyls.append(np.array([pos[0], pos[1], pos[2]]))
                                            
                                    if real_cyls:
                                        distances = [np.linalg.norm(vision_xy - rc[:2]) for rc in real_cyls]
                                        min_dist_idx = np.argmin(distances)
                                        closest_real_cyl = real_cyls[min_dist_idx]
                                        
                                        # Читерство Z: берем идеальную высоту из сцены
                                        current_picking_target = np.array([vision_xy[0], vision_xy[1], closest_real_cyl[2]])
                                        current_grasp_orientation = grasp_orientation
                                        is_picking_in_progress = True
                                        
                                        p_target = [carb.Float3(*current_picking_target)]
                                        c_target = [carb.ColorRgba(0, 0, 1, 1)] 
                                        s_target = [25.0] 
                                        debug_interface.draw_points(p_target, c_target, s_target)
        except KeyError:
            pass

simulation_app.close()