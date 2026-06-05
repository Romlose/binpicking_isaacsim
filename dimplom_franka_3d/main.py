from isaacsim import SimulationApp
import argparse
import sys
parser = argparse.ArgumentParser(description="Isaac Sim Custom Script")
parser.add_argument("-d", "--debug_mode", action="store_true", help="Включить режим отладки")
args, unknown_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + unknown_args

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})


# Основной код
import numpy as np

import carb.settings
import omni.usd
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdGeom, UsdPhysics, Gf, Sdf
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCylinder
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from isaacsim.util.debug_draw import _debug_draw
import omni.kit.viewport.utility as vp_utils

from behavior_scripts import CylinderBehavior, FrankaController
from camera_pc import DepthCameraPCL, FittingDetection
from scene_config import create_table, create_source_container, create_placing_area

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# Получаем интерфейс для дебажнной отрисовки
debug_interface = _debug_draw.acquire_debug_draw_interface()

# Создаем сцену
table_z = create_table()
spawn_point = create_source_container(table_z)
placing_targets = create_placing_area(table_z)



stage = omni.usd.get_context().get_stage()
ground_path = "/World/DefaultGround"

# Получаем путь к серверу ассетов Isaac Sim
assets_root_path = get_assets_root_path()
# Путь к оригинальной синей сетке со встроенной физикой
grid_environment_usd = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"

# Подгружаем готовую сцену-сетку как референс в ваш мир
ground_prim = stage.DefinePrim(ground_path)
ground_prim.GetReferences().AddReference(grid_environment_usd)

print("Default Isaac Sim grid environment added.")


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

# Создаем камеру
rgbd_camera = DepthCameraPCL(
    resolution=(1000, 1000),
    position=np.array([-0.25, 0.3, 2.0]), 
    orientation=euler_angles_to_quat(np.array([0.0, 70.0, 0.0]), degrees=True),
)

# 2
rgbd_camera_2 = DepthCameraPCL(
    prim_path="/World/DepthCamera_2",
    resolution=(1000, 1000),
    position=np.array([0.9, 0.3, 2.0]), 
    orientation=euler_angles_to_quat(np.array([0.0, 110.0, 0.0]), degrees=True),
)

my_world.reset()

# После world reset судя по документации метода
rgbd_camera.initialize()
rgbd_camera_2.initialize()

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

step = 0
sensor_initialized = False
sensor_warned = False
loop_counter = 0

CYL_HEIGHT = 0.06
Z_SLICE_MULTIPLIER = 1.5
loop_counter = 0

while simulation_app.is_running(): 
    my_world.step(render=True)
    debug_interface.clear_points()
    loop_counter += 1
    
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
                    step = max(1, len(filtered_pc) // 5000)
                    p_carb = [carb.Float3(p[0], p[1], p[2]) for p in filtered_pc[::step]]
                    c_carb = [carb.ColorRgba(0, 1, 0, 1) for _ in p_carb]
                    s_carb = [2.0 for _ in p_carb]
                    debug_interface.draw_points(p_carb, c_carb, s_carb)
                    
                    targets = detector.get_centroid_clusters_grasp(eps=0.015, min_points=30)
                    
                    if len(targets) > 0:
                        p_center = [carb.Float3(t['position'][0], t['position'][1], t['position'][2]) for t in targets]
                        c_center = [carb.ColorRgba(1, 0, 0, 1) for _ in targets]
                        s_center = [15.0 for _ in targets]
                        debug_interface.draw_points(p_center, c_center, s_center)
    except KeyError:
        pass

simulation_app.close()