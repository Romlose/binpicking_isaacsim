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


#Пока с одим фитингом
# initial_orientation = euler_angles_to_quat(np.array([0.0, 0.3, 0.0])) 
# 
# cyl_script = CylinderBehavior(
    # prim_path="/World/SourceContainer", 
    # name="fittings_spawner",
    # number_cylinders=10,
    # position=spawn_point.tolist(),
    # orientation=initial_orientation.tolist()
CYL_RADIUS = 0.025
CYL_HEIGHT = 0.08
CYL_COLOR = np.array([0.1, 0.2, 0.9])

lay_flat_orientation = euler_angles_to_quat(np.array([90.0, 0.0, 0.0]), degrees=True)

z_pos = spawn_point[2] + CYL_RADIUS

DynamicCylinder(prim_path="/World/Cyl_1", 
                position=[spawn_point[0], spawn_point[1], z_pos], 
                orientation=lay_flat_orientation,
                radius=CYL_RADIUS, height=CYL_HEIGHT, color=CYL_COLOR)

DynamicCylinder(prim_path="/World/Cyl_2", 
                position=[spawn_point[0] - 0.05, spawn_point[1] + 0.02, z_pos], 
                orientation=lay_flat_orientation,
                radius=CYL_RADIUS, height=CYL_HEIGHT, color=CYL_COLOR)

DynamicCylinder(prim_path="/World/Cyl_3", 
                position=[spawn_point[0] + 0.04, spawn_point[1] - 0.03, z_pos + 0.01], 
                orientation=lay_flat_orientation,
                radius=CYL_RADIUS, height=CYL_HEIGHT, color=CYL_COLOR)

# Создаем камеру
rgbd_camera = DepthCameraPCL(
    resolution=(1000, 1000),
    position=np.array([0.35, 0.3, 2.5]), 
    orientation=euler_angles_to_quat(np.array([0.0, 90.0, 0.0]), degrees=True),
)

my_world.reset()

# После world reset судя по документации метода
rgbd_camera.initialize()

for _ in range(20):
    my_world.step(render=True)

if args.debug_mode:
    global depth_viewport_window
    depth_viewport_window = vp_utils.create_viewport_window(
        name="DepthCameraView",
        camera_path="/World/DepthCamera"
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
    [0.17, 0.53],   
    [0.16, 0.44], 
    [0.38, 0.55]   
]

step = 0
sensor_initialized = False
sensor_warned = False

while simulation_app.is_running(): 
    my_world.step(render=True)
    step += 1
    debug_interface.clear_points()
    try:
        pcl_data = rgbd_camera.get_point_cloud_data()
        if len(pcl_data) > 0:
            detector = FittingDetection(pcl_data, CONTAINER_CONSTRAINTS)
            detector.extract_container_pc()
            detector.voxelize_pc()
            detector.ransac_open3d()
            
            filtered_pc = detector._filtered_point_cloud
            if step % 10 == 0:
                print(filtered_pc.shape[0])
                
            if filtered_pc is not None and len(filtered_pc) > 0:
                debug_interface.draw_points(points=filtered_pc.tolist(), colors=[[0, 1, 0, 1]] * len(filtered_pc), sizes=[2] * len(filtered_pc))
            
            targets = detector.cluster_and_detect()
            if len(targets) > 0:
                centers = [t['position'].tolist() for t in targets]
                debug_interface.draw_points(points=centers, colors=[[1, 0, 0, 1]] * len(centers), sizes=[15] * len(centers))
    except KeyError:
        pass