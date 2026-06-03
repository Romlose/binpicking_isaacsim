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
CYL_RADIUS = 0.0125
CYL_HEIGHT = 0.04
CYL_COLOR = np.array([0.1, 0.2, 0.9])

z_pos = spawn_point[2] + CYL_RADIUS

for i in range(10):
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
loop_counter = 0

while simulation_app.is_running(): 
    my_world.step(render=True)
    debug_interface.clear_points()
    loop_counter += 1
    try:
        pcl_data = rgbd_camera.get_point_cloud_data()
        if len(pcl_data) > 0:
            detector = FittingDetection(pcl_data, CONTAINER_CONSTRAINTS)
            detector.extract_container_pc()
            detector.voxelize_pc(voxel_size=0.005)
            detector.ransac_open3d(distance_threshold=0.002)
            
            filtered_pc = detector._filtered_point_cloud
            if filtered_pc is not None and len(filtered_pc) > 0:
                step = max(1, len(filtered_pc) // 5000)
                p_carb = [carb.Float3(p[0], p[1], p[2]) for p in filtered_pc[::step]]
                c_carb = [carb.ColorRgba(0, 1, 0, 1) for _ in p_carb]
                s_carb = [2.0 for _ in p_carb]
                debug_interface.draw_points(p_carb, c_carb, s_carb)
            
            targets = detector.cluster_and_detect(eps=0.015, min_points=5)
            if len(targets) > 0:
                p_center = [carb.Float3(t['position'][0], t['position'][1], t['position'][2]) for t in targets]
                c_center = [carb.ColorRgba(1, 0, 0, 1) for _ in targets]
                s_center = [15.0 for _ in targets]
                debug_interface.draw_points(p_center, c_center, s_center)
                
                if loop_counter % 10 == 0:
                    real_cyls = []
                    for i in range(10):
                        p_path = f"/World/Cyl_{i+1}"
                        prim = stage.GetPrimAtPath(p_path)
                        if prim.IsValid():
                            xform = UsdGeom.Xformable(prim)
                            mat = xform.ComputeLocalToWorldTransform(0.0)
                            pos = mat.GetRow3(3)
                            real_cyls.append(np.array([pos[0], pos[1], pos[2]]))
                    if real_cyls:
                        if len(targets) > len(real_cyls):
                            print("Вычисленных точек больше, чем реальных цилиндров!")
                        for t in targets:
                            t_pos = np.array(t['position'])
                            min_diff = min(np.linalg.norm(t_pos - rc) for rc in real_cyls)
                            print(f"Наименьшая разница для точки: {min_diff:.4f}")
    except KeyError:
        pass

simulation_app.close()