from isaacsim import SimulationApp
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test-arg", type=str, default="test", help="Test argument.")
args, _ = parser.parse_known_args()

launch_config = {"headless": False}
simulation_app = SimulationApp(launch_config)


# Основной код
import numpy as np
import omni.usd
from pxr import UsdGeom, UsdPhysics, Gf, Sdf
from isaacsim.core.api import World
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from isaacsim.util.debug_draw import _debug_draw

from behavior_scripts import CylinderBehavior, FrankaController
from camera_pc import DepthCameraPCL, FittingDetection
from scene_config import create_table, create_source_container, create_placing_area

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()
my_world.reset()

# Получаем интерфейс для дебажнной отрисовки
debug_interface = _debug_draw.acquire_debug_draw_interface()

# Создаем сцену
table_z = create_table()
spawn_point = create_source_container(table_z)
placing_targets = create_placing_area(table_z)

# ==========================================
# 1. ДОБАВЛЯЕМ ПОЛ (Дефолтное окружение со скриншота)
# ==========================================
from isaacsim.storage.native import get_assets_root_path

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

# ==========================================
# 2. ДОБАВЛЯЕМ ДОПОЛНИТЕЛЬНЫЙ СВЕТ (при необходимости)
# ==========================================
# В default_environment.usd уже встроен базовый свет. 
# Если фитинги покажутся вам слишком темными, раскомментируйте код ниже для усиления:
# light_path = "/World/SunLight"
# light_prim = stage.DefinePrim(light_path, "DistantLight")
# light_prim.CreateAttribute("intensity", Sdf.ValueTypeNames.Float).Set(2000.0)
# UsdGeom.XformCommonAPI(light_prim).SetRotate(Gf.Vec3f(45.0, 0.0, 45.0))

# ==========================================
# Спавн фитингов
# ==========================================
initial_orientation = euler_angles_to_quat(np.array([0.0, 0.3, 0.0])) 

cyl_script = CylinderBehavior(
    prim_path="/World/SourceContainer", 
    name="fittings_spawner",
    number_cylinders=10,
    position=spawn_point.tolist(),
    orientation=initial_orientation.tolist()
)

# Создаем камеру
rgbd_camera = DepthCameraPCL(
    resolution=(1000, 1000),
    position=np.array([0.0, 0.0, 1.0]), 
    orientation=euler_angles_to_quat(np.array([0.0, 180.0, 0.0]), degrees=True),
)
# После world reset судя по документации метода
rgbd_camera.initialize()

# Границы контейнера
CONTAINER_CONSTRAINTS = [
    [0.2, 0.5],   
    [0.15, 0.45], 
    [0.42, 0.6]   
]


step = 0
sensor_initialized = False
sensor_warned = False  # Флаг для одноразового вывода сообщения

while simulation_app.is_running(): 
    my_world.step(render=True)

    position, quaternion = rgbd_camera.get_world_pose()
    print("Ориентация камеры:", quat_to_euler_angles(quaternion, degrees=True))
    print("Позиция камеры:", position)
    # Получаем данные с камеры

    point_cloud_raw = rgbd_camera.get_point_cloud_data()
    
    if point_cloud_raw is None and len(point_cloud_raw) < 0:
        continue
    
    if point_cloud_raw is not None and len(point_cloud_raw) > 0:
        # --- ПАЙПЛАЙН ОБНАРУЖЕНИЯ ФИТИНГОВ ---
        detector = FittingDetection(point_cloud_raw, CONTAINER_CONSTRAINTS)
        
        # 1. Обрезаем всё, что вне контейнера
        detector.extract_container_pc()
        
        # 2. Вокселизация 
        detector.voxelize_pc(voxel_size=0.005)
        
        # 3. Удаляем плоскость дна контейнера
        detector.ransac_open3d(distance_threshold=0.005)
        
        # 4. Ищем отдельные фитинги и их центры
        targets = detector.cluster_and_detect(eps=0.025, min_points=20)
        
        # Очищаем старую отрисовку
        debug_interface.clear_points()
        
        # Если нашли цели - рисуем их
        if len(targets) > 0:
            target_positions = []
            target_colors = []
            target_sizes = []
            
            for i, target in enumerate(targets):
                pos = target['position']
                target_positions.append(pos.tolist())
                
                if i == 0:
                    target_colors.append([0.0, 1.0, 0.0, 1.0]) # Зеленый
                else:
                    target_colors.append([0.0, 0.5, 1.0, 1.0]) # Голубой
                    
                target_sizes.append(25.0) 
            
            debug_interface.draw_points(target_positions, target_colors, target_sizes)

simulation_app.close()
