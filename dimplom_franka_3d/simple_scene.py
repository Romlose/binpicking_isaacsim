from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController

print("[INIT] Создание мира...")
my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

print("[INIT] Создание Франки с безопасным раскрытием схвата...")
franka = Franka(
    prim_path="/World/Franka", 
    name="franka_robot",
    position=np.array([0.0, 0.0, 0.0]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    gripper_open_position=np.array([0.025, 0.025]), # 2.5 см на палец
    deltas=np.array([0.025, 0.025])
)
my_world.scene.add(franka)

print("[INIT] Сброс мира...")
my_world.reset()

print("[INIT] Инициализация робота...")
franka.initialize()

# Больше никаких хаков с ограничением физики суставов! Всё решено через API

print("[INIT] Создание контроллера Франки...")
pick_place_controller = PickPlaceController(
    name="pick_place_controller",
    gripper=franka.gripper,
    robot_articulation=franka,
    end_effector_initial_height=0.3, 
)

# Тестовые точки (впереди робота, на высоте 10 см над столом)
picking_position = np.array([0.5, 0.3, 0.1])  
placing_position = np.array([0.5, -0.3, 0.1]) 

print("[INIT] Готово! Франка двигается и использует безопасное открытие схвата.")

cycle_count = 0

while simulation_app.is_running():
    my_world.step(render=True)
    
    joint_positions = franka.get_joint_positions()
    
    if pick_place_controller.is_done():
        cycle_count += 1
        print(f"\n[TEST] Цикл {cycle_count} завершен! Сброс...\n")
        pick_place_controller.reset()
    
    action = pick_place_controller.forward(
        picking_position=picking_position,
        placing_position=placing_position,
        current_joint_positions=joint_positions
    )
    franka.get_articulation_controller().apply_action(action)

simulation_app.close()