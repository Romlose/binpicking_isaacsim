import numpy as np
import omni.usd
from pxr.UsdGeom import Cube, Cylinder
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.core.api.materials import PhysicsMaterial, OmniPBR
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
import isaacsim.core.utils.stage as stage_utils
from omni.isaac.core.prims import GeometryPrim



table_physics = PhysicsMaterial(prim_path="/World/physics_materials/table", static_friction=0.8, dynamic_friction=0.7, restitution=0.1)
table_visual = OmniPBR(prim_path="/World/material/table_visual", color=np.array([0.7, 0.5, 0.3]))

tray_physics = PhysicsMaterial(prim_path="/World/physics_materials/tray", static_friction=0.8, dynamic_friction=0.6, restitution=0.1)
tray_visual = OmniPBR(prim_path="/World/material/tray_visual", color=np.array([0.4, 0.4, 0.45]))



def create_box(prim_path: str, name: str, position: np.ndarray, size: np.ndarray, physics_mat: PhysicsMaterial, visual_mat: OmniPBR, collision: bool = True):

    stage = omni.usd.get_context().get_stage()
    
    Cube.Define(stage, prim_path)
    
    box_prim = SingleGeometryPrim(
        prim_path=prim_path, 
        name=name, 
        position=position, 
        scale=size,
        collision=collision 
    )
    
    box_prim.apply_physics_material(physics_mat)
    box_prim.apply_visual_material(visual_mat)
    
    return box_prim


def create_table(position=np.array([0.2, 0.0, 0.4]), size=np.array([0.4, 0.8, 0.04])) -> float:
    """Создает стол и возвращает Z-координату его поверхности"""
    create_box("/World/Table", "table", position, size, table_physics, table_visual)
    table_top_z = position[2] + size[2] / 2
    return table_top_z


def create_source_container(table_z: float, offset_x=0.35, offset_y=0.3) -> np.ndarray:
    source_pos = np.array([offset_x, offset_y, table_z])
    prim_path = "/World/SourceContainer"
    
    BIN_W, BIN_D, BIN_H = 0.36, 0.28, 0.10
    WALL_T = 0.008
    FLOOR_T = 0.02
    HALF_W, HALF_D = BIN_W/2, BIN_D/2
    
    MTUCI_PURPLE    = np.array([0.42, 0.10, 0.58])
    MTUCI_PURPLE_LT = np.array([0.50, 0.18, 0.65])
    
    FixedCuboid(prim_path=f"{prim_path}/bottom", name="bin_bottom",
        position=np.array([offset_x, offset_y, table_z + FLOOR_T/2]),
        scale=np.array([BIN_W, BIN_D, FLOOR_T]), color=MTUCI_PURPLE)
    
    for wn, dx, dy, sx, sy in [
        ("wall_front", 0, -(HALF_D-WALL_T/2), BIN_W, WALL_T),
        ("wall_back",  0,  (HALF_D-WALL_T/2), BIN_W, WALL_T),
        ("wall_left", -(HALF_W-WALL_T/2), 0, WALL_T, BIN_D),
        ("wall_right", (HALF_W-WALL_T/2), 0, WALL_T, BIN_D),
    ]:
        FixedCuboid(prim_path=f"{prim_path}/{wn}", name=f"bin_{wn}",
            position=np.array([offset_x+dx, offset_y+dy, table_z + FLOOR_T + BIN_H/2]),
            scale=np.array([sx, sy, BIN_H]), color=MTUCI_PURPLE_LT)
        
    return source_pos + np.array([0, 0, FLOOR_T + 0.01])


def create_placing_area(table_top_z: float, offset_x=0.35, offset_y=-0.3) -> list:
    """Создает зону укладки с маркерами. Возвращает список из 10 координат для робота"""
    dest_pos = np.array([offset_x, offset_y, table_top_z])
    thickness = 0.01
    dest_length = 5 * 0.06 + 0.02
    dest_width = 2 * 0.06 + 0.02
    dest_wall_height = 0.015

    # Дно
    create_box("/World/DestTray/Bottom", "dest_bottom", 
               position=dest_pos + np.array([0, 0, thickness/2]), 
               size=np.array([dest_length, dest_width, thickness]), 
               physics_mat=tray_physics, visual_mat=tray_visual)

    # 4 стенки
    for i, offset in enumerate([
        np.array([ dest_length/2 - thickness/2, 0, 0]),
        np.array([-dest_length/2 + thickness/2, 0, 0]),
        np.array([0,  dest_width/2 - thickness/2, 0]),
        np.array([0, -dest_width/2 + thickness/2, 0]),
    ]):
        wall_size = np.array([
            thickness if i < 2 else dest_length, 
            dest_width if i < 2 else thickness, 
            dest_wall_height
        ])
        create_box(f"/World/DestTray/Wall_{i}", f"dest_wall_{i}", 
                   position=dest_pos + offset + np.array([0, 0, dest_wall_height/2]), 
                   size=wall_size, 
                   physics_mat=tray_physics, visual_mat=tray_visual)

    # Маркеры и сбор координат
    stage = omni.usd.get_context().get_stage()
    start_x = dest_pos[0] - 2 * 0.06
    start_y = dest_pos[1] - 0.5 * 0.06
    marker_z = dest_pos[2] + thickness + 0.002

    target_positions = []
    
    for col in range(5):
        for row in range(2):
            x = start_x + col * 0.06
            y = start_y + row * 0.06
            
            # Создаем визуальный маркер
            marker_path = f"/World/DestTray/Slot_{col}_{row}"
            cyl_usd = Cylinder.Define(stage, marker_path)
            cyl_usd.GetRadiusAttr().Set(0.018)
            cyl_usd.GetHeightAttr().Set(0.004)
            cyl_usd.GetAxisAttr().Set("Z")
            
            marker_prim = SingleGeometryPrim(
                prim_path=marker_path, 
                name=f"slot_{col}_{row}",
                position=np.array([x, y, marker_z]),
                collision=False # Маркерам коллизия не нужна!
            )
            marker_prim.apply_visual_material(tray_visual)
            
            place_z = marker_z + 0.002 + (0.09 / 2) 
            target_positions.append(np.array([round(x, 4), round(y, 4), round(place_z, 4)]))

    return target_positions