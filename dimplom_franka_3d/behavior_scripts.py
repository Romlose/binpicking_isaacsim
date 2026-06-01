import numpy as np
import carb
import omni
from pxr.UsdGeom import Cylinder
from omni.kit.scripting import BehaviorScript
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.core.utils.types import ArticulationActions
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.core.api.materials import PhysicsMaterial, OmniPBR
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController



class CylinderBehavior(BehaviorScript):
    def __init__(self, prim_path, name: str, 
                 number_cylinders: int, 
                 position: list, 
                 orientation: list,
                 radius: float = 0.02,
                 height: float = 0.06,
                 material_path = "/World/physics_materials/polypropylene",
                 visual_material_path = "/World/material/polypropylene_visual"):
        self.name = name
        self.num_cyl = number_cylinders
        self.position = position
        self.orientation = orientation
        self.radius = radius
        self.height = height
        self.cylinder_prims = []
        
        self.material = PhysicsMaterial(prim_path=material_path,
                                        static_friction=0.45,
                                        dynamic_friction=0.40, 
                                        restitution=0.3)
        self.visual_material = OmniPBR(
                                prim_path=visual_material_path,
                                color=np.array([0.5, 0.5, 0.5])
                            )
        self.visual_material.set_reflection_roughness(0.45)
        self.visual_material.set_metallic_constant(0.0)
        
        super().__init__(prim_path)
    def on_init(self):
        carb.log_info(f"{type(self).__name__}.on_init()->{self.prim_path}")
        stage = omni.usd.get_context().get_stage()
        
        for i in range(self.num_cyl):
            cyl_path = f"{self.prim_path}/Cylinder_{i}"           
            cyl_usd = Cylinder.Define(stage, cyl_path)
            cyl_usd.GetRadiusAttr().Set(self.radius)
            cyl_usd.GetHeightAttr().Set(self.height)            
            cyl_prim = SingleGeometryPrim(prim_path=cyl_path, name=f"{self.name}_{i}")
            
            self.cylinder_prims.append(cyl_prim)

    def on_destroy(self):
        carb.log_info(f"{type(self).__name__}.on_destroy()->{self.prim_path}")
        self.name = ""
        self.num_cyl = 0
        self.cylinder_prims.clear()

    def on_play(self):
        carb.log_info(f"{type(self).__name__}.on_play()->{self.prim_path}")
        for prim in self.cylinder_prims: 
            prim.set_collision_enabled(True)
            prim.set_rigid_body_enabled(True)
            prim.apply_physics_material(self.material)
            prim.apply_visual_material(self.visual_material)
            prim.set_world_pose(position=np.array(self.position), 
                                orientation=np.array(self.orientation))

    def on_pause(self):
        carb.log_info(f"{type(self).__name__}.on_pause()->{self.prim_path}")

    def on_stop(self):
        carb.log_info(f"{type(self).__name__}.on_stop()->{self.prim_path}")

    def on_update(self, current_time: float, delta_time: float):
        pass


class FrankaController(BehaviorScript):
    def __init__(self, prim_path):
        super().__init__(prim_path)
        self.franka = Franka(prim_path="/World/Franka",
                                name="franka_robot",
                                position=np.array([0.0, 0.0, 0.0]),
                                orientation=np.array([1.0, 0.0, 0.0, 0.0])
                                )
        self.gripper = self.franka.gripper

    def on_init(self):
        carb.log_info(f"{type(self).__name__}.on_init()->{self.prim_path}")
        self.base_controller = RMPFlowController(name="franka_rmp_core", 
                                                robot_articulation=self.franka
                                                )
        # Для позиции с модуля ориентации фитинга
        self.postion = None
        self.orientation = None

    def on_destroy(self):
        carb.log_info(f"{type(self).__name__}.on_destroy()->{self.prim_path}")
        self.franka = None
        self.gripper = None
        self.base_controller = None

    def on_play(self):
        carb.log_info(f"{type(self).__name__}.on_play()->{self.prim_path}")
        # Если мы получили позы с камеры то окей
        if self.postion and self.orientation:
            rmp_action = self.base_controller.forward(self.postion, 
                                                      self.orientation,
                                                      current_joint_positions=self.franka.get_joint_positions()
                                                      )
            self.franka.apply_action(rmp_action)

    def on_pause(self):
        carb.log_info(f"{type(self).__name__}.on_pause()->{self.prim_path}")

    def on_stop(self):
        carb.log_info(f"{type(self).__name__}.on_stop()->{self.prim_path}")

    def on_update(self, current_time: float, delta_time: float):
        carb.log_info(f"{type(self).__name__}.on_update({current_time}, {delta_time})->{self.prim_path}")
        #Здесь логика достижения цели и захвата
