import numpy as np
import omni.kit.ui
import omni.ui as ui
import omni.usd
import carb

from pxr import Gf, UsdGeom
from isaacsim.util.debug_draw import _debug_draw

class UIBuilder:
    """Manage extension UI and 3D Drawing"""

    def __init__(self, window_title, menu_path=None):
        self._menu = None
        self._window = None

        self._menu_path = menu_path
        self._window_title = window_title
        
        # Интерфейс отрисовки
        self._draw_interface = _debug_draw.acquire_debug_draw_interface()

        # create menu
        if self._menu_path:
            self._menu = omni.kit.ui.get_editor_menu().add_item(self._menu_path, self.on_toggle, toggle=True, value=False)

    def on_toggle(self, *args, **kwargs):
        """Toggle window visibility"""
        self.build_ui()
        if self._window is not None:
            self._window.visible = not self._window.visible

    def build_ui(self):
        """Build the Graphical User Interface (GUI) in the underlying windowing system"""
        if not self._window:
            self._window = ui.Window(title=self._window_title, visible=False, width=300, height=300)
            with self._window.frame:
                self._fitting_button = ui.Button("Draw_fitting_center", clicked_fn=self.fitting_clicked_fn)
                self._camera_button = ui.Button("Show camera orientation", clicked_fn=self.camera_clicked_fn)
                self._pc_button = ui.Button("Draw point cloud", clicked_fn=self.draw_pc)
                ui.Button("Clear Drawing", clicked_fn=self.clear_drawing)

    def fitting_clicked_fn(self):
        """Тестовая отрисовка точки"""
        self.draw_target_position(np.array([0.0, 0.0, 1.0]))

    def camera_clicked_fn(self):
        """Тестовая отрисовка лучей камеры"""
        self.draw_camera_rays(camera_prim_path="/World/PerspectiveCamera", ray_length=2.0)

    def draw_target_position(self, position):
        """Рисует красную точку в заданных координатах"""
        self._draw_interface.clear_points()
        self._draw_interface.draw_points(
            [position.tolist()], 
            colors=[[1.0, 0.0, 0.0, 1.0]], 
            sizes=[20.0]
        )

    def draw_camera_rays(self, camera_prim_path: str, ray_length: float = 1.5):
        """
        Рисует 4 луча из камеры, показывающие её направление (пирамиду обзора).
        Аргументы:
            camera_prim_path - путь к примитиву камеры на сцене (str)
            ray_length - длина лучей в метрах (float)
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(camera_prim_path)
        
        if not prim.IsValid():
            carb.log_warn(f"Camera prim not found at {camera_prim_path}")
            return

        xformable = UsdGeom.Xformable(prim)
        time_code = omni.usd.get_context().get_time_code()
        world_transform = xformable.ComputeLocalToWorldTransform(time_code)
        
        cam_pos = world_transform.ExtractTranslation()
        rot_matrix = world_transform.ExtractRotationMatrix()
        
        forward = rot_matrix.Transform(Gf.Vec3d(0, 0, -1))
        up = rot_matrix.Transform(Gf.Vec3d(0, 1, 0))
        right = rot_matrix.Transform(Gf.Vec3d(1, 0, 0))
        
        spread = ray_length * 0.5 
        
        ray_directions = [
            forward * ray_length + right * spread + up * spread,  # Верхний правый
            forward * ray_length - right * spread + up * spread,  # Верхний левый
            forward * ray_length + right * spread - up * spread,  # Нижний правый
            forward * ray_length - right * spread - up * spread,  # Нижний левый
        ]
        
        starts = []
        ends = []
        colors = []
        widths = []
        
        for d in ray_directions:
            start_pos = [cam_pos[0], cam_pos[1], cam_pos[2]]
            end_pos_gf = cam_pos + d
            end_pos = [end_pos_gf[0], end_pos_gf[1], end_pos_gf[2]]
            
            starts.extend(start_pos)
            ends.extend(end_pos)
            colors.extend([0.0, 1.0, 0.0, 1.0])
            widths.append(2.0)
            
        self._draw_interface.draw_lines(starts, ends, colors, widths)
    
    def draw_pc(self, pc: np.ndarray):
        color = [[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]
        self._draw_interface.draw_points(pc, color)

    def clear_drawing(self):
        """Очищает все отрисованные 3D объекты"""
        self._draw_interface.clear_points()
        self._draw_interface.clear_lines()

    # ====================================================================

    def cleanup(self):
        """Clean up window, menu and drawing"""
        self.clear_drawing()
        
        if self._window is not None:
            self._window.destroy()
            self._window = None
        if self._menu is not None:
            try:
                omni.kit.ui.get_editor_menu().remove_item(self._menu)
            except:
                omni.kit.ui.get_editor_menu().remove_item(self._menu_path)
            self._menu = None