import carb
import omni.ext
import omni.kit.app

from .ui_builder import UIBuilder

class Extension(omni.ext.IExt):
    """The Extension class"""

    def on_startup(self, ext_id):
        """Method called when the extension is loaded/enabled"""
        carb.log_info(f"on_startup {ext_id}")
        ext_path = omni.kit.app.get_app().get_extension_manager().get_extension_path(ext_id)

        self.ui_builder = UIBuilder(window_title="Pc Camera Ui", menu_path="Window/Pc Camera Ui")
        self.ui_builder.build_ui()
        if self.ui_builder._window:
            self.ui_builder._window.visible = True
            
    def on_shutdown(self):
        """Method called when the extension is disabled"""
        carb.log_info(f"on_shutdown")
        self.ui_builder.cleanup()

    def get_ui_builder(self) -> UIBuilder:
        return self.ui_builder