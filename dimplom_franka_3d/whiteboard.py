import numpy as np

class PCfromCamera: 
    def __init__(self):
        self.point_cloud = None
        self.is_grasped = False
    
    def set_pc(self, point_cloud: np.ndarray):
        self.point_cloud = point_cloud
    def get_pc(self):
        return self.point_cloud
    
    def set_flag(self, flag: bool): 
        self.is_grasped = flag
    def get_flag(self):
        return self.is_grasped

# Создаем экзепляр для всей области видимости
whiteboard = PCfromCamera()