import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA

from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.rotations import euler_angles_to_quat



class DepthCameraPCL:
    """
    Класс для создания камеры глубины
    """
    def __init__(
        self,
        prim_path: str = "/World/DepthCamera",
        resolution: tuple = (1000, 1000),
        position: np.ndarray = None,
        orientation: np.ndarray = None,
    ):
        self._camera_path = prim_path
        self._resolution = resolution
        self._position = position if position is not None else np.array([0.0, 0.0, 1.5])
        self._orientation = orientation if orientation is not None else euler_angles_to_quat(np.array([0.0, 180.0, 0.0]), degrees=True)

        self._camera = Camera(
            prim_path=self._camera_path,
            resolution=self._resolution,
            position=self._position,
            orientation=self._orientation,
            frequency=20,
            annotator_device = "cuda:0"
        )
        self._is_initialized = False

    def initialize(self):
        if not self._is_initialized:
            self._camera.initialize()
            self._camera.add_distance_to_image_plane_to_frame()
            self._camera.add_pointcloud_to_frame()
            self._is_initialized = True
            print(f"Camera {self._camera_path} initialized (Official API).")

    def get_point_cloud_data(self) -> np.ndarray:
        """Возвращает массив точек облака (Nx3) в мировых координатах."""
        if not self._is_initialized:
            return np.array([])
        
        pcl_data = self._camera.get_pointcloud()
        
        if pcl_data is None or len(pcl_data) == 0:
            return np.array([])
            
        return pcl_data

    def get_world_pose(self):
        """Возвращает позицию и ориентацию в кватернионах камеры в мире"""
        if self._is_initialized:
            return self._camera.get_world_pose()
        return self._position, self._orientation

    def set_world_pose(self, position: np.ndarray = None, orientation: np.ndarray = None) -> None:
        if position is not None:
            self._position = position
        if orientation is not None:
            self._orientation = orientation
        if self._is_initialized:
            self._camera.set_world_pose(position=self._position, orientation=self._orientation)


class FittingDetection:
    def __init__(self, point_cloud: np.ndarray, coord_constraints: list[list]):
        self._point_cloud_raw = point_cloud
        self._coord_constraints = coord_constraints
        self._filtered_point_cloud = None
    
    def extract_container_pc(self, coord_constraints: list[list] = None) -> np.ndarray:
        if coord_constraints is not None:
            self._coord_constraints = coord_constraints
            
        if self._point_cloud_raw is None or len(self._point_cloud_raw) == 0:
            return np.array([])
            
        mask = (
            (self._point_cloud_raw[:, 0] >= self._coord_constraints[0][0]) & 
            (self._point_cloud_raw[:, 0] <= self._coord_constraints[0][1]) &
            (self._point_cloud_raw[:, 1] >= self._coord_constraints[1][0]) & 
            (self._point_cloud_raw[:, 1] <= self._coord_constraints[1][1]) &
            (self._point_cloud_raw[:, 2] >= self._coord_constraints[2][0]) & 
            (self._point_cloud_raw[:, 2] <= self._coord_constraints[2][1])
        )
        self._filtered_point_cloud = self._point_cloud_raw[mask]
        return self._filtered_point_cloud

    def voxelize_pc(self, voxel_size: float = 0.005) -> np.ndarray:
        if self._filtered_point_cloud is None or len(self._filtered_point_cloud) == 0:
            return np.array([])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._filtered_point_cloud)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        self._filtered_point_cloud = np.asarray(downsampled_pcd.points)
        return self._filtered_point_cloud

    def ransac_open3d(self, distance_threshold: float = 0.005, num_iterations: int = 1000) -> np.ndarray:
        if self._filtered_point_cloud is None or len(self._filtered_point_cloud) == 0:
            return np.array([])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._filtered_point_cloud)
        
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold, 
            ransac_n=3, 
            num_iterations=num_iterations
        )
        
        outlier_pcd = pcd.select_by_index(inliers, invert=True)
        self._filtered_point_cloud = np.asarray(outlier_pcd.points)
        return self._filtered_point_cloud

    def cluster_and_detect(self, eps: float = 0.02, min_points: int = 10) -> list[dict]:
        """
        Разделяет кучу фитингов на кластеры (DBSCAN) и для каждого находит центр и ориентацию (PCA).
        
        Args:
            eps (float): Максимальное расстояние между точками, чтобы они считались одним кластером.
                         Для фитингов 40мм (0.04м) значение 0.02-0.03 - идеальный старт.
            min_points (int): Минимальное количество точек в кластере, чтобы это не считалось мусором.
        
        Returns:
            list[dict]: Список словарей с данными по каждому найденному фитингу:
                        [{'position': np.array, 'orientation_axes': np.array, 'num_points': int}, ...]
        """
        if self._filtered_point_cloud is None or len(self._filtered_point_cloud) < min_points:
            return []

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._filtered_point_cloud)

        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
        
        max_label = labels.max()
        if max_label < 0:
            print("No clusters found, only noise")
            return []

        fitting_targets = []

        for cluster_id in range(max_label + 1):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_pcd = pcd.select_by_index(cluster_indices)
            cluster_points = np.asarray(cluster_pcd.points)

            if len(cluster_points) < 3:
                continue

            # --- PCA ДЛЯ ОДНОГО ФИТИНГА ---
            pca = PCA(n_components=3)
            pca.fit(cluster_points)
            
            centroid = pca.mean_
            axes = pca.components_

            fitting_targets.append({
                'position': centroid,
                'orientation_axes': axes,
                'num_points': len(cluster_points)
            })

        # Сортируем цели по размеру кластера (крупные кластеры = более целые фитинги - берем первыми)
        fitting_targets.sort(key=lambda x: x['num_points'], reverse=True)

        return fitting_targets
        