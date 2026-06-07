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
            frequency=60,
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
        if hasattr(point_cloud, 'numpy'):
            point_cloud = point_cloud.numpy()
            
        self._point_cloud_raw = np.asarray(point_cloud, dtype=np.float32)
        
        if len(self._point_cloud_raw.shape) == 1 and self._point_cloud_raw.size > 0:
            self._point_cloud_raw = self._point_cloud_raw.reshape(-1, 3)
            
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

    def voxelize_pc(self, voxel_size: float = 0.001) -> np.ndarray:
        if self._filtered_point_cloud is None or len(self._filtered_point_cloud) == 0:
            return np.array([])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._filtered_point_cloud)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        self._filtered_point_cloud = np.asarray(downsampled_pcd.points)
        return self._filtered_point_cloud

    def remove_container_planes_iterative(self, distance_threshold: float = 0.004, min_plane_ratio: float = 0.05, max_iterations: int = 1000) -> np.ndarray:
        """
        Итеративно находит и удаляет плоские поверхности (пол и стенки контейнера).
        Останавливается, когда самая большая оставшаяся плоскость составляет меньше min_plane_ratio от облака.
        """
        if self._filtered_point_cloud is None or len(self._filtered_point_cloud) < 100:
            return self._filtered_point_cloud

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._filtered_point_cloud)
        
        planes_removed = 0
        
        for _ in range(5):
            if len(pcd.points) < 100:
                break
                
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=distance_threshold, 
                ransac_n=3, 
                num_iterations=max_iterations
            )
            
            if len(inliers) / len(pcd.points) < min_plane_ratio:
                break
                
            pcd = pcd.select_by_index(inliers, invert=True)
            planes_removed += 1

        self._filtered_point_cloud = np.asarray(pcd.points)
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
    
    def get_highest_point_grasp(self):
        if self._filtered_point_cloud is None or len(self._filtered_point_cloud) == 0:
            return None
            
        highest_idx = np.argmax(self._filtered_point_cloud[:, 2])
        position = self._filtered_point_cloud[highest_idx]
        
        return {
            'position': position,
            'approach': np.array([0.0, 0.0, -1.0])
        }

    def get_top_cluster_grasp(self, eps=0.02, min_points=10):
        if self._filtered_point_cloud is None or len(self._filtered_point_cloud) < min_points:
            return None
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._filtered_point_cloud)
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
        
        if labels.max() < 0:
            return None
            
        max_z = -np.inf
        top_cluster_pts = None
        
        for cid in range(labels.max() + 1):
            idx = np.where(labels == cid)[0]
            pts = self._filtered_point_cloud[idx]
            cz = np.max(pts[:, 2])
            
            if cz > max_z:
                max_z = cz
                top_cluster_pts = pts
                
        if top_cluster_pts is None or len(top_cluster_pts) < 3:
            return None
            
        pca = PCA(n_components=3)
        pca.fit(top_cluster_pts)
        
        return {
            'position': pca.mean_,
            'cylinder_axis': pca.components_[0],
            'approach': np.array([0.0, 0.0, -1.0])
        }
        
    def get_centroid_clusters_grasp(self, eps: float = 0.015, min_points: int = 30) -> list[dict]:
        if self._filtered_point_cloud is None or len(self._filtered_point_cloud) < min_points:
            return []

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._filtered_point_cloud)
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
        
        if len(labels) == 0 or labels.max() < 0:
            return []

        targets = []
        for cid in range(labels.max() + 1):
            idx = np.where(labels == cid)[0]
            pts = self._filtered_point_cloud[idx]

            if len(pts) < min_points:
                continue

            min_bound = np.min(pts, axis=0)
            max_bound = np.max(pts, axis=0)
            extent = max_bound - min_bound
            
            if np.max(extent) > 0.10:
                continue

            # --- НАДЕЖНОЕ ВЫЧИСЛЕНИЕ ОРИЕНТАЦИИ ---
            pca = PCA(n_components=3)
            pca.fit(pts)
            cylinder_axis = pca.components_[0]
            
            # ФИКСИРУЕМ ЗНАК ОСИ! 
            # Гарантируем, что вектор всегда смотрит в одну сторону полупространства (X > 0)
            # Это убирает скачки ориентации на 180 градусов
            if cylinder_axis[0] < 0:
                cylinder_axis = -cylinder_axis
            elif cylinder_axis[0] == 0 and cylinder_axis[1] < 0:
                cylinder_axis = -cylinder_axis
            
            axis_xy = np.array([cylinder_axis[0], cylinder_axis[1]])
            
            if np.linalg.norm(axis_xy) > 1e-6:
                # Угол оси цилиндра
                alpha = np.arctan2(axis_xy[1], axis_xy[0])
                # Схват перпендикулярен оси (+ 90 градусов)
                yaw_angle = alpha + np.pi/2
            else:
                # Цилиндр стоит вертикально, вращение не критично
                yaw_angle = 0.0
                
            # ИСПОЛЬЗУЕМ ВСТРОЕННУЮ ФУНКЦИЮ EULER -> QUAT (Она гораздо стабильнее!)
            # X=0, Y=pi (смотрит вниз), Z=yaw_angle (поворот вокруг своей оси)
            # Убедись, что euler_angles_to_quat импортирована в camera_pc.py!
            from isaacsim.core.utils.rotations import euler_angles_to_quat
            target_orientation = euler_angles_to_quat(np.array([0, np.pi, yaw_angle]), degrees=False)
            # ----------------------------------------

            centroid = np.mean(pts, axis=0)

            targets.append({
                'position': centroid,
                'orientation': target_orientation,
                'num_points': len(pts),
                'cluster_points': pts
            })

        targets.sort(key=lambda x: x['num_points'], reverse=True)
        return targets
    
    def ransac_remove_horizontal_planes(self, distance_threshold: float = 0.005, num_iterations: int = 1000, min_plane_ratio: float = 0.03, z_normal_threshold: float = 0.95) -> np.ndarray:
        if self._filtered_point_cloud is None or len(self._filtered_point_cloud) == 0:
            return np.array([])
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._filtered_point_cloud)
        
        while True:
            if len(pcd.points) < 100:
                break
                
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=distance_threshold, 
                ransac_n=3, 
                num_iterations=num_iterations
            )
            
            if len(inliers) / len(pcd.points) < min_plane_ratio:
                break
                
            pcd = pcd.select_by_index(inliers, invert=True)
        
        self._filtered_point_cloud = np.asarray(pcd.points)
        return self._filtered_point_cloud
    
    def remove_noise_and_edges(self, nb_neighbors=20, std_ratio=1.5):
        """
        Удаляет статистические выбросы (шум, края стенок, одинокие точки).
        Оставляет только плотные кластеры (наши цилиндры).
        """
        if self._filtered_point_cloud is None or len(self._filtered_point_cloud) < nb_neighbors:
            return self._filtered_point_cloud

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._filtered_point_cloud)
        
        # nb_neighbors: сколько соседей рассматривать для каждой точки
        # std_ratio: насколько точка может отклоняться от среднего расстояния до соседей. 
        # Чем меньше, тем агрессивнее удаление. 1.5 - хороший баланс.
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        
        if len(ind) == 0:
            self._filtered_point_cloud = np.array([])
        else:
            clean_pcd = pcd.select_by_index(ind)
            self._filtered_point_cloud = np.asarray(clean_pcd.points)
            
        print(f"[Noise Removal] Точек после фильтрации плотности: {len(self._filtered_point_cloud)}")
        return self._filtered_point_cloud