"""
目标函数模块

该模块定义了优化问题的目标函数，
用于评估优化算法找到的解的质量。

目标函数是优化算法的核心组成部分，
它将参数空间中的点映射到目标值（适应度）。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ObjectiveFunctionConfig:
    """
    目标函数配置类。
    
    Attributes:
        minimize (bool): 是否为最小化问题
        discrete_params (List[int]): 需要离散化的参数索引
        param_ranges (List[Tuple[float, float]]): 参数的离散取值范围
    """
    minimize: bool = True
    discrete_params: List[int] = None
    param_ranges: List[Tuple[float, float]] = None
    
    def __post_init__(self) -> None:
        if self.discrete_params is None:
            self.discrete_params = []
        if self.param_ranges is None:
            self.param_ranges = []


class ObjectiveFunction(ABC):
    """
    目标函数抽象基类。
    
    所有具体的目标函数都应该继承此类并实现evaluate方法。
    
    Attributes:
        config (ObjectiveFunctionConfig): 目标函数配置
        name (str): 目标函数名称
        dimension (int): 参数维度
    """
    
    def __init__(
        self, 
        name: str = "ObjectiveFunction",
        config: Optional[ObjectiveFunctionConfig] = None
    ) -> None:
        """
        初始化目标函数。
        
        Args:
            name: 目标函数名称
            config: 目标函数配置
        """
        self.name = name
        self.config = config if config is not None else ObjectiveFunctionConfig()
        self.dimension = 0
    
    @abstractmethod
    def evaluate(self, params: np.ndarray) -> float:
        """
        评估目标函数值（由子类实现）。
        
        Args:
            params: 参数向量
            
        Returns:
            float: 目标函数值
        """
        pass
    
    def __call__(self, params: np.ndarray) -> float:
        """
        使目标函数可调用。
        
        Args:
            params: 参数向量
            
        Returns:
            float: 目标函数值
        """
        return self.evaluate(params)
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取目标函数配置。
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            'name': self.name,
            'minimize': self.config.minimize,
            'dimension': self.dimension
        }


class ClusteringObjective(ObjectiveFunction):
    """
    聚类优化目标函数类。
    
    该类封装了聚类问题的目标函数，
    用于优化聚类算法的超参数。
    
    优化目标：
    - 最大化调整兰德指数（ARI）
    - 最大化归一化互信息（NMI）
    
    优化参数：
    - 聚类数量（n_clusters）
    - 亲和力参数（delta）
    
    Attributes:
        ball_dict (Dict): 粒度球字典
        original_features (np.ndarray): 原始特征
        original_ground_truth (np.ndarray): 真实标签
        discrete_params (List[int]): 需要离散化的参数索引
    """
    
    def __init__(
        self,
        ball_dict: Dict,
        original_features: np.ndarray,
        original_ground_truth: np.ndarray,
        config: Optional[ObjectiveFunctionConfig] = None
    ) -> None:
        """
        初始化聚类目标函数。
        
        Args:
            ball_dict: 粒度球字典
            original_features: 原始特征数据
            original_ground_truth: 真实标签
            config: 目标函数配置
        """
        super().__init__("ClusteringObjective", config)
        self.ball_dict = ball_dict
        self.original_features = original_features
        self.original_ground_truth = original_ground_truth
        self.dimension = 2
        self.discrete_params = [0]
    
    def evaluate(self, params: np.ndarray) -> float:
        """
        评估聚类目标函数值。
        
        Args:
            params: 参数向量 [n_clusters, delta]
            
        Returns:
            float: 负ARI值（用于最小化）
        """
        num_clusters = int(params[0])
        continuous_delta = float(params[1])
        
        if self.config.param_ranges:
            discrete_delta = self._discretize_delta(continuous_delta)
        else:
            discrete_delta = continuous_delta
        
        if num_clusters < 1:
            return 1.0
        
        ari_score, _ = self._perform_clustering(num_clusters, discrete_delta)
        
        return -ari_score
    
    def _discretize_delta(self, continuous_delta: float) -> float:
        """
        将连续delta离散化到指定范围。
        
        Args:
            continuous_delta: 连续delta值
            
        Returns:
            float: 离散化后的delta值
        """
        if not self.config.param_ranges:
            return continuous_delta
        
        min_delta, max_delta = self.config.param_ranges[1]
        step = 0.1
        
        snapped_delta = round(continuous_delta * 10.0) / 10.0
        final_delta = max(min_delta, min(max_delta, snapped_delta))
        
        return final_delta
    
    def _perform_clustering(self, num_clusters: int, delta: float) -> Tuple[float, np.ndarray]:
        """
        执行聚类并计算ARI分数。
        
        Args:
            num_clusters: 聚类数量
            delta: 亲和力参数
            
        Returns:
            Tuple[float, np.ndarray]: (调整兰德指数, 聚类标签)
        """
        import warnings
        from sklearn import metrics
        from sklearn.cluster import SpectralClustering, KMeans
        
        ball_keys = list(self.ball_dict.keys())
        num_balls = len(ball_keys)
        
        if num_balls == 0:
            return 0.0, np.array([])
        
        if num_clusters < 2:
            num_clusters = 2
        if num_clusters > num_balls:
            num_clusters = num_balls
        
        if num_clusters == 1:
            return 0.0, np.zeros(len(self.original_features), dtype=int)
        
        ball_centers = np.array([self.ball_dict[key].center for key in ball_keys])
        ball_radii = np.array([self.ball_dict[key].radius for key in ball_keys])
        
        num_valid_balls = len(ball_centers)
        if num_valid_balls < num_clusters:
            return 0.0, np.zeros(len(self.original_features), dtype=int)
        
        affinity_matrix = self._calculate_affinity_matrix(ball_centers, ball_radii, delta)
        
        affinity_matrix = np.nan_to_num(affinity_matrix)
        affinity_matrix = np.maximum(affinity_matrix, 0)
        affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                spectral = SpectralClustering(
                    n_clusters=num_clusters,
                    affinity='precomputed',
                    assign_labels='discretize',
                    random_state=42,
                    n_init=10,
                    n_jobs=-1
                )
                ball_cluster_labels = spectral.fit_predict(affinity_matrix)
        except Exception:
            try:
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                ball_cluster_labels = kmeans.fit_predict(ball_centers)
            except Exception:
                return 0.0, np.zeros(len(self.original_features), dtype=int)
        
        final_labels = self._assign_points_to_clusters(
            ball_centers, 
            ball_cluster_labels, 
            ball_keys
        )
        
        valid_indices = np.where(
            (self.original_ground_truth != -1) & (final_labels != -1)
        )[0]
        
        if len(valid_indices) < 2:
            return 0.0, np.zeros(len(self.original_features), dtype=int)
        
        ari_score = metrics.adjusted_rand_score(
            self.original_ground_truth[valid_indices],
            final_labels[valid_indices]
        )
        
        return ari_score, final_labels
    
    def _calculate_affinity_matrix(
        self, 
        centers: np.ndarray, 
        radii: np.ndarray, 
        delta: float
    ) -> np.ndarray:
        """
        计算亲和力矩阵。
        
        Args:
            centers: 球心坐标数组
            radii: 球半径数组
            delta: 亲和力参数
            
        Returns:
            np.ndarray: 亲和力矩阵
        """
        from math import exp
        
        num_balls = len(centers)
        affinity_matrix = np.zeros((num_balls, num_balls))
        
        delta_squared = 2 * (delta ** 2) if delta > 1e-9 else 1e-12
        
        for i in range(num_balls):
            affinity_matrix[i, i] = 1.0
            for j in range(i + 1, num_balls):
                distance = np.linalg.norm(centers[i] - centers[j])
                gap = distance - radii[i] - radii[j]
                
                if delta_squared <= 1e-12:
                    affinity = 1.0 if gap < 1e-9 else 0.0
                else:
                    affinity = exp(-gap / delta_squared) if gap > 0 else 1.0
                
                affinity = max(0, affinity)
                affinity_matrix[i, j] = affinity
                affinity_matrix[j, i] = affinity
        
        return affinity_matrix
    
    def _assign_points_to_clusters(
        self, 
        ball_centers: np.ndarray, 
        ball_labels: np.ndarray, 
        ball_keys: List
    ) -> np.ndarray:
        """
        将数据点分配到最近的聚类。
        
        Args:
            ball_centers: 球心坐标数组
            ball_labels: 球体聚类标签
            ball_keys: 球体键列表
            
        Returns:
            np.ndarray: 数据点聚类标签
        """
        num_points = len(self.original_features)
        final_labels = np.full(num_points, -1, dtype=int)
        
        key_to_cluster = {ball_keys[i]: ball_labels[i] for i in range(len(ball_keys))}
        
        for point_idx, point in enumerate(self.original_features):
            distances_sq = np.sum((ball_centers - point) ** 2, axis=1)
            nearest_ball_idx = np.argmin(distances_sq)
            nearest_ball_key = ball_keys[nearest_ball_idx]
            
            if nearest_ball_key in key_to_cluster:
                final_labels[point_idx] = key_to_cluster[nearest_ball_key]
        
        return final_labels
    
    def evaluate_with_nmi(self, params: np.ndarray) -> Tuple[float, float, float]:
        """
        评估聚类效果并返回ARI和NMI分数。
        
        Args:
            params: 参数向量
            
        Returns:
            Tuple[float, float, float]: (负ARI, ARI, NMI)
        """
        from sklearn import metrics
        
        num_clusters = int(params[0])
        continuous_delta = float(params[1])
        
        if self.config.param_ranges:
            discrete_delta = self._discretize_delta(continuous_delta)
        else:
            discrete_delta = continuous_delta
        
        if num_clusters < 1:
            return 1.0, 0.0, 0.0
        
        ari_score = self._perform_clustering(num_clusters, discrete_delta)
        
        final_labels = self._get_clustering_labels(num_clusters, discrete_delta)
        
        valid_indices = np.where(
            (self.original_ground_truth != -1) & (final_labels != -1)
        )[0]
        
        if len(valid_indices) < 2:
            return -ari_score, ari_score, 0.0
        
        try:
            nmi_score = metrics.normalized_mutual_info_score(
                self.original_ground_truth[valid_indices],
                final_labels[valid_indices],
                average_method='arithmetic'
            )
        except Exception:
            nmi_score = 0.0
        
        return -ari_score, ari_score, nmi_score
    
    def _get_clustering_labels(self, num_clusters: int, delta: float) -> np.ndarray:
        """
        获取聚类标签。
        
        Args:
            num_clusters: 聚类数量
            delta: 亲和力参数
            
        Returns:
            np.ndarray: 聚类标签
        """
        import warnings
        from sklearn.cluster import SpectralClustering, KMeans
        
        ball_keys = list(self.ball_dict.keys())
        num_balls = len(ball_keys)
        
        if num_balls == 0:
            return np.array([])
        
        if num_clusters < 2:
            num_clusters = 2
        if num_clusters > num_balls:
            num_clusters = num_balls
        
        if num_clusters == 1:
            return np.zeros(len(self.original_features), dtype=int)
        
        ball_centers = np.array([self.ball_dict[key].center for key in ball_keys])
        ball_radii = np.array([self.ball_dict[key].radius for key in ball_keys])
        
        affinity_matrix = self._calculate_affinity_matrix(ball_centers, ball_radii, delta)
        
        affinity_matrix = np.nan_to_num(affinity_matrix)
        affinity_matrix = np.maximum(affinity_matrix, 0)
        affinity_matrix = 0.5 * (affinity_matrix + affinity_matrix.T)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                spectral = SpectralClustering(
                    n_clusters=num_clusters,
                    affinity='precomputed',
                    assign_labels='discretize',
                    random_state=42,
                    n_init=10,
                    n_jobs=-1
                )
                ball_cluster_labels = spectral.fit_predict(affinity_matrix)
        except Exception:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            ball_cluster_labels = kmeans.fit_predict(ball_centers)
        
        final_labels = self._assign_points_to_clusters(
            ball_centers, 
            ball_cluster_labels, 
            ball_keys
        )
        
        return final_labels


def create_clustering_objective(
    ball_dict: Dict,
    features: np.ndarray,
    ground_truth: np.ndarray,
    delta_range: Tuple[float, float] = (0.1, 1.0)
) -> ClusteringObjective:
    """
    创建聚类目标函数的工厂函数。
    
    Args:
        ball_dict: 粒度球字典
        features: 特征数据
        ground_truth: 真实标签
        delta_range: delta参数范围
        
    Returns:
        ClusteringObjective: 聚类目标函数对象
    """
    config = ObjectiveFunctionConfig(
        minimize=True,
        discrete_params=[0],
        param_ranges=[(2, 20), delta_range]
    )
    
    return ClusteringObjective(
        ball_dict=ball_dict,
        original_features=features,
        original_ground_truth=ground_truth,
        config=config
    )
