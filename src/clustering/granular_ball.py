"""
粒度球核心类

该模块实现了粒度球（Granular Ball）的核心数据结构，
用于无监督学习任务中的聚类分析。

粒度球是一种基于密度的数据表示方法，将数据点组织成球形区域，
每个球体包含具有相似特征的数据点。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import numpy.linalg as la


@dataclass
class GranularBallConfig:
    """
    粒度球配置类。
    
    Attributes:
        min_points_for_split (int): 进行分裂的最小点数
        min_points_after_split (int): 分裂后每个子球的最小点数
        max_iterations (int): 最大迭代次数
        pca_components (int): PCA降维维度
        normalize_range (Tuple[float, float]): 归一化范围
    """
    min_points_for_split: int = 8
    min_points_after_split: int = 4
    max_iterations: int = 50
    pca_components: int = 2
    normalize_range: Tuple[float, float] = (0, 1)


class GranularBall:
    """
    粒度球类。
    
    粒度球是数据点的集合，具有以下属性：
    - points: 包含的所有数据点
    - center: 球心（所有点的均值）
    - label: 聚类标签
    - radius: 半径（最远点到球心的距离）
    - point_count: 包含的点数
    
    粒度球的核心思想：
    1. 将数据空间划分为多个球形区域
    2. 每个球体代表一个密集的数据簇
    3. 通过分裂稀疏的球体来细化聚类
    
    Attributes:
        points (np.ndarray): 包含的数据点，形状为 [n_points, n_features]
        center (np.ndarray): 球心坐标，形状为 [n_features]
        label (int): 聚类标签
        radius (float): 球体半径
        point_count (int): 数据点数量
    """
    
    def __init__(
        self, 
        points: np.ndarray, 
        label: int = 0,
        calculate_radius: bool = True
    ) -> None:
        """
        初始化粒度球。
        
        Args:
            points: 数据点数组，形状为 [n_points, n_features]
            label: 初始聚类标签，默认为0
            calculate_radius: 是否计算半径，默认为True
        """
        self.points = points
        self.label = label
        self.point_count = len(points)
        
        if self.point_count > 0:
            self.center = np.mean(points, axis=0)
        else:
            self.center = np.array([])
        
        if calculate_radius:
            self.radius = self._calculate_radius()
        else:
            self.radius = 0.0
    
    def _calculate_radius(self) -> float:
        """
        计算球体半径。
        
        半径定义为球心到所有点的最大欧氏距离。
        
        Returns:
            float: 球体半径，如果无数据点则返回0.0
        """
        if self.point_count == 0:
            return 0.0
        if self.point_count == 1:
            return 0.0
        
        distances = la.norm(self.points - self.center, axis=1)
        return np.max(distances)
    
    def get_density(self) -> float:
        """
        计算球体的密度。
        
        密度定义为点数与到球心平均距离的比值。
        密度越高表示球体越紧凑。
        
        Returns:
            float: 球体密度，点数不足或半径为0时返回特殊值
        """
        if self.point_count <= 1:
            return float(self.point_count)
        
        distances = la.norm(self.points - self.center, axis=1)
        sum_radius = np.sum(distances)
        mean_radius = sum_radius / self.point_count if self.point_count > 0 else 0
        
        if sum_radius > 1e-9:
            density = self.point_count / sum_radius
        else:
            density = float('inf') if self.point_count > 0 else 0.0
        
        return density
    
    def is_valid(self) -> bool:
        """
        检查粒度球是否有效。
        
        有效球体至少包含一个数据点。
        
        Returns:
            bool: 球体是否有效
        """
        return self.point_count > 0
    
    def split(self) -> Tuple['GranularBall', 'GranularBall']:
        """
        将球体分裂为两个子球。
        
        基于距离最远的两个点将数据分为两组。
        
        Returns:
            Tuple[GranularBall, GranularBall]: 两个子球体
        """
        if self.point_count < 2:
            return self, GranularBall(np.array([]), self.label)
        
        points_child1, points_child2 = self._split_by_distance()
        
        ball1 = GranularBall(points_child1, self.label)
        ball2 = GranularBall(points_child2, self.label)
        
        return ball1, ball2
    
    def _split_by_distance(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于距离分裂数据点。
        
        找到距离最远的两个点作为分裂中心，
        其他点根据到这两个中心的距离分配到两个子球。
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 两个子球的数据点
        """
        ball1_points: List[np.ndarray] = []
        ball2_points: List[np.ndarray] = []
        num_points, num_features = self.points.shape
        
        if num_points < 2:
            return self.points, np.array([])
        
        transposed_points = self.points.T
        gram_matrix = np.dot(transposed_points.T, transposed_points)
        diag_gram = np.diag(gram_matrix)
        h_matrix = np.tile(diag_gram, (num_points, 1))
        
        distance_matrix_sq = np.maximum(0, h_matrix + h_matrix.T - gram_matrix * 2)
        distance_matrix = np.sqrt(distance_matrix_sq)
        
        if np.max(distance_matrix) == 0:
            mid_idx = num_points // 2
            return self.points[:mid_idx], self.points[mid_idx:]
        
        row_indices, col_indices = np.where(distance_matrix == np.max(distance_matrix))
        
        point1_idx = -1
        point2_idx = -1
        
        for r_idx, c_idx in zip(row_indices, col_indices):
            if r_idx != c_idx:
                point1_idx = r_idx
                point2_idx = c_idx
                break
        
        if point1_idx == -1:
            mid_idx = num_points // 2
            return self.points[:mid_idx], self.points[mid_idx:]
        
        for j in range(num_points):
            dist_to_p1 = distance_matrix[j, point1_idx]
            dist_to_p2 = distance_matrix[j, point2_idx]
            if dist_to_p1 <= dist_to_p2:
                ball1_points.append(self.points[j, :])
            else:
                ball2_points.append(self.points[j, :])
        
        if not ball1_points:
            ball1_points.append(self.points[point2_idx])
            ball2_points = [p for i, p in enumerate(ball2_points) 
                           if not np.array_equal(p, self.points[point2_idx])]
        elif not ball2_points:
            ball2_points.append(self.points[point1_idx])
            ball1_points = [p for i, p in enumerate(ball1_points) 
                           if not np.array_equal(p, self.points[point1_idx])]
        
        return np.array(ball1_points), np.array(ball2_points)
    
    def normalize(self, reference_radius: float) -> 'GranularBall':
        """
        根据参考半径归一化球体。
        
        如果球体半径远大于参考半径，则分裂球体。
        
        Args:
            reference_radius: 参考半径
            
        Returns:
            GranularBall: 归一化后的球体（可能是分裂后的）
        """
        if self.point_count < 2:
            return self
        
        if self.radius <= 2.0 * reference_radius:
            return self
        
        ball1_points, ball2_points = self._split_by_distance()
        
        if len(ball1_points) > 0:
            return GranularBall(ball1_points, self.label)
        return self
    
    def __repr__(self) -> str:
        """
        返回粒度球的字符串表示。
        
        Returns:
            str: 字符串表示
        """
        return (f"GranularBall(points={self.point_count}, "
                f"center={self.center[:3] if len(self.center) > 3 else self.center}, "
                f"radius={self.radius:.4f}, label={self.label})")
    
    def __len__(self) -> int:
        """
        返回球体包含的数据点数量。
        
        Returns:
            int: 数据点数量
        """
        return self.point_count
    
    def to_dict(self) -> dict:
        """
        将粒度球转换为字典表示。
        
        Returns:
            dict: 包含粒度球信息的字典
        """
        return {
            'points': self.points,
            'center': self.center,
            'label': self.label,
            'radius': self.radius,
            'point_count': self.point_count
        }


class GranularBallCollection:
    """
    粒度球集合类。
    
    该类管理一组粒度球，提供批量操作和聚类功能。
    
    Attributes:
        balls (List[GranularBall]): 粒度球列表
        config (GranularBallConfig): 粒度球配置
    """
    
    def __init__(self, config: Optional[GranularBallConfig] = None) -> None:
        """
        初始化粒度球集合。
        
        Args:
            config: 粒度球配置，默认为默认配置
        """
        self.balls: List[GranularBall] = []
        self.config = config if config is not None else GranularBallConfig()
    
    def add_ball(self, ball: GranularBall) -> None:
        """
        向集合中添加一个粒度球。
        
        Args:
            ball: 要添加的粒度球
        """
        if ball.is_valid():
            self.balls.append(ball)
    
    def add_balls_from_points(self, points: np.ndarray, start_label: int = 0) -> None:
        """
        从数据点创建并添加粒度球。
        
        Args:
            points: 数据点数组
            start_label: 初始标签
        """
        ball = GranularBall(points, start_label)
        self.add_ball(ball)
    
    def get_total_points(self) -> int:
        """
        获取所有粒度球包含的总点数。
        
        Returns:
            int: 总点数
        """
        return sum(len(ball) for ball in self.balls)
    
    def get_all_centers(self) -> np.ndarray:
        """
        获取所有粒度球的球心。
        
        Returns:
            np.ndarray: 球心数组，形状为 [n_balls, n_features]
        """
        return np.array([ball.center for ball in self.balls if ball.is_valid()])
    
    def get_all_radii(self) -> np.ndarray:
        """
        获取所有粒度球的半径。
        
        Returns:
            np.ndarray: 半径数组
        """
        return np.array([ball.radius for ball in self.balls if ball.is_valid()])
    
    def get_valid_balls(self) -> List[GranularBall]:
        """
        获取所有有效的粒度球。
        
        Returns:
            List[GranularBall]: 有效粒度球列表
        """
        return [ball for ball in self.balls if ball.is_valid()]
    
    def __len__(self) -> int:
        """
        返回集合中粒度球的数量。
        
        Returns:
            int: 粒度球数量
        """
        return len(self.balls)
    
    def __iter__(self):
        """
        返回迭代器。
        """
        return iter(self.balls)
    
    def __getitem__(self, index: int) -> GranularBall:
        """
        获取指定索引的粒度球。
        
        Args:
            index: 索引
            
        Returns:
            GranularBall: 粒度球
        """
        return self.balls[index]
