"""
分裂策略模块

该模块提供了粒度球的分裂策略实现，
用于根据不同的策略将粒度球分裂成更小的子球。

分裂策略是粒度球聚类算法的核心操作，
决定了如何将一个大的数据球体分解为更小的、更加紧凑的子球。
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Union
import numpy as np
import numpy.linalg as la


class SplittingStrategy(Enum):
    """
    分裂策略枚举类。
    
    定义了不同的分裂策略：
    - DISTANCE_BASED: 基于距离的分裂策略
    - DENSITY_BASED: 基于密度的分裂策略
    - RANDOM: 随机分裂策略
    - KMEANS: 基于K-Means的分裂策略
    """
    DISTANCE_BASED = "distance_based"
    DENSITY_BASED = "density_based"
    RANDOM = "random"
    KMEANS = "kmeans"


@dataclass
class SplittingConfig:
    """
    分裂策略配置类。
    
    Attributes:
        strategy (SplittingStrategy): 分裂策略类型
        min_points_for_split (int): 进行分裂的最小点数
        min_points_after_split (int): 分裂后每个子球的最小点数
        density_threshold (float): 密度阈值
    """
    strategy: SplittingStrategy = SplittingStrategy.DISTANCE_BASED
    min_points_for_split: int = 8
    min_points_after_split: int = 4
    density_threshold: float = 0.5


def split_ball_by_distance(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于距离将数据点分裂为两个子集。
    
    该方法找到距离最远的两个点作为分裂中心，
    然后将其他点根据到这两个中心的距离分配到两个子集。
    
    分裂算法步骤：
    1. 计算所有点对之间的距离矩阵
    2. 找到距离最远的两个点（p1, p2）
    3. 对于每个其他点，计算到p1和p2的距离
    4. 将点分配到距离较近的中心对应的子集
    5. 确保两个子集都不为空
    
    Args:
        points: 数据点数组，形状为 [n_points, n_features]
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 两个子集的数据点
        
    Example:
        >>> points = np.random.randn(100, 5)
        >>> ball1, ball2 = split_ball_by_distance(points)
        >>> ball1.shape  # (n1, 5)
        >>> ball2.shape  # (n2, 5)
    """
    num_points, num_features = points.shape
    
    if num_points < 2:
        return points, np.array([])
    
    transposed_points = points.T
    gram_matrix = np.dot(transposed_points.T, transposed_points)
    diag_gram = np.diag(gram_matrix)
    h_matrix = np.tile(diag_gram, (num_points, 1))
    
    distance_matrix_sq = np.maximum(0, h_matrix + h_matrix.T - gram_matrix * 2)
    distance_matrix = np.sqrt(distance_matrix_sq)
    
    if np.max(distance_matrix) == 0:
        mid_idx = num_points // 2
        return points[:mid_idx], points[mid_idx:]
    
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
        return points[:mid_idx], points[mid_idx:]
    
    ball1_points: List[np.ndarray] = []
    ball2_points: List[np.ndarray] = []
    
    for j in range(num_points):
        dist_to_p1 = distance_matrix[j, point1_idx]
        dist_to_p2 = distance_matrix[j, point2_idx]
        if dist_to_p1 <= dist_to_p2:
            ball1_points.append(points[j, :])
        else:
            ball2_points.append(points[j, :])
    
    if not ball1_points:
        ball1_points.append(points[point2_idx])
        ball2_points = [p for i, p in enumerate(ball2_points) 
                       if not np.array_equal(p, points[point2_idx])]
    elif not ball2_points:
        ball2_points.append(points[point1_idx])
        ball1_points = [p for i, p in enumerate(ball1_points) 
                       if not np.array_equal(p, points[point1_idx])]
    
    return np.array(ball1_points), np.array(ball2_points)


def split_ball_by_kmeans(points: np.ndarray, n_clusters: int = 2) -> List[np.ndarray]:
    """
    基于K-Means聚类将数据点分裂为多个子集。
    
    该方法使用K-Means算法将数据点聚类为指定数量的簇，
    每个簇作为一个子球的数据点。
    
    Args:
        points: 数据点数组，形状为 [n_points, n_features]
        n_clusters: 分裂的子集数量，默认为2
        
    Returns:
        List[np.ndarray]: 分裂后的数据子集列表
    """
    from sklearn.cluster import KMeans
    
    num_points = len(points)
    
    if num_points < n_clusters:
        n_clusters = num_points
    
    if n_clusters < 2:
        return [points]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(points)
    
    clusters = [points[labels == i] for i in range(n_clusters)]
    
    return clusters


def split_ball_random(points: np.ndarray, n_splits: int = 2) -> List[np.ndarray]:
    """
    随机将数据点分裂为多个子集。
    
    该方法随机打乱数据点，然后将其均匀分配到指定数量的子集中。
    
    Args:
        points: 数据点数组，形状为 [n_points, n_features]
        n_splits: 分裂的子集数量，默认为2
        
    Returns:
        List[np.ndarray]: 分裂后的数据子集列表
    """
    num_points = len(points)
    
    if num_points < n_splits:
        n_splits = num_points
    
    if n_splits < 2:
        return [points]
    
    indices = np.random.permutation(num_points)
    
    split_size = num_points // n_splits
    splits = []
    
    for i in range(n_splits):
        start_idx = i * split_size
        if i == n_splits - 1:
            end_idx = num_points
        else:
            end_idx = (i + 1) * split_size
        
        splits.append(points[indices[start_idx:end_idx]])
    
    return splits


def calculate_split_quality(
    parent_points: np.ndarray, 
    child1_points: np.ndarray, 
    child2_points: np.ndarray
) -> float:
    """
    计算分裂质量分数。
    
    分裂质量基于以下因素评估：
    1. 子球密度的加权和与父球密度的比较
    2. 两个子球的大小均衡性
    3. 子球内部的紧密度
    
    Args:
        parent_points: 父球的数据点
        child1_points: 第一个子球的数据点
        child2_points: 第二个子球的数据点
        
    Returns:
        float: 分裂质量分数（越高越好）
    """
    if len(parent_points) == 0:
        return 0.0
    
    parent_density = calculate_density(parent_points)
    child1_density = calculate_density(child1_points) if len(child1_points) > 0 else 0
    child2_density = calculate_density(child2_points) if len(child2_points) > 0 else 0
    
    total_points = len(child1_points) + len(child2_points)
    if total_points == 0:
        return 0.0
    
    weight1 = len(child1_points) / total_points
    weight2 = len(child2_points) / total_points
    
    weighted_density = weight1 * child1_density + weight2 * child2_density
    
    density_improvement = weighted_density - parent_density if parent_density > 0 else 0
    
    balance_score = 1 - abs(weight1 - weight2)
    
    quality = density_improvement * 0.7 + balance_score * 0.3
    
    return max(0, quality)


def split_based_on_strategy(
    points: np.ndarray, 
    config: SplittingConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据指定策略将数据点分裂为两个子集。
    
    Args:
        points: 数据点数组
        config: 分裂策略配置
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 两个子集的数据点
        
    Raises:
        ValueError: 如果点数不足或策略不支持
    """
    if len(points) < config.min_points_for_split:
        return points, np.array([])
    
    if config.strategy == SplittingStrategy.DISTANCE_BASED:
        return split_ball_by_distance(points)
    elif config.strategy == SplittingStrategy.KMEANS:
        clusters = split_ball_by_kmeans(points, n_clusters=2)
        if len(clusters) >= 2:
            return clusters[0], clusters[1]
        return points, np.array([])
    elif config.strategy == SplittingStrategy.RANDOM:
        splits = split_ball_random(points, n_splits=2)
        if len(splits) >= 2:
            return splits[0], splits[1]
        return points, np.array([])
    else:
        raise ValueError(f"不支持的分裂策略: {config.strategy}")


def split_ball_collection(
    ball_list: List[np.ndarray], 
    config: SplittingConfig
) -> List[np.ndarray]:
    """
    对粒度球列表应用分裂策略。
    
    该方法遍历所有球体，对每个满足条件的球体应用分裂策略，
    生成新的球体列表。
    
    Args:
        ball_list: 粒度球数据点列表
        config: 分裂策略配置
        
    Returns:
        List[np.ndarray]: 分裂后的新球体列表
        
    Example:
        >>> config = SplittingConfig(
        ...     strategy=SplittingStrategy.DISTANCE_BASED,
        ...     min_points_for_split=8,
        ...     min_points_after_split=4
        ... )
        >>> balls = [np.random.randn(100, 5), np.random.randn(80, 5)]
        >>> new_balls = split_ball_collection(balls, config)
    """
    new_ball_list: List[np.ndarray] = []
    
    for ball_points in ball_list:
        if len(ball_points) < config.min_points_for_split:
            new_ball_list.append(ball_points)
            continue
        
        child1, child2 = split_based_on_strategy(ball_points, config)
        
        if len(child1) == 0 or len(child2) == 0:
            new_ball_list.append(ball_points)
            continue
        
        if (len(child1) >= config.min_points_after_split and 
            len(child2) >= config.min_points_after_split):
            new_ball_list.extend([child1, child2])
        else:
            new_ball_list.append(ball_points)
    
    return new_ball_list


def should_split(
    points: np.ndarray, 
    config: SplittingConfig,
    current_density: float = 0.0
) -> bool:
    """
    判断是否应该对数据点应用分裂操作。
    
    考虑因素：
    1. 数据点数量是否足够
    2. 当前密度是否低于阈值
    3. 分裂是否能提高质量
    
    Args:
        points: 数据点数组
        config: 分裂策略配置
        current_density: 当前密度，默认为0
        
    Returns:
        bool: 是否应该分裂
    """
    if len(points) < config.min_points_for_split:
        return False
    
    if current_density > 0 and current_density > config.density_threshold:
        return False
    
    return True
