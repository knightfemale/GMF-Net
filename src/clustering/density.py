"""
密度计算模块

该模块提供了粒度球相关的密度计算和归一化功能。

密度是粒度球聚类算法中的重要概念：
- 高密度球体表示数据点紧凑集中
- 低密度球体表示数据点稀疏分散
- 通过分裂低密度球体可以提高整体聚类质量
"""

from typing import List, Tuple, Optional, Union
import numpy as np
import numpy.linalg as la


def calculate_radius(points: np.ndarray) -> float:
    """
    计算数据点的球体半径。
    
    半径定义为数据点集的中心到所有点的最大欧氏距离。
    
    Args:
        points: 数据点数组，形状为 [n_points, n_features]
        
    Returns:
        float: 球体半径
        
    Example:
        >>> points = np.array([[0, 0], [1, 1], [2, 2]])
        >>> radius = calculate_radius(points)
        >>> radius  # 大约为 2.828 (sqrt(8))
    """
    num_points = len(points)
    
    if num_points <= 1:
        return 0.0
    
    center = points.mean(0)
    distances = la.norm(points - center, axis=1)
    radius = np.max(distances)
    
    return radius


def calculate_density(points: np.ndarray) -> float:
    """
    计算数据点的密度。
    
    密度定义为数据点数量与到中心平均距离的比值。
    密度越高表示数据点越紧凑。
    
    密度计算公式：
        density = n_points / sum(distances)
    
    特殊情况：
    - 点数<=1时，返回点数本身
    - 所有距离为0时（所有点重合），返回无穷大
    
    Args:
        points: 数据点数组，形状为 [n_points, n_features]
        
    Returns:
        float: 数据点密度
        
    Example:
        >>> points = np.random.randn(100, 5)
        >>> density = calculate_density(points)
    """
    num_points = len(points)
    
    if num_points <= 1:
        return float(num_points)
    
    center = points.mean(0)
    distances = la.norm(points - center, axis=1)
    sum_radius = np.sum(distances)
    
    if sum_radius > 1e-9:
        density_volume = num_points / sum_radius
    else:
        density_volume = float('inf') if num_points > 0 else 0.0
    
    return density_volume


def calculate_centroid(points: np.ndarray) -> np.ndarray:
    """
    计算数据点的质心（中心）。
    
    质心是所有数据点的算术平均值。
    
    Args:
        points: 数据点数组，形状为 [n_points, n_features]
        
    Returns:
        np.ndarray: 质心坐标，形状为 [n_features]
        
    Example:
        >>> points = np.array([[1, 2], [3, 4], [5, 6]])
        >>> centroid = calculate_centroid(points)
        >>> centroid  # array([3., 4.])
    """
    if len(points) == 0:
        return np.array([])
    
    return np.mean(points, axis=0)


def calculate_variance(points: np.ndarray) -> float:
    """
    计算数据点的方差。
    
    方差表示数据点的离散程度。
    
    Args:
        points: 数据点数组，形状为 [n_points, n_features]
        
    Returns:
        float: 平均方差
    """
    if len(points) == 0:
        return 0.0
    
    centroid = calculate_centroid(points)
    variances = np.var(points - centroid, axis=0)
    return np.mean(variance for variance in variances)


def calculate_std(points: np.ndarray) -> float:
    """
    计算数据点的标准差。
    
    标准差是方差的平方根，表示数据点的离散程度。
    
    Args:
        points: 数据点数组
        
    Returns:
        float: 平均标准差
    """
    return np.sqrt(calculate_variance(points))


def calculate_radius_statistics(radii: List[float]) -> dict:
    """
    计算半径的统计信息。
    
    Args:
        radii: 半径列表
        
    Returns:
        dict: 包含各种统计信息的字典
    """
    if not radii:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }
    
    return {
        'mean': np.mean(radii),
        'median': np.median(radii),
        'std': np.std(radii),
        'min': np.min(radii),
        'max': np.max(radii)
    }


def calculate_density_statistics(densities: List[float]) -> dict:
    """
    计算密度的统计信息。
    
    Args:
        densities: 密度列表
        
    Returns:
        dict: 包含各种统计信息的字典
    """
    valid_densities = [d for d in densities if d != float('inf')]
    
    if not valid_densities:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'infinite_count': len(densities) - len(valid_densities)
        }
    
    return {
        'mean': np.mean(valid_densities),
        'median': np.median(valid_densities),
        'std': np.std(valid_densities),
        'min': np.min(valid_densities),
        'max': np.max(valid_densities),
        'infinite_count': len(densities) - len(valid_densities)
    }


def normalize_balls_by_radius(
    ball_list: List[np.ndarray], 
    detection_radius: float,
    radius_threshold_factor: float = 2.0,
    min_points_for_normalize: int = 2
) -> List[np.ndarray]:
    """
    根据参考半径归一化粒度球列表。
    
    该方法遍历所有球体，对于半径过大的球体进行分裂，
    以确保所有球体的半径都在合理范围内。
    
    归一化条件：
        radius <= radius_threshold_factor * detection_radius
    
    Args:
        ball_list: 粒度球数据点列表
        detection_radius: 参考半径（检测半径）
        radius_threshold_factor: 半径阈值因子，默认为2.0
        min_points_for_normalize: 归一化的最小点数
        
    Returns:
        List[np.ndarray]: 归一化后的粒度球列表
        
    Example:
        >>> balls = [np.random.randn(100, 5), np.random.randn(80, 5)]
        >>> detection_radius = 0.5
        >>> normalized_balls = normalize_balls_by_radius(balls, detection_radius)
    """
    temp_ball_list: List[np.ndarray] = []
    radius_threshold = radius_threshold_factor * detection_radius
    
    for ball_points in ball_list:
        if len(ball_points) < min_points_for_normalize:
            temp_ball_list.append(ball_points)
            continue
        
        current_radius = calculate_radius(ball_points)
        
        if current_radius <= radius_threshold:
            temp_ball_list.append(ball_points)
        else:
            from .splitting import split_ball_by_distance
            points_child1, points_child2 = split_ball_by_distance(ball_points)
            
            if len(points_child1) > 0:
                temp_ball_list.append(points_child1)
            if len(points_child2) > 0:
                temp_ball_list.append(points_child2)
    
    return temp_ball_list


def normalize_balls_by_radius_iterative(
    ball_list: List[np.ndarray],
    detection_radius: float,
    radius_threshold_factor: float = 2.0,
    min_points_for_normalize: int = 2,
    max_iterations: int = 50
) -> List[np.ndarray]:
    """
    迭代归一化粒度球列表。

    该方法循环调用归一化，直到球体数量不再变化或达到最大迭代次数。

    Args:
        ball_list: 粒度球数据点列表
        detection_radius: 参考半径（检测半径）
        radius_threshold_factor: 半径阈值因子
        min_points_for_normalize: 归一化的最小点数
        max_iterations: 最大迭代次数

    Returns:
        List[np.ndarray]: 归一化后的粒度球列表
    """
    current_ball_list = ball_list.copy()
    
    for _ in range(max_iterations):
        ball_count_before = len(current_ball_list)
        current_ball_list = normalize_balls_by_radius(
            current_ball_list,
            detection_radius,
            radius_threshold_factor,
            min_points_for_normalize
        )
        ball_count_after = len(current_ball_list)
        if ball_count_after == ball_count_before:
            break
    
    return current_ball_list


def normalize_balls_by_density(
    ball_list: List[np.ndarray],
    min_density_threshold: float = 1.0,
    max_iterations: int = 50
) -> List[np.ndarray]:
    """
    根据密度阈值归一化粒度球列表。
    
    该方法通过迭代分裂低密度球体来提高整体聚类质量。
    
    Args:
        ball_list: 粒度球数据点列表
        min_density_threshold: 最小密度阈值
        max_iterations: 最大迭代次数
        
    Returns:
        List[np.ndarray]: 归一化后的粒度球列表
    """
    from .splitting import split_ball_by_distance
    
    current_ball_list = ball_list.copy()
    iteration_count = 0
    
    while iteration_count < max_iterations:
        iteration_count += 1
        ball_count_before = len(current_ball_list)
        
        new_ball_list = []
        
        for ball_points in current_ball_list:
            density = calculate_density(ball_points)
            
            if density >= min_density_threshold:
                new_ball_list.append(ball_points)
            else:
                child1, child2 = split_ball_by_distance(ball_points)
                
                if len(child1) > 0 and len(child2) > 0:
                    new_ball_list.extend([child1, child2])
                else:
                    new_ball_list.append(ball_points)
        
        current_ball_list = new_ball_list
        ball_count_after = len(current_ball_list)
        
        if ball_count_after == ball_count_before:
            break
    
    return current_ball_list


def split_based_on_density(
    ball_list: List[np.ndarray],
    min_points_for_split: int = 8,
    min_points_after_split: int = 4,
    max_balls: int = 1000,
    max_iterations: int = 50
) -> List[np.ndarray]:
    """
    基于密度对粒度球列表进行分裂。

    该方法迭代地对球体进行分裂，如果分裂后子球的加权密度高于父球，
    且子球有足够的点数，则进行分裂。

    Args:
        ball_list: 粒度球数据点列表
        min_points_for_split: 进行分裂的最小点数
        min_points_after_split: 分裂后每个子球的最小点数
        max_balls: 最大球体数量
        max_iterations: 最大迭代次数

    Returns:
        List[np.ndarray]: 分裂后的粒度球列表
    """
    from .splitting import split_ball_by_distance
    
    current_ball_list = ball_list.copy()
    
    for iteration in range(max_iterations):
        if len(current_ball_list) >= max_balls:
            break
            
        new_ball_list: List[np.ndarray] = []
        any_split = False
        
        for ball_points in current_ball_list:
            if len(ball_points) < min_points_for_split:
                new_ball_list.append(ball_points)
                continue
            
            child1_points, child2_points = split_ball_by_distance(ball_points)
            
            if len(child1_points) == 0 or len(child2_points) == 0:
                new_ball_list.append(ball_points)
                continue
            
            if len(child1_points) < min_points_after_split or len(child2_points) < min_points_after_split:
                new_ball_list.append(ball_points)
                continue
            
            parent_density = calculate_density(ball_points)
            child1_density = calculate_density(child1_points)
            child2_density = calculate_density(child2_points)
            
            total_points = len(child1_points) + len(child2_points)
            weight1 = len(child1_points) / total_points if total_points > 0 else 0
            weight2 = len(child2_points) / total_points if total_points > 0 else 0
            
            weighted_child_density = weight1 * child1_density + weight2 * child2_density
            
            density_improves = weighted_child_density > parent_density
            sufficient_points = (len(child1_points) >= min_points_after_split and 
                               len(child2_points) >= min_points_after_split)
            
            if density_improves and sufficient_points:
                new_ball_list.extend([child1_points, child2_points])
                any_split = True
            else:
                new_ball_list.append(ball_points)
        
        current_ball_list = new_ball_list
        
        if not any_split:
            break
    
    return current_ball_list


def calculate_detection_radius(ball_list: List[np.ndarray]) -> float:
    """
    计算用于归一化的检测半径。
    
    检测半径基于所有球体半径的中位数和均值计算，
    取两者中的较大值。
    
    Args:
        ball_list: 粒度球数据点列表
        
    Returns:
        float: 检测半径
    """
    radii = [calculate_radius(ball_points) for ball_points in ball_list 
            if len(ball_points) >= 2]
    
    if not radii:
        return 0.0
    
    radius_median = np.median(radii)
    radius_mean = np.mean(radii)
    detection_radius = max(radius_median, radius_mean, 1e-6)
    
    return detection_radius


def evaluate_clustering_quality(
    ball_list: List[np.ndarray],
    ground_truth: Optional[np.ndarray] = None
) -> dict:
    """
    评估聚类质量。
    
    Args:
        ball_list: 粒度球数据点列表
        ground_truth: 真实标签（可选）
        
    Returns:
        dict: 包含各种质量指标的字典
    """
    if not ball_list:
        return {
            'num_balls': 0,
            'total_points': 0,
            'avg_radius': 0.0,
            'avg_density': 0.0,
            'radius_stats': {},
            'density_stats': {}
        }
    
    radii = [calculate_radius(ball) for ball in ball_list]
    densities = [calculate_density(ball) for ball in ball_list]
    
    total_points = sum(len(ball) for ball in ball_list)
    
    radius_stats = calculate_radius_statistics(radii)
    density_stats = calculate_density_statistics(densities)
    
    return {
        'num_balls': len(ball_list),
        'total_points': total_points,
        'avg_radius': np.mean(radii) if radii else 0.0,
        'avg_density': np.mean([d for d in densities if d != float('inf')]) 
                     if any(d != float('inf') for d in densities) else 0.0,
        'radius_stats': radius_stats,
        'density_stats': density_stats
    }
