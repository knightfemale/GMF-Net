"""
黑翅鸢优化算法（BKOA）实现

该模块实现了黑翅鸢优化算法（Black-winged Kite Optimization Algorithm, BKOA），
这是一种基于黑翅鸢觅食行为的元启发式优化算法。

黑翅鸢优化算法原理：
1. 黑翅鸢在空中盘旋搜索猎物
2. 发现猎物后快速俯冲捕捉
3. 个体之间存在信息共享和行为协调

算法特点：
- 探索能力强：能够在搜索空间中进行广泛探索
- 开发能力好：能够精细开发有前景的区域
- 参数少：只需要设置种群大小和迭代次数
- 适应性强：适用于连续优化问题
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import time

from .base import BaseOptimizer, OptimizerConfig


@dataclass
class BKOAConfig(OptimizerConfig):
    """
    黑翅鸢优化算法配置类。
    
    在基础OptimizerConfig基础上添加了BKOA特定的配置参数。
    
    Attributes:
        alpha (float): 探索系数，控制全局搜索范围
        beta (float): 开发系数，控制局部开发范围
        gamma (float): 扰动系数，控制随机扰动范围
        active_ratio (float): 活跃个体比例
    """
    alpha: float = 0.8
    beta: float = 0.5
    gamma: float = 0.1
    active_ratio: float = 0.5
    
    def __post_init__(self) -> None:
        """
        初始化后验证参数。
        """
        super().__post_init__()
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha 必须在[0,1]范围内，当前值: {self.alpha}")
        if not 0 <= self.beta <= 1:
            raise ValueError(f"beta 必须在[0,1]范围内，当前值: {self.beta}")
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma 必须在[0,1]范围内，当前值: {self.gamma}")
        if not 0 < self.active_ratio <= 1:
            raise ValueError(f"active_ratio 必须在(0,1]范围内，当前值: {self.active_ratio}")


class BKOAOptimizer(BaseOptimizer):
    """
    黑翅鸢优化算法（BKOA）类。
    
    该类实现了黑翅鸢优化算法，用于求解连续优化问题。
    
    算法流程：
    1. 初始化种群：在搜索空间中随机生成初始解
    2. 评估适应度：计算每个个体的目标函数值
    3. 更新位置：根据当前最优解和随机扰动更新位置
    4. 选择更新：如果新位置更优则接受
    5. 重复步骤3-4直到满足终止条件
    
    主要参数：
    - alpha：控制向最优解移动的步长
    - beta：控制向随机个体移动的步长
    - gamma：控制随机扰动的范围
    
    Attributes:
        config (BKOAConfig): BKOA配置对象
        global_best_position (np.ndarray): 全局最优位置
        global_best_fitness (float): 全局最优适应度
    """
    
    def __init__(
        self, 
        objective_function: Callable,
        bounds: List[Tuple[float, float]],
        config: Optional[BKOAConfig] = None,
        **kwargs
    ) -> None:
        """
        初始化黑翅鸢优化算法。
        
        Args:
            objective_function: 目标函数
            bounds: 参数边界列表
            config: BKOA配置对象
            **kwargs: 额外参数
        """
        self.bkoa_config = config if config is not None else BKOAConfig()
        super().__init__(objective_function, bounds, self.bkoa_config, **kwargs)
        
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_fitness = float('inf')
    
    def _initialize_population(self) -> np.ndarray:
        """
        初始化BKOA种群。
        
        采用两阶段初始化策略：
        1. 随机初始化一小部分个体
        2. 基于初始最优解引导其余个体的初始化
        
        Returns:
            np.ndarray: 初始化的种群，形状为 [pop_size, dimension]
        """
        pop_size = self.config.pop_size
        dimension = self.dimension
        
        population = np.random.uniform(
            self.bounds[:, 0], 
            self.bounds[:, 1], 
            size=(pop_size, dimension)
        )
        
        return population
    
    def _update_population(
        self, 
        population: np.ndarray, 
        fitness: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新BKOA种群。
        
        更新策略分为两部分：
        1. 活跃个体：向最优解和随机个体移动
        2. 非活跃个体：在最优解附近进行探索
        
        Args:
            population: 当前种群
            fitness: 适应度值
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 更新后的种群和适应度值
        """
        pop_size = self.config.pop_size
        dimension = self.dimension
        
        current_best_idx = np.argmin(fitness)
        current_best_pos = population[current_best_idx].copy()
        
        if fitness[current_best_idx] < self.global_best_fitness:
            self.global_best_fitness = fitness[current_best_idx]
            self.global_best_position = current_best_pos.copy()
        
        active_size = max(3, int(pop_size * self.bkoa_config.active_ratio))
        
        for i in range(1, active_size):
            other_indices = np.delete(np.arange(active_size), i)
            if len(other_indices) == 0:
                continue
            
            rand_idx = np.random.choice(other_indices)
            rand_pos = population[rand_idx]
            
            r1 = np.random.rand()
            r2 = np.random.rand()
            
            term1 = self.bkoa_config.alpha * r1 * (current_best_pos - population[i])
            term2 = self.bkoa_config.beta * r2 * (rand_pos - population[i])
            term3 = self.bkoa_config.gamma * (np.random.rand(dimension) - 0.5) * (
                self.bounds[:, 1] - self.bounds[:, 0]
            )
            
            new_position = population[i] + term1 + term2 + term3
            new_position = self._ensure_bounds(new_position)
            new_fitness = self._evaluate_fitness(new_position)
            
            if new_fitness < fitness[i]:
                population[i] = new_position
                fitness[i] = new_fitness
        
        bound_range = self.bounds[:, 1] - self.bounds[:, 0]
        
        for i in range(active_size, pop_size):
            noise_scale = 0.2 * bound_range
            noise = np.random.normal(0, noise_scale, dimension)
            new_position = self._ensure_bounds(current_best_pos + noise)
            
            population[i] = new_position
            fitness[i] = float('inf')
        
        fitness = self._evaluate_population(population, fitness)
        
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < self.global_best_fitness:
            self.global_best_fitness = fitness[current_best_idx]
            self.global_best_position = population[current_best_idx].copy()
        
        return population, fitness
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        执行优化算法（run方法的别名）。
        
        Returns:
            Tuple[np.ndarray, float]: 最优位置和最优适应度值
        """
        return self.run()
    
    def get_best_solution(self) -> Dict[str, Any]:
        """
        获取最优解信息。
        
        Returns:
            Dict[str, Any]: 包含最优解信息的字典
        """
        return {
            'position': self.best_position,
            'fitness': self.best_fitness,
            'time': self.get_optimization_time(),
            'iterations': len(self.history)
        }
    
    def get_convergence_curve(self) -> Tuple[List[int], List[float]]:
        """
        获取收敛曲线数据。
        
        Returns:
            Tuple[List[int], List[float]]: (迭代次数列表, 最优适应度列表)
        """
        iterations = [h['iteration'] for h in self.history]
        fitnesses = [h['best_fitness'] for h in self.history]
        return iterations, fitnesses


class DiscreteBKOAOptimizer(BKOAOptimizer):
    """
    离散黑翅鸢优化算法类。
    
    该类专门用于处理离散优化问题，
    在BKOA基础上添加了离散化处理逻辑。
    
    适用于：
    - 聚类数量优化（整数参数）
    - 离散参数选择
    """
    
    def __init__(
        self, 
        objective_function: Callable,
        bounds: List[Tuple[float, float]],
        discrete_indices: List[int] = None,
        config: Optional[BKOAConfig] = None,
        **kwargs
    ) -> None:
        """
        初始化离散BKOA优化器。
        
        Args:
            objective_function: 目标函数
            bounds: 参数边界列表
            discrete_indices: 需要离散化的参数索引列表
            config: BKOA配置对象
            **kwargs: 额外参数
        """
        super().__init__(objective_function, bounds, config, **kwargs)
        
        self.discrete_indices = discrete_indices if discrete_indices is not None else [0]
    
    def _ensure_bounds(self, position: np.ndarray) -> np.ndarray:
        """
        确保位置在边界范围内，并对离散参数进行取整。
        
        Args:
            position: 参数位置
            
        Returns:
            np.ndarray: 处理后的位置
        """
        position = np.clip(position, self.bounds[:, 0], self.bounds[:, 1])
        
        for idx in self.discrete_indices:
            position[idx] = max(self.bounds[idx, 0], round(position[idx]))
        
        return position
    
    def optimize_with_discretization(
        self, 
        discrete_ranges: List[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        执行优化并对离散参数进行后处理。
        
        Args:
            discrete_ranges: 离散参数的取值范围
            
        Returns:
            Tuple[np.ndarray, float]: 离散化后的最优位置和适应度值
        """
        best_pos, best_fit = self.run()
        
        if discrete_ranges is not None:
            for idx, (low, high) in enumerate(discrete_ranges):
                if idx < len(best_pos):
                    step = 0.1
                    discrete_values = np.arange(low, high + step, step)
                    discrete_pos = best_pos.copy()
                    discrete_pos[idx] = min(discrete_values, key=lambda x: abs(x - best_pos[idx]))
                    
                    discrete_fit = self._evaluate_fitness(discrete_pos)
                    if discrete_fit < best_fit:
                        best_pos = discrete_pos
                        best_fit = discrete_fit
        
        return best_pos, best_fit


class AdaptiveBKOAOptimizer(BKOAOptimizer):
    """
    自适应黑翅鸢优化算法类。
    
    该类实现了自适应参数调整策略，
    在优化过程中动态调整alpha、beta、gamma参数。
    
    自适应策略：
    - 前期：强调探索（高alpha）
    - 后期：强调开发（高beta）
    - 添加噪声扰动避免早熟收敛
    """
    
    def __init__(
        self, 
        objective_function: Callable,
        bounds: List[Tuple[float, float]],
        config: Optional[BKOAConfig] = None,
        **kwargs
    ) -> None:
        super().__init__(objective_function, bounds, config, **kwargs)
        self.initial_alpha = self.bkoa_config.alpha
        self.initial_beta = self.bkoa_config.beta
    
    def _update_population(
        self, 
        population: np.ndarray, 
        fitness: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用自适应参数更新种群。
        
        Args:
            population: 当前种群
            fitness: 适应度值
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 更新后的种群和适应度值
        """
        pop_size = self.config.pop_size
        dimension = self.dimension
        max_iter = self.config.max_iter
        current_iter = len(self.history)
        
        progress = current_iter / max_iter if max_iter > 0 else 1.0
        
        self.bkoa_config.alpha = self.initial_alpha * (1 - progress)
        self.bkoa_config.beta = self.initial_beta * progress
        
        return super()._update_population(population, fitness)
