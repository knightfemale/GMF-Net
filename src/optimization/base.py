"""
基础优化器抽象类

该模块定义了优化器的抽象基类，提供了通用的优化接口。
所有具体的优化器都应该继承此类并实现必要的方法。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import time
from tqdm import tqdm


@dataclass
class OptimizerConfig:
    """
    优化器配置类，用于存储优化器的通用参数。

    Attributes:
        pop_size (int): 种群大小
        max_iter (int): 最大迭代次数
        timeout (float): 超时时间（秒）
        random_seed (int): 随机种子
        verbose (bool): 是否打印详细信息
        early_stopping (bool): 是否启用早停
        early_stopping_threshold (float): 早停阈值
        early_stopping_patience (int): 早停耐心值
    """

    pop_size: int = 20
    max_iter: int = 50
    timeout: float = 300.0
    random_seed: int = 42
    verbose: bool = True
    early_stopping: bool = True
    early_stopping_threshold: float = 1e-6
    early_stopping_patience: int = 5

    def __post_init__(self) -> None:
        """
        初始化后验证参数。
        """
        if self.pop_size < 1:
            raise ValueError(f"pop_size 必须大于0，当前值: {self.pop_size}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter 必须大于0，当前值: {self.max_iter}")
        if self.timeout < 0:
            raise ValueError(f"timeout 必须大于等于0，当前值: {self.timeout}")
        if not 0 <= self.early_stopping_threshold < 1:
            raise ValueError(f"early_stopping_threshold 必须在[0,1)范围内，当前值: {self.early_stopping_threshold}")
        if self.early_stopping_patience < 1:
            raise ValueError(f"early_stopping_patience 必须大于0，当前值: {self.early_stopping_patience}")


class BaseOptimizer(ABC):
    """
    基础优化器抽象类。

    该类定义了优化器的通用接口，所有具体的优化器都需要继承此类。
    提供了优化器初始化、运行、结果获取等功能。

    Attributes:
        config (OptimizerConfig): 优化器配置对象
        objective_function (Callable): 目标函数
        bounds (np.ndarray): 参数边界，形状为 [n_params, 2]
        dimension (int): 参数维度
        best_position (Optional[np.ndarray]): 最优位置
        best_fitness (float): 最优适应度值
        history (List[Dict]): 优化历史记录
    """

    def __init__(self, objective_function: Callable, bounds: List[Tuple[float, float]], config: Optional[OptimizerConfig] = None, **kwargs) -> None:
        """
        初始化基础优化器。

        Args:
            objective_function: 目标函数，输入为参数向量，输出为适应度值
            bounds: 参数边界列表，每个元素为 (下界, 上界)
            config: 优化器配置对象
            **kwargs: 额外参数
        """
        self.config = config if config is not None else OptimizerConfig()
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.dimension = len(bounds)

        self.best_position: Optional[np.ndarray] = None
        self.best_fitness = float("inf")

        self.history: List[Dict] = []
        self.start_time: float = 0.0
        self.end_time: float = 0.0

        self._setup_random_seed()

    def _setup_random_seed(self) -> None:
        """
        设置随机种子以确保可重复性。
        """
        np.random.seed(self.config.random_seed)

    @abstractmethod
    def _initialize_population(self) -> np.ndarray:
        """
        初始化种群（由子类实现）。

        Returns:
            np.ndarray: 初始化的种群，形状为 [pop_size, dimension]
        """
        pass

    @abstractmethod
    def _update_population(self, population: np.ndarray, fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        更新种群（由子类实现）。

        Args:
            population: 当前种群
            fitness: 适应度值

        Returns:
            Tuple[np.ndarray, np.ndarray]: 更新后的种群和适应度值
        """
        pass

    def _ensure_bounds(self, position: np.ndarray) -> np.ndarray:
        """
        确保位置在边界范围内。

        Args:
            position: 参数位置

        Returns:
            np.ndarray: 边界内的位置
        """
        position = np.clip(position, self.bounds[:, 0], self.bounds[:, 1])
        return position

    def _evaluate_fitness(self, position: np.ndarray) -> float:
        """
        计算适应度值。

        Args:
            position: 参数位置

        Returns:
            float: 适应度值
        """
        return self.objective_function(position)

    def run(self) -> Tuple[np.ndarray, float]:
        """
        运行优化算法。

        执行以下步骤：
        1. 初始化种群
        2. 评估初始适应度
        3. 迭代更新种群
        4. 检查早停条件
        5. 返回最优解

        Returns:
            Tuple[np.ndarray, float]: 最优位置和最优适应度值
        """
        self.start_time = time.time()

        population = self._initialize_population()
        fitness = np.full(self.config.pop_size, float("inf"))

        fitness = self._evaluate_population(population, fitness)

        self.best_position = population[np.argmin(fitness)].copy()
        self.best_fitness = float(np.min(fitness))

        self._record_history(0, self.best_fitness)

        early_stop_count = 0
        previous_best_fitness = self.best_fitness

        if self.config.verbose:
            progress_bar = tqdm(range(1, self.config.max_iter + 1), desc="BKOA迭代", ncols=80, unit="次", unit_scale=False)
        else:
            progress_bar = range(1, self.config.max_iter + 1)

        for iteration in progress_bar:

            if self._check_timeout():
                if self.config.verbose:
                    print(f"\n优化超时 ({self.config.timeout}s)，提前终止")
                break

            population, fitness = self._update_population(population, fitness)

            current_best_idx = np.argmin(fitness)
            current_best_fitness = float(fitness[current_best_idx])

            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_position = population[current_best_idx].copy()

            self._record_history(iteration, self.best_fitness)

            if self.config.early_stopping:
                fitness_change = abs(previous_best_fitness - self.best_fitness)

                if fitness_change < self.config.early_stopping_threshold:
                    early_stop_count += 1
                    if early_stop_count >= self.config.early_stopping_patience:
                        if self.config.verbose:
                            progress_bar.set_postfix({"状态": "收敛", "最优适应度": f"{self.best_fitness:.6f}"})

                        break
                else:
                    early_stop_count = 0

            previous_best_fitness = self.best_fitness

            if self.config.verbose:
                progress_bar.set_postfix({"最优": f"{self.best_fitness:.6f}"})
            elif self.config.verbose:
                print(f"迭代 {iteration}/{self.config.max_iter}, " f"最优适应度: {self.best_fitness:.6f}    ", end="\r")

        self.end_time = time.time()

        if self.config.verbose:
            total_time = self.end_time - self.start_time
            print(f"\n优化完成，总时间: {total_time:.2f}秒")
            print(f"最优适应度: {self.best_fitness:.6f}")

        return self._ensure_bounds(self.best_position), self.best_fitness

    def _evaluate_population(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        评估整个种群的适应度。

        Args:
            population: 种群
            fitness: 适应度数组

        Returns:
            np.ndarray: 更新后的适应度数组
        """
        indices_to_eval = [i for i in range(len(population)) if fitness[i] == float("inf")]

        if self.config.verbose and len(indices_to_eval) > 1:
            for i in tqdm(indices_to_eval, desc="评估适应度", ncols=60, unit="个", leave=False):
                fitness[i] = self._evaluate_fitness(population[i])
        else:
            for i in indices_to_eval:
                fitness[i] = self._evaluate_fitness(population[i])
        return fitness

    def _check_timeout(self) -> bool:
        """
        检查是否超时。

        Returns:
            bool: 是否超时
        """
        return (time.time() - self.start_time) > self.config.timeout

    def _record_history(self, iteration: int, best_fitness: float) -> None:
        """
        记录优化历史。

        Args:
            iteration: 当前迭代次数
            best_fitness: 当前最优适应度
        """
        self.history.append({"iteration": iteration, "best_fitness": best_fitness, "time": time.time() - self.start_time})

    def get_history(self) -> List[Dict]:
        """
        获取优化历史记录。

        Returns:
            List[Dict]: 优化历史记录列表
        """
        return self.history

    def get_optimization_time(self) -> float:
        """
        获取优化执行时间。

        Returns:
            float: 优化时间（秒）
        """
        return self.end_time - self.start_time

    def get_result(self) -> Dict[str, Any]:
        """
        获取优化结果字典。

        Returns:
            Dict[str, Any]: 包含优化结果的字典
        """
        return {"best_position": self.best_position, "best_fitness": self.best_fitness, "history": self.history, "total_time": self.get_optimization_time(), "iterations": len(self.history)}
