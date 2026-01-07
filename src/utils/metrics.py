"""
评估指标模块

该模块提供了各种评估指标的计算函数，
用于评估模型预测和聚类结果的质量。

主要指标包括：
- 回归评估指标：MSE, RMSE, MAE, MAPE, R²
- 聚类评估指标：ARI, NMI
"""

from typing import Tuple, Union
import numpy as np


def calculate_mse(
    predictions: np.ndarray, 
    actuals: np.ndarray
) -> float:
    """
    计算均方误差（MSE）。
    
    MSE是最常用的回归评估指标之一，
    衡量预测值与真实值之间差异的平方的平均值。
    
    公式：MSE = (1/n) * Σ(y_true - y_pred)²
    
    Args:
        predictions: 预测值数组
        actuals: 真实值数组
        
    Returns:
        float: 均方误差值
        
    Example:
        >>> y_pred = np.array([1, 2, 3])
        >>> y_true = np.array([1.1, 1.9, 3.2])
        >>> mse = calculate_mse(y_pred, y_true)
    """
    return np.mean((predictions - actuals) ** 2)


def calculate_rmse(
    predictions: np.ndarray, 
    actuals: np.ndarray
) -> float:
    """
    计算均方根误差（RMSE）。
    
    RMSE是MSE的平方根，与原始数据具有相同的单位，
    更直观地表示预测误差的大小。
    
    公式：RMSE = sqrt(MSE)
    
    Args:
        predictions: 预测值数组
        actuals: 真实值数组
        
    Returns:
        float: 均方根误差值
    """
    return np.sqrt(calculate_mse(predictions, actuals))


def calculate_mae(
    predictions: np.ndarray, 
    actuals: np.ndarray
) -> float:
    """
    计算平均绝对误差（MAE）。
    
    MAE是预测值与真实值之间绝对差异的平均值，
    对异常值不如MSE敏感。
    
    公式：MAE = (1/n) * Σ|y_true - y_pred|
    
    Args:
        predictions: 预测值数组
        actuals: 真实值数组
        
    Returns:
        float: 平均绝对误差值
    """
    return np.mean(np.abs(predictions - actuals))


def calculate_mape(
    predictions: np.ndarray, 
    actuals: np.ndarray
) -> float:
    """
    计算平均绝对百分比误差（MAPE）。
    
    MAPE以百分比形式表示预测误差，
    直观反映预测的相对准确程度。
    
    公式：MAPE = (1/n) * Σ|(y_true - y_pred) / y_true| * 100%
    
    注意：当真实值为0时无法计算MAPE
    
    Args:
        predictions: 预测值数组
        actuals: 真实值数组
        
    Returns:
        float: MAPE值（百分比），无法计算时返回NaN
    """
    mask = actuals != 0
    if np.sum(mask) == 0:
        return np.nan
    
    return np.mean(np.abs((predictions[mask] - actuals[mask]) / actuals[mask])) * 100


def calculate_r2(
    predictions: np.ndarray, 
    actuals: np.ndarray
) -> float:
    """
    计算决定系数R²。
    
    R²表示模型解释目标变量变异的能力，
    值越接近1表示模型拟合越好。
    
    公式：R² = 1 - SS_res / SS_tot
    其中：SS_res = Σ(y_true - y_pred)²
          SS_tot = Σ(y_true - mean(y_true))²
    
    Args:
        predictions: 预测值数组
        actuals: 真实值数组
        
    Returns:
        float: R²值，范围通常为(-∞, 1]
    """
    ss_res = np.sum((predictions - actuals) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    
    if ss_tot == 0:
        return np.nan
    
    return 1 - (ss_res / ss_tot)


def calculate_r2_per_step(
    predictions: np.ndarray, 
    actuals: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    计算多步预测的R²值。
    
    Args:
        predictions: 预测值数组，形状为 [n_samples, n_steps]
        actuals: 真实值数组，形状为 [n_samples, n_steps]
        
    Returns:
        Tuple[float, np.ndarray]: (平均R², 每步R²数组)
    """
    n_steps = actuals.shape[1] if actuals.ndim > 1 else 1
    
    if n_steps == 1:
        return calculate_r2(predictions.flatten(), actuals.flatten()), np.array([])
    
    r2_per_step = []
    for i in range(n_steps):
        r2_step = calculate_r2(predictions[:, i], actuals[:, i])
        r2_per_step.append(r2_step)
    
    avg_r2 = np.nanmean(r2_per_step)
    
    return avg_r2, np.array(r2_per_step)


def calculate_ari(
    predictions: np.ndarray, 
    actuals: np.ndarray
) -> float:
    """
    计算调整兰德指数（ARI）。
    
    ARI是兰德指数的调整版本，校正了随机聚类的期望，
    取值范围为[-1, 1]，越接近1表示聚类效果越好。
    
    Args:
        predictions: 预测聚类标签
        actuals: 真实聚类标签
        
    Returns:
        float: ARI值
    """
    from sklearn import metrics
    
    valid_mask = (predictions != -1) & (actuals != -1)
    
    if np.sum(valid_mask) < 2:
        return 0.0
    
    return metrics.adjusted_rand_score(actuals[valid_mask], predictions[valid_mask])


def calculate_nmi(
    predictions: np.ndarray, 
    actuals: np.ndarray
) -> float:
    """
    计算归一化互信息（NMI）。
    
    NMI衡量两个聚类之间的互信息，
    取值范围为[0, 1]，越接近1表示聚类效果越好。
    
    Args:
        predictions: 预测聚类标签
        actuals: 真实聚类标签
        
    Returns:
        float: NMI值
    """
    from sklearn import metrics
    
    valid_mask = (predictions != -1) & (actuals != -1)
    
    if np.sum(valid_mask) < 2:
        return 0.0
    
    try:
        return metrics.normalized_mutual_info_score(
            actuals[valid_mask], 
            predictions[valid_mask],
            average_method='arithmetic'
        )
    except Exception:
        return 0.0


class EvaluationMetrics:
    """
    评估指标综合计算类。
    
    该类封装了各种评估指标的计算方法，
    可以一次性计算多种指标并返回结构化结果。
    
    Attributes:
        regression_metrics (List[str]): 要计算的回归指标列表
        clustering_metrics (List[str]): 要计算的聚类指标列表
    """
    
    def __init__(
        self, 
        regression_metrics: list = None,
        clustering_metrics: list = None
    ) -> None:
        """
        初始化评估指标计算器。
        
        Args:
            regression_metrics: 要计算的回归指标
            clustering_metrics: 要计算的聚类指标
        """
        if regression_metrics is None:
            regression_metrics = ['mse', 'rmse', 'mae', 'mape', 'r2']
        if clustering_metrics is None:
            clustering_metrics = ['ari', 'nmi']
        
        self.regression_metrics = regression_metrics
        self.clustering_metrics = clustering_metrics
    
    def evaluate_regression(
        self, 
        predictions: np.ndarray, 
        actuals: np.ndarray
    ) -> dict:
        """
        计算回归评估指标。
        
        Args:
            predictions: 预测值数组
            actuals: 真实值数组
            
        Returns:
            dict: 包含各指标的字典
        """
        results = {}
        
        if 'mse' in self.regression_metrics:
            results['mse'] = calculate_mse(predictions, actuals)
        
        if 'rmse' in self.regression_metrics:
            results['rmse'] = calculate_rmse(predictions, actuals)
        
        if 'mae' in self.regression_metrics:
            results['mae'] = calculate_mae(predictions, actuals)
        
        if 'mape' in self.regression_metrics:
            mape = calculate_mape(predictions, actuals)
            results['mape'] = mape if not np.isnan(mape) else None
        
        if 'r2' in self.regression_metrics:
            r2 = calculate_r2(predictions, actuals)
            results['r2'] = r2 if not np.isnan(r2) else None
        
        return results
    
    def evaluate_clustering(
        self, 
        predictions: np.ndarray, 
        actuals: np.ndarray
    ) -> dict:
        """
        计算聚类评估指标。
        
        Args:
            predictions: 预测聚类标签
            actuals: 真实聚类标签
            
        Returns:
            dict: 包含各指标的字典
        """
        results = {}
        
        if 'ari' in self.clustering_metrics:
            results['ari'] = calculate_ari(predictions, actuals)
        
        if 'nmi' in self.clustering_metrics:
            results['nmi'] = calculate_nmi(predictions, actuals)
        
        return results
    
    def evaluate(
        self, 
        predictions: np.ndarray, 
        actuals: np.ndarray,
        task_type: str = 'regression'
    ) -> dict:
        """
        根据任务类型计算评估指标。
        
        Args:
            predictions: 预测值或标签
            actuals: 真实值或标签
            task_type: 任务类型 ('regression' 或 'clustering')
            
        Returns:
            dict: 包含各指标的字典
        """
        if task_type == 'regression':
            return self.evaluate_regression(predictions, actuals)
        elif task_type == 'clustering':
            return self.evaluate_clustering(predictions, actuals)
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")
    
    def print_results(
        self, 
        results: dict, 
        task_type: str = 'regression'
    ) -> None:
        """
        打印评估结果。
        
        Args:
            results: 评估结果字典
            task_type: 任务类型
        """
        print(f"\n{'='*50}")
        print(f"{task_type.upper()} 评估结果")
        print(f"{'='*50}")
        
        for metric, value in results.items():
            if value is not None:
                if metric == 'mape':
                    print(f"{metric.upper()}: {value:.4f}%")
                else:
                    print(f"{metric.upper()}: {value:.4f}")
        
        print(f"{'='*50}\n")
