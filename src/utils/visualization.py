"""
可视化工具模块

该模块提供了各种可视化功能，用于展示模型训练结果、
预测效果和聚类分析结果。

主要功能：
- 绘制预测结果对比图
- 绘制训练损失曲线
- 绘制聚类结果图
- 保存评估结果
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class VisualizationConfig:
    """
    可视化配置类。
    
    Attributes:
        figure_size (Tuple[int, int]): 图形尺寸
        dpi (int): 图形分辨率
        save_format (str): 保存格式 ('png', 'pdf', 'svg')
        style (str): 绘图风格
        color_scheme (str): 颜色方案
    """
    
    def __init__(
        self,
        figure_size: Tuple[int, int] = (12, 6),
        dpi: int = 100,
        save_format: str = 'png',
        style: str = 'default',
        color_scheme: str = 'default'
    ) -> None:
        self.figure_size = figure_size
        self.dpi = dpi
        self.save_format = save_format
        self.style = style
        self.color_scheme = color_scheme


def plot_predictions(
    predictions: np.ndarray,
    actuals: np.ndarray,
    save_path: Optional[str] = None,
    config: Optional[VisualizationConfig] = None,
    title: str = '预测值 vs 实际值',
    step_index: int = 0,
    show_grid: bool = True
) -> Figure:
    """
    绘制预测值与实际值的对比图。
    
    Args:
        predictions: 预测值数组
        actuals: 实际值数组
        save_path: 保存路径（可选）
        config: 可视化配置
        title: 图形标题
        step_index: 预测步索引（用于多步预测）
        show_grid: 是否显示网格
        
    Returns:
        Figure: matplotlib图形对象
        
    Example:
        >>> fig = plot_predictions(predictions, actuals, save_path='results/prediction.png')
    """
    if config is None:
        config = VisualizationConfig()
    
    plt.style.use(config.style)
    
    fig, ax = plt.subplots(figsize=config.figure_size, dpi=config.dpi)
    
    plot_actual = actuals[:, step_index] if actuals.ndim > 1 else actuals
    plot_pred = predictions[:, step_index] if predictions.ndim > 1 else predictions
    
    x_axis = np.arange(len(plot_actual))
    
    ax.plot(x_axis, plot_actual, label='实际值', color='blue', linewidth=1.5, alpha=0.8)
    ax.plot(x_axis, plot_pred, label='预测值', color='red', 
            linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('时间步', fontsize=12)
    ax.set_ylabel('值', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format=config.save_format, dpi=config.dpi)
        print(f"图形已保存到: {save_path}")
    
    return fig


def plot_loss_curve(
    train_losses: List[float],
    test_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    config: Optional[VisualizationConfig] = None,
    title: str = '训练和测试损失曲线',
    show_grid: bool = True
) -> Figure:
    """
    绘制训练损失曲线。
    
    Args:
        train_losses: 训练损失列表
        test_losses: 测试损失列表（可选）
        save_path: 保存路径（可选）
        config: 可视化配置
        title: 图形标题
        show_grid: 是否显示网格
        
    Returns:
        Figure: matplotlib图形对象
    """
    if config is None:
        config = VisualizationConfig()
    
    plt.style.use(config.style)
    
    fig, ax = plt.subplots(figsize=config.figure_size, dpi=config.dpi)
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, label='训练损失', color='blue', linewidth=1.5)
    
    if test_losses is not None:
        ax.plot(epochs, test_losses, label='测试损失', color='green', linewidth=1.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('轮次 (Epoch)', fontsize=12)
    ax.set_ylabel('损失 (MSE)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format=config.save_format, dpi=config.dpi)
        print(f"图形已保存到: {save_path}")
    
    return fig


def plot_clustering_results(
    features: np.ndarray,
    labels: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    config: Optional[VisualizationConfig] = None,
    title: str = '聚类结果可视化',
    show_legend: bool = True
) -> Figure:
    """
    绘制聚类结果可视化图。
    
    Args:
        features: 特征数据（2D，用于可视化）
        labels: 聚类标签
        ground_truth: 真实标签（可选）
        save_path: 保存路径（可选）
        config: 可视化配置
        title: 图形标题
        show_legend: 是否显示图例
        
    Returns:
        Figure: matplotlib图形对象
    """
    if config is None:
        config = VisualizationConfig()
    
    plt.style.use(config.style)
    
    if features.shape[1] > 2:
        from sklearn.decomposition import PCA
        features = PCA(n_components=2).fit_transform(features)
    
    fig, axes = plt.subplots(1, 2 if ground_truth is not None else 1, 
                             figsize=config.figure_size, dpi=config.dpi)
    
    if ground_truth is not None:
        axes[0].scatter(features[:, 0], features[:, 1], c=ground_truth, 
                       cmap='viridis', alpha=0.6, s=30)
        axes[0].set_title('真实标签', fontsize=12)
        
        axes[1].scatter(features[:, 0], features[:, 1], c=labels, 
                       cmap='viridis', alpha=0.6, s=30)
        axes[1].set_title('聚类结果', fontsize=12)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
    else:
        axes.scatter(features[:, 0], features[:, 1], c=labels, 
                    cmap='viridis', alpha=0.6, s=30)
        axes.set_title(title, fontsize=14, fontweight='bold')
    
    for ax in (axes if isinstance(axes, list) else [axes]):
        ax.set_xlabel('特征1', fontsize=10)
        ax.set_ylabel('特征2', fontsize=10)
        if show_legend:
            ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format=config.save_format, dpi=config.dpi)
        print(f"图形已保存到: {save_path}")
    
    return fig


def plot_multi_step_predictions(
    predictions: np.ndarray,
    actuals: np.ndarray,
    save_path: Optional[str] = None,
    config: Optional[VisualizationConfig] = None,
    title: str = '多步预测结果',
    num_steps_to_show: int = 5
) -> Figure:
    """
    绘制多步预测结果对比图。
    
    Args:
        predictions: 预测值数组，形状为 [n_samples, n_steps]
        actuals: 实际值数组，形状为 [n_samples, n_steps]
        save_path: 保存路径（可选）
        config: 可视化配置
        title: 图形标题
        num_steps_to_show: 显示的预测步数
        
    Returns:
        Figure: matplotlib图形对象
    """
    if config is None:
        config = VisualizationConfig()
    
    plt.style.use(config.style)
    
    n_steps = min(predictions.shape[1], num_steps_to_show)
    
    fig, axes = plt.subplots(n_steps, 1, figsize=(config.figure_size[0], 
                                                  config.figure_size[1] * n_steps),
                            dpi=config.dpi)
    
    if n_steps == 1:
        axes = [axes]
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for step in range(n_steps):
        ax = axes[step]
        
        x_axis = np.arange(len(predictions))
        
        ax.plot(x_axis, actuals[:, step], label=f'实际值 (步{step+1})', 
                color='blue', linewidth=1.5, alpha=0.8)
        ax.plot(x_axis, predictions[:, step], label=f'预测值 (步{step+1})', 
                color=colors[step % len(colors)], linestyle='--', linewidth=1.5)
        
        ax.set_title(f'预测步 {step + 1}', fontsize=12)
        ax.set_xlabel('样本', fontsize=10)
        ax.set_ylabel('值', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format=config.save_format, dpi=config.dpi)
        print(f"图形已保存到: {save_path}")
    
    return fig


def plot_convergence_curve(
    iterations: List[int],
    fitness_values: List[float],
    save_path: Optional[str] = None,
    config: Optional[VisualizationConfig] = None,
    title: str = '优化收敛曲线'
) -> Figure:
    """
    绘制优化算法收敛曲线。
    
    Args:
        iterations: 迭代次数列表
        fitness_values: 适应度值列表
        save_path: 保存路径（可选）
        config: 可视化配置
        title: 图形标题
        
    Returns:
        Figure: matplotlib图形对象
    """
    if config is None:
        config = VisualizationConfig()
    
    plt.style.use(config.style)
    
    fig, ax = plt.subplots(figsize=config.figure_size, dpi=config.dpi)
    
    ax.plot(iterations, fitness_values, color='blue', linewidth=1.5)
    
    ax.fill_between(iterations, fitness_values, alpha=0.3)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('最优适应度', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    min_idx = np.argmin(fitness_values)
    ax.scatter([iterations[min_idx]], [fitness_values[min_idx]], 
              color='red', s=100, zorder=5, label=f'最优: {fitness_values[min_idx]:.4f}')
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format=config.save_format, dpi=config.dpi)
        print(f"图形已保存到: {save_path}")
    
    return fig


def save_evaluation_results(
    results: Dict[str, float],
    save_path: str,
    metrics_order: List[str] = None
) -> None:
    """
    将评估结果保存到文本文件。
    
    Args:
        results: 评估结果字典
        save_path: 保存路径
        metrics_order: 指标顺序列表
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("评估结果\n")
        f.write("=" * 50 + "\n\n")
        
        if metrics_order is None:
            metrics_order = list(results.keys())
        
        for metric in metrics_order:
            if metric in results:
                value = results[metric]
                if metric == 'mape':
                    f.write(f"{metric.upper()}: {value:.4f}%\n")
                else:
                    f.write(f"{metric.upper()}: {value:.4f}\n")
        
        f.write("\n" + "=" * 50 + "\n")
    
    print(f"评估结果已保存到: {save_path}")


def create_results_directory(base_path: str = "results") -> str:
    """
    创建结果保存目录。
    
    Args:
        base_path: 基础路径
        
    Returns:
        str: 创建的目录路径
    """
    timestamp = np.datetime64('now').astype('datetime64[D]').astype(str).replace('-', '')
    results_dir = os.path.join(base_path, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    subdirs = ['predictions', 'loss_curves', 'clustering', 'convergence']
    for subdir in subdirs:
        os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)
    
    return results_dir


def show_plots() -> None:
    """
    显示所有待显示的图形。
    
    在需要时调用此函数来显示所有图形。
    """
    plt.show()
