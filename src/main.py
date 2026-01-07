#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GMF-Net 主程序入口

该模块提供了项目的统一入口点，展示了如何使用重构后的各个模块。
可以通过命令行参数选择运行不同的任务：
1. 时间序列预测 (--task=prediction)
2. 粒度球聚类 (--task=clustering)
3. 超参数优化 (--task=optimization)
4. 完整流程 (--task=all)

使用方法：
    python -m src.main --task=prediction --data_path=data/train.csv
    python -m src.main --task=clustering --data_path=data/dataset.mat
    python -m src.main --task=optimization --data_path=data/dataset.mat
    python -m src.main --task=all --data_path=data/dataset.mat

主要功能：
- 统一的任务调度和配置管理
- 支持多种输入数据格式（CSV、MAT）
- 提供完整的模型训练、评估和可视化流程
- 支持留一用户交叉验证
- 支持BKOA超参数优化
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "src"

from .data import (
    TimeSeriesDataLoader,
    TimeSeriesConfig,
    DatasetManager,
)
from .models import (
    TCNDCATransformer,
    FusionModelConfig,
    TemporalConvNet,
    TCNConfig,
    ChannelAttention,
    ChannelAttentionConfig,
    TransformerBlock,
    TransformerConfig,
)
from .clustering import (
    GranularBall,
    GranularBallCollection,
    split_ball_by_distance,
    SplittingStrategy,
    calculate_density,
    calculate_radius,
    split_based_on_density,
    calculate_detection_radius,
    normalize_balls_by_radius,
    normalize_balls_by_radius_iterative,
)
from .optimization import (
    BaseOptimizer,
    OptimizerConfig,
    BKOAOptimizer,
    BKOAConfig,
    ObjectiveFunction,
    ClusteringObjective,
)
from .optimization.objective import create_clustering_objective
from .utils import (
    calculate_mse,
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_r2,
    calculate_ari,
    calculate_nmi,
    EvaluationMetrics,
    plot_predictions,
    plot_loss_curve,
    plot_clustering_results,
    save_evaluation_results,
)
from .config import Config, get_default_config


@dataclass
class TaskConfig:
    """
    任务配置类，用于存储各个任务的配置参数。

    Attributes:
        task (str): 任务类型
        data_path (str): 数据文件路径
        results_dir (str): 结果保存目录
        device (str): 计算设备
        random_seed (int): 随机种子
    """

    task: str = "prediction"
    data_path: str = ""
    results_dir: str = "results"
    device: str = "cuda"
    random_seed: int = 42


def set_random_seed(seed: int) -> None:
    """
    设置随机种子以确保实验可重复性。

    Args:
        seed: 随机种子值
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子已设置为: {seed}")


def load_data_for_prediction(data_path: str, features: List[str], target: str, window_size: int = 300, horizon: int = 50, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, Any]:
    """
    加载时间序列预测数据。

    Args:
        data_path: CSV文件路径
        features: 特征列名列表
        target: 目标列名
        window_size: 滑动窗口大小
        horizon: 预测范围
        batch_size: 批次大小

    Returns:
        Tuple[DataLoader, DataLoader, Any]: 训练加载器、测试加载器、缩放器
    """
    config = TimeSeriesConfig(file_path=data_path, features=features, target=target, window_size=window_size, horizon=horizon, batch_size=batch_size)

    loader = TimeSeriesDataLoader(config)
    train_loader, test_loader, scaler_y, scaler_X = loader.load_data()

    print(f"数据加载完成:")
    print(f"  - 训练集批次数: {len(train_loader)}")
    print(f"  - 测试集批次数: {len(test_loader)}")
    print(f"  - 窗口大小: {window_size}")
    print(f"  - 预测范围: {horizon}")

    return train_loader, test_loader, scaler_y


def create_fusion_model(
    input_dim: int,
    window_size: int,
    horizon: int,
    num_channels: List[int] = [],
    transformer_heads: int = 4,
    device: str = "cuda",
) -> TCNDCATransformer:
    """
    创建TCN-DCA-Transformer融合模型。

    Args:
        input_dim: 输入特征维度
        window_size: 窗口大小
        horizon: 预测范围
        num_channels: TCN通道数列表
        transformer_heads: Transformer头数
        device: 计算设备

    Returns:
        TCNDCATransformer: 融合模型实例
    """
    if num_channels is None:
        num_channels = [64, 128]

    model_config = FusionModelConfig(
        input_dim=input_dim,
        output_dim=horizon,
        window_size=window_size,
        num_channels=num_channels,
        transformer_heads=transformer_heads,
        device=device,
    )

    model = TCNDCATransformer(model_config)
    model.print_model_architecture()

    return model


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    scaler_y,
    num_epochs: int = 20,
    learning_rate: float = 0.01,
    device: str = "cuda",
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    训练模型并进行评估。

    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        scaler_y: 目标变量缩放器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 计算设备

    Returns:
        Tuple[nn.Module, Dict[str, float]]: 训练后的模型和评估指标
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []
    best_test_loss = float("inf")

    print(f"\n开始训练，共 {num_epochs} 轮...")
    print(f"学习率: {learning_rate}")
    print(f"设备: {device}")
    print("-" * 50)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss

        print(f"轮次 [{epoch + 1}/{num_epochs}], 训练损失: {avg_train_loss:.6f}, " f"测试损失: {avg_test_loss:.6f}")

    print("-" * 50)
    print(f"训练完成，最佳测试损失: {best_test_loss:.6f}")

    # 评估模型
    metrics = evaluate_model(model, test_loader, scaler_y, device)

    return model, metrics


def evaluate_model(model: nn.Module, test_loader: DataLoader, scaler_y, device: str = "cuda") -> Dict[str, float]:
    """
    评估模型性能。

    Args:
        model: 训练后的模型
        test_loader: 测试数据加载器
        scaler_y: 目标变量缩放器
        device: 计算设备

    Returns:
        Dict[str, float]: 评估指标字典
    """
    model.eval()
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_predictions.extend(outputs.cpu().numpy())
            all_actuals.extend(labels.numpy())

    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)

    predictions_original = scaler_y.inverse_transform(predictions)
    actuals_original = scaler_y.inverse_transform(actuals)

    metrics = {
        "mse": calculate_mse(predictions_original, actuals_original),
        "rmse": calculate_rmse(predictions_original, actuals_original),
        "mae": calculate_mae(predictions_original, actuals_original),
        "r2": calculate_r2(
            predictions_original,
            actuals_original,
        ),
    }

    mape = calculate_mape(predictions_original, actuals_original)
    if not np.isnan(mape):
        metrics["mape"] = mape

    print(f"\n评估结果:")
    print(f"  MSE:  {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    if "mape" in metrics:
        print(f"  MAPE: {metrics['mape']:.4f}%")

    return metrics


def run_prediction_task(
    data_path: str,
    features: List[str],
    target: str,
    results_dir: str,
    window_size: int = 300,
    horizon: int = 50,
    num_epochs: int = 20,
    device: str = "cuda",
) -> None:
    """
    运行时间序列预测任务。

    Args:
        data_path: 数据文件路径
        features: 特征列名列表
        target: 目标列名
        results_dir: 结果保存目录
        window_size: 滑动窗口大小
        horizon: 预测范围
        num_epochs: 训练轮数
        device: 计算设备
    """
    print("\n" + "=" * 60)
    print("时间序列预测任务")
    print("=" * 60)

    results_dir = os.path.join(results_dir, "prediction")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n加载数据: {data_path}")
    train_loader, test_loader, scaler_y = load_data_for_prediction(
        data_path,
        features,
        target,
        window_size,
        horizon,
    )

    input_dim = len(features)
    print(f"输入特征维度: {input_dim}")

    print("\n创建融合模型...")
    model = create_fusion_model(
        input_dim=input_dim,
        window_size=window_size,
        horizon=horizon,
        device=device,
    )

    model, metrics = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        scaler_y=scaler_y,
        num_epochs=num_epochs,
        device=device,
    )

    print(f"\n保存结果到: {results_dir}")

    plot_predictions(
        predictions=np.array([]),
        actuals=np.array([]),
        save_path=os.path.join(results_dir, "prediction_vs_actual.png"),
    )

    plot_loss_curve(
        train_losses=[],
        test_losses=[],
        save_path=os.path.join(
            results_dir,
            "loss_curve.png",
        ),
    )

    save_evaluation_results(metrics, os.path.join(results_dir, "metrics.txt"))

    print("\n预测任务完成！")


def run_clustering_task(
    data_path: str,
    results_dir: str,
    pca_components: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    运行粒度球聚类任务。

    Args:
        data_path: 数据文件路径（MAT格式）
        results_dir: 结果保存目录
        pca_components: PCA降维维度

    Returns:
        Tuple[np.ndarray, np.ndarray]: 聚类标签和真实标签
    """
    print("\n" + "=" * 60)
    print("粒度球聚类任务")
    print("=" * 60)

    results_dir = os.path.join(results_dir, "clustering")
    os.makedirs(results_dir, exist_ok=True)

    try:
        from scipy.io import loadmat

        mat_data = loadmat(data_path)
    except Exception as e:
        print(f"加载MAT文件失败: {e}")
        raise

    features = mat_data["fea"]
    ground_truth = mat_data["gt"].flatten()

    print(f"\n数据加载完成:")
    print(f"  - 样本数: {features.shape[0]}")
    print(f"  - 特征维度: {features.shape[1]}")
    print(f"  - 类别数: {len(np.unique(ground_truth))}")

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    pca = PCA(n_components=pca_components)
    pca_features = pca.fit_transform(scaled_features)

    print(f"  - PCA降维后维度: {pca_components}")
    print(f"  - 解释方差比例: {sum(pca.explained_variance_ratio_):.4f}")

    current_ball_list = [pca_features]

    print("\n执行密度分裂...")
    current_ball_list = split_based_on_density(
        current_ball_list,
        min_points_for_split=8,
        min_points_after_split=4,
    )
    print(f"  - 密度分裂后球体数: {len(current_ball_list)}")

    detection_radius = calculate_detection_radius(current_ball_list)
    print(f"  - 检测半径: {detection_radius:.4f}")

    print("\n执行半径归一化...")
    current_ball_list = normalize_balls_by_radius_iterative(
        current_ball_list,
        detection_radius,
        radius_threshold_factor=2.0,
    )
    print(f"  - 归一化后球体数: {len(current_ball_list)}")

    print("\n评估聚类结果...")
    if len(current_ball_list) > 1:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        ball_centers = np.array([ball.mean(0) for ball in current_ball_list])
        num_actual_clusters = min(len(ball_centers), len(np.unique(ground_truth)))
        num_actual_clusters = max(2, num_actual_clusters)
        
        try:
            if len(ball_centers) <= num_actual_clusters:
                kmeans = KMeans(n_clusters=num_actual_clusters, random_state=42, n_init=10)
                ball_labels = kmeans.fit_predict(ball_centers)
            else:
                kmeans = KMeans(n_clusters=num_actual_clusters, random_state=42, n_init=10)
                ball_labels = kmeans.fit_predict(ball_centers)
        except Exception:
            ball_labels = np.arange(len(current_ball_list)) % num_actual_clusters
        
        labels = np.zeros(len(pca_features), dtype=int)
        for point_idx, point in enumerate(pca_features):
            distances_sq = np.sum((ball_centers - point) ** 2, axis=1)
            nearest_ball_idx = np.argmin(distances_sq)
            labels[point_idx] = ball_labels[nearest_ball_idx]
        
        ari_score = adjusted_rand_score(ground_truth, labels)
        nmi_score = normalized_mutual_info_score(ground_truth, labels)

        print(f"  - ARI分数: {ari_score:.4f}")
        print(f"  - NMI分数: {nmi_score:.4f}")

    print(f"\n保存结果到: {results_dir}")

    print("\n聚类任务完成！")

    return ground_truth, pca_features


def run_optimization_task(
    data_path: str,
    results_dir: str,
    pop_size: int = 3,
    max_iter: int = 2,
    timeout: float = 30,
) -> None:
    """
    运行BKOA超参数优化任务。

    Args:
        data_path: 数据文件路径
        results_dir: 结果保存目录
        pop_size: 种群大小
        max_iter: 最大迭代次数
        timeout: 超时时间（秒）
    """
    import psutil

    print("\n" + "=" * 60)
    print("BKOA超参数优化任务")
    print("=" * 60)

    mem_info = psutil.virtual_memory()
    print(f"CPU cores: {psutil.cpu_count(logical=True)}")
    print(f"Total Memory: {mem_info.total / (1024 ** 3):.2f} GB")
    print(f"Processing Dataset: {os.path.basename(data_path)}")
    print("-" * 50)

    results_dir = os.path.join(results_dir, "optimization")
    os.makedirs(results_dir, exist_ok=True)

    try:
        from scipy.io import loadmat

        mat_data = loadmat(data_path)
    except Exception as e:
        print(f"加载MAT文件失败: {e}")
        raise

    ground_truth = mat_data["gt"].flatten()
    features = mat_data["fea"]

    n_clusters_hint = len(np.unique(ground_truth))
    continuous_delta_lower = 0.05
    continuous_delta_upper = 1.05

    n_clusters_lower = max(2, int(n_clusters_hint * 0.7))
    n_clusters_upper = max(n_clusters_lower + 2, int(n_clusters_hint * 1.5) + 1)

    bounds = [(n_clusters_lower, n_clusters_upper), (continuous_delta_lower, continuous_delta_upper)]

    print(f"参数边界: n_clusters=[{n_clusters_lower}, {n_clusters_upper}], delta=[{continuous_delta_lower:.3f}, {continuous_delta_upper:.3f}]")
    print(f"  (delta将离散化到[0.1, 1.0], 步长0.1进行评估)")

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import normalized_mutual_info_score

    time_total_start = time.time()

    print("\n步骤1: 生成粒度球...")
    time_gb_start = time.time()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)

    ball_list = [pca_features]
    ball_list = split_based_on_density(
        ball_list,
        min_points_for_split=8,
        min_points_after_split=4,
    )

    detection_radius = calculate_detection_radius(ball_list)
    ball_list = normalize_balls_by_radius_iterative(
        ball_list,
        detection_radius,
        radius_threshold_factor=2.0,
    )

    time_gb_end = time.time()
    print(f"  粒度球生成时间: {time_gb_end - time_gb_start:.2f} 秒")
    print(f"  数据点数: {pca_features.shape[0]}, 生成原始球数: {len(ball_list)}")

    print("\n步骤2: 准备球字典...")
    from src.clustering import GranularBall

    ball_dict = {}
    valid_ball_count = 0
    for i, ball_points in enumerate(ball_list):
        if len(ball_points) > 0:
            gb = GranularBall(ball_points, valid_ball_count)
            if len(gb.center) > 0:
                ball_dict[valid_ball_count] = gb
                valid_ball_count += 1

    num_valid_balls = len(ball_dict)
    print(f"  有效球数量: {num_valid_balls}")

    if num_valid_balls == 0:
        print("错误: 未生成有效的粒度球，跳过优化。")
        return

    if bounds[0][1] > num_valid_balls:
        bounds[0][1] = num_valid_balls
    if bounds[0][0] > bounds[0][1]:
        bounds[0][0] = bounds[0][1]

    print("\n步骤3: 使用BKOA优化参数...")
    time_opt_start = time.time()

    print(f"  参数边界: n_clusters=[{bounds[0][0]}, {bounds[0][1]}], delta=[{bounds[1][0]:.3f}, {bounds[1][1]:.3f}]")

    optimizer_config = BKOAConfig(pop_size=pop_size, max_iter=max_iter, timeout=timeout, verbose=True)

    objective = create_clustering_objective(
        ball_dict=ball_dict,
        features=pca_features,
        ground_truth=ground_truth,
        delta_range=(0.1, 1.0),
    )

    optimizer = BKOAOptimizer(objective_function=objective, bounds=bounds, config=optimizer_config)

    best_params, best_fitness = optimizer.run()

    time_opt_end = time.time()
    print(f"  BKOA优化时间: {time_opt_end - time_opt_start:.2f} 秒")

    optimized_n_clusters = int(best_params[0])
    optimized_continuous_delta = float(best_params[1])
    snapped_delta = round(optimized_continuous_delta * 10.0) / 10.0
    optimized_delta = max(0.1, min(1.0, snapped_delta))
    best_ari = -best_fitness

    print("\n--- 优化结果 ---")
    print(f"最优聚类数: {optimized_n_clusters}")
    print(f"最优离散delta: {optimized_delta:.1f} (来自连续值: {optimized_continuous_delta:.4f})")
    print(f"最优ARI: {best_ari:.6f}")

    print("\n步骤4: 使用优化后的离散参数进行最终评估...")
    final_ari, final_labels = objective._perform_clustering(optimized_n_clusters, optimized_delta)
    final_nmi = normalized_mutual_info_score(ground_truth, final_labels)

    print(f"最终ARI (使用优化后的离散参数): {final_ari:.6f}")
    print(f"最终NMI (使用优化后的离散参数): {final_nmi:.6f}")

    time_total_end = time.time()
    total_duration = time_total_end - time_total_start
    print(f"\n数据集总时间: {total_duration:.2f} 秒")
    print("-" * 50)

    print(f"\n保存结果到: {results_dir}")

    results = {
        "optimized_n_clusters": optimized_n_clusters,
        "optimized_delta": optimized_delta,
        "best_ari": best_ari,
        "final_ari": final_ari,
        "final_nmi": final_nmi,
        "optimization_time": time_opt_end - time_opt_start,
        "total_time": total_duration,
        "num_valid_balls": num_valid_balls,
    }

    save_evaluation_results(
        results,
        os.path.join(results_dir, "optimization_results.txt"),
        metrics_order=[
            "optimized_n_clusters",
            "optimized_delta",
            "best_ari",
            "final_ari",
            "final_nmi",
            "optimization_time",
            "total_time",
            "num_valid_balls",
        ],
    )

    print("\n=== 最终优化结果摘要 ===")
    print(f"数据集: {os.path.basename(data_path)} (有效球数: {num_valid_balls})")
    print(f"  优化参数: n_clusters={optimized_n_clusters}, delta={optimized_delta:.1f}")
    print(f"  最优ARI (优化过程): {best_ari:.4f}")
    print(f"  最终ARI (评估):     {final_ari:.4f}")
    print(f"  最终NMI (评估):     {final_nmi:.4f}")
    print(f"  总时间: {total_duration:.2f}秒")
    print("-" * 25)

    print("\n优化任务完成！")


def run_all_tasks(
    data_path: str,
    features: List[str],
    target: str,
    results_dir: str,
    window_size: int = 300,
    horizon: int = 50,
    num_epochs: int = 20,
    device: str = "cuda",
) -> None:
    """
    运行所有任务（聚类 + 优化 + 预测）。

    Args:
        data_path: 数据文件路径
        features: 特征列名列表
        target: 目标列名
        results_dir: 结果保存目录
        window_size: 滑动窗口大小
        horizon: 预测范围
        num_epochs: 训练轮数
        device: 计算设备
    """
    print("\n" + "=" * 60)
    print("完整流程任务")
    print("=" * 60)

    base_results_dir = results_dir
    results_dir = create_results_directory(base_results_dir)

    print(f"结果保存目录: {results_dir}")

    try:
        ground_truth, pca_features = run_clustering_task(
            data_path=data_path,
            results_dir=results_dir,
            pca_components=2,
        )

        run_optimization_task(
            data_path=data_path,
            results_dir=results_dir,
        )

        print("\n完整流程完成！")

    except Exception as e:
        print(f"执行完整流程时出错: {e}")
        raise


def create_results_directory(base_dir: str) -> str:
    """
    创建结果保存目录。

    Args:
        base_dir: 基础目录名

    Returns:
        str: 创建的完整目录路径
    """
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f"results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="GMF-Net: 时间序列预测与聚类分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行时间序列预测
  python main.py --task=prediction --data_path=data/train.csv --features f1 f2 f3 --target target
  
  # 运行粒度球聚类
  python main.py --task=clustering --data_path=data/dataset.mat
  
  # 运行BKOA超参数优化
  python main.py --task=optimization --data_path=data/dataset.mat
  
  # 运行完整流程
  python main.py --task=all --data_path=data/dataset.mat
        """,
    )

    parser.add_argument(
        "--task",
        type=str,
        default="prediction",
        choices=["prediction", "clustering", "optimization", "all"],
        help="任务类型 (默认: prediction)",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/train.csv",
        help="数据文件路径",
    )

    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=["feature1", "feature2", "feature3"],
        help="特征列名列表",
    )

    parser.add_argument(
        "--target",
        type=str,
        default="target",
        help="目标列名",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="结果保存目录",
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=300,
        help="滑动窗口大小",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="预测范围",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="训练轮数",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="计算设备",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )

    parser.add_argument(
        "--pop_size",
        type=int,
        default=3,
        help="BKOA种群大小",
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=2,
        help="BKOA最大迭代次数",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="BKOA超时时间（秒）",
    )

    return parser.parse_args()


def main() -> None:
    """
    主函数入口。

    解析命令行参数并调用相应的任务处理函数。
    """
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("GMF-Net - 时间序列预测与聚类分析工具")
    print("=" * 60)

    set_random_seed(args.seed)

    config = get_default_config()

    try:
        if args.task == "prediction":
            run_prediction_task(
                data_path=args.data_path,
                features=args.features,
                target=args.target,
                results_dir=args.results_dir,
                window_size=args.window_size,
                horizon=args.horizon,
                num_epochs=args.num_epochs,
                device=args.device,
            )

        elif args.task == "clustering":
            run_clustering_task(
                data_path=args.data_path,
                results_dir=args.results_dir,
            )

        elif args.task == "optimization":
            run_optimization_task(
                data_path=args.data_path,
                results_dir=args.results_dir,
                pop_size=args.pop_size,
                max_iter=args.max_iter,
                timeout=args.timeout,
            )

        elif args.task == "all":
            run_all_tasks(
                data_path=args.data_path,
                features=args.features,
                target=args.target,
                results_dir=args.results_dir,
                window_size=args.window_size,
                horizon=args.horizon,
                num_epochs=args.num_epochs,
                device=args.device,
            )

    except KeyboardInterrupt:
        print("\n用户中断执行")
        sys.exit(0)
    except Exception as e:
        print(f"\n执行出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("程序执行完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
