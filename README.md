# GMF-Net

## 项目概述

GMF-Net（Granular-ball Multi-model Fusion Network）是一个基于粒度球结构的时间序列预测与聚类分析框架。项目采用模块化设计，将时间序列预测、粒度球聚类和超参数优化有机结合，提供完整的从数据处理到模型训练、聚类分析的端到端解决方案。

核心功能包括：基于 TCN-DCA-Transformer 融合模型的多步时间序列预测、基于粒度球结构的无监督聚类算法、以及使用黑翅鸢优化算法（BKOA）自动优化聚类超参数。框架全程使用 Python 类型注解，支持完整的类型检查，并提供丰富的中文注释和详细的运行日志输出。

## 项目结构

```txt
GMF-Net/
├── src/                         # 源代码目录
│   ├── __init__.py              # 包初始化
│   ├── main.py                  # 主程序入口
│   ├── config.py                # 配置文件
│   │
│   ├── data/                    # 数据处理模块
│   │   ├── __init__.py
│   │   ├── base.py              # 基础数据加载器抽象类
│   │   ├── time_series.py       # 时间序列数据处理
│   │   └── dataset.py           # 数据集管理
│   │
│   ├── models/                  # 神经网络模型模块
│   │   ├── __init__.py
│   │   ├── base.py              # 基础模型抽象类
│   │   ├── tcn.py               # 时间卷积网络
│   │   ├── attention.py         # 通道注意力机制
│   │   ├── transformer.py       # Transformer编码器
│   │   └── fusion.py            # TCN-DCA-Transformer融合模型
│   │
│   ├── clustering/              # 聚类算法模块
│   │   ├── __init__.py
│   │   ├── granular_ball.py     # 粒度球核心类
│   │   ├── splitting.py         # 分裂策略
│   │   └── density.py           # 密度计算和归一化
│   │
│   ├── optimization/            # 优化算法模块
│   │   ├── __init__.py
│   │   ├── base.py              # 优化器基类
│   │   ├── bkoa.py              # 黑翅鸢优化算法
│   │   └── objective.py         # 目标函数定义
│   │
│   └── utils/                   # 工具模块
│       ├── __init__.py
│       ├── metrics.py           # 评估指标
│       └── visualization.py     # 可视化工具
│
├── cluster/                     # 原始聚类模块（参考实现）
├── data/                        # 数据目录
├── results/                     # 结果保存目录
├── README.md                    # 项目说明文档
└── requirements.txt             # 依赖列表
```

## 环境要求

- **Python 版本**：3.12 ~ 3.13
- **包管理器**：推荐使用 [uv](https://github.com/astral-sh/uv)

项目使用 `pyproject.toml` 管理依赖：

```toml
[project]
name = "gmf-net"
version = "0.0.1"
requires-python = ">=3.12,<=3.13"
dependencies = [
    "tqdm",
    "pandas",
    "psutil",
    "matplotlib",
    "scikit-learn",
    "torch==2.9.1",
]
```

安装依赖：

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip + requirements.txt
pip install -r requirements.txt
```

## 快速开始

### 命令行使用

项目提供了便捷的命令行接口，支持四种任务模式：

```bash
# 时间序列预测任务
python -m src.main --task=prediction --data_path=./dataset/train.csv --features feature1 feature2 feature3 --target target --num_epochs=20

# 粒度球聚类任务
python -m src.main --task=clustering --data_path=./dataset/dataset.mat

# BKOA超参数优化任务
python -m src.main --task=optimization --data_path=./dataset/dataset.mat --pop_size=3 --max_iter=2 --timeout=30

# 运行完整流程（预测+聚类+优化）
python -m src.main --task=all --data_path=./dataset/dataset.mat
```

### 参数说明

#### 通用参数

| 参数            | 类型 | 默认值  | 说明                                             |
| --------------- | ---- | ------- | ------------------------------------------------ |
| `--task`        | str  | 必填    | 任务类型：prediction/clustering/optimization/all |
| `--data_path`   | str  | 必填    | 数据文件路径（CSV 或 MAT 格式）                  |
| `--results_dir` | str  | results | 结果保存目录                                     |

#### 预测任务参数

| 参数              | 类型  | 默认值 | 说明                 |
| ----------------- | ----- | ------ | -------------------- |
| `--features`      | str   | 必填   | 特征列名（空格分隔） |
| `--target`        | str   | 必填   | 目标列名             |
| `--window_size`   | int   | 300    | 滑动窗口大小         |
| `--horizon`       | int   | 50     | 预测步长             |
| `--num_epochs`    | int   | 20     | 训练轮数             |
| `--batch_size`    | int   | 32     | 批大小               |
| `--learning_rate` | float | 0.01   | 学习率               |

#### 聚类任务参数

| 参数                       | 类型 | 默认值 | 说明               |
| -------------------------- | ---- | ------ | ------------------ |
| `--pca_components`         | int  | 2      | PCA 降维维度       |
| `--min_points_for_split`   | int  | 8      | 密度分裂最小点数   |
| `--min_points_after_split` | int  | 4      | 分裂后子球最小点数 |

#### 优化任务参数

| 参数         | 类型  | 默认值 | 说明                |
| ------------ | ----- | ------ | ------------------- |
| `--pop_size` | int   | 3      | BKOA 种群大小       |
| `--max_iter` | int   | 2      | BKOA 最大迭代次数   |
| `--timeout`  | float | 30     | BKOA 超时时间（秒） |

### 编程接口

除了命令行方式，项目也支持通过编程方式调用各模块：

```python
# 数据加载示例
from src.data import TimeSeriesDataLoader, TimeSeriesConfig

config = TimeSeriesConfig(
    file_path='data/train.csv',
    features=['feature1', 'feature2', 'feature3'],
    target='target',
    window_size=300,
    horizon=50
)

loader = TimeSeriesDataLoader(config)
train_loader, test_loader, scaler_y, scaler_X = loader.load_data()
```

```python
# 模型创建示例
from src.models import TCNDCATransformer, FusionModelConfig

config = FusionModelConfig(
    input_dim=5,
    output_dim=50,
    window_size=300,
    num_channels=[64, 128],
    transformer_heads=4
)

model = TCNDCATransformer(config)
model.print_model_architecture()
```

```python
# 聚类示例
from src.clustering import (
    GranularBall,
    split_based_on_density,
    normalize_balls_by_radius,
    calculate_detection_radius
)

# 基于密度分裂
ball_list = split_based_on_density(ball_list)

# 归一化
detection_radius = calculate_detection_radius(ball_list)
ball_list = normalize_balls_by_radius(ball_list, detection_radius)
```

```python
# BKOA优化示例
from src.optimization import BKOAOptimizer, BKOAConfig

config = BKOAConfig(
    pop_size=3,
    max_iter=2,
    timeout=30
)

optimizer = BKOAOptimizer(
    objective_function=objective_func,
    bounds=[(2, 20), (0.1, 1.0)],
    config=config
)

best_params, best_fitness = optimizer.run()
```

## 模块详解

### 数据处理模块（src/data）

数据处理模块提供了时间序列数据的加载、预处理和窗口化功能。模块采用抽象基类设计，支持灵活的数据源扩展。

**核心组件**：

- `DataLoaderBase`：基础数据加载器抽象类，定义了数据加载的标准接口
- `TimeSeriesConfig`：时间序列配置类，包含窗口大小、预测步长等参数
- `TimeSeriesDataLoader`：时间序列数据加载器，支持 CSV 格式数据
- `DatasetManager`：数据集管理器，支持多数据集管理和批量处理

该模块的主要功能包括：支持多种数据格式（CSV、MAT）、内置数据验证和错误处理、滑动窗口自动生成、留一用户交叉验证支持。数据加载过程中会自动进行归一化处理，并将数据拆分为训练集和测试集。

### 模型模块（src/models）

模型模块实现了基于 TCN-DCA-Transformer 融合的时间序列预测模型，采用模块化设计，各组件可独立使用或组合使用。

**核心组件**：

- `BaseModel`：基础模型抽象类，定义了模型的标准接口
- `ModelConfig`：模型配置基类
- `TCNConfig`：时间卷积网络配置类
- `TemporalConvNet`：时间卷积网络实现
- `ChannelAttentionConfig`：通道注意力机制配置类
- `ChannelAttention`：通道注意力机制（DCA）
- `TransformerConfig`：Transformer 编码器配置类
- `TransformerBlock`：Transformer 编码器块
- `FusionModelConfig`：融合模型配置类
- `TCNDCATransformer`：TCN-DCA-Transformer 融合模型

模型架构特点：TCN 负责提取局部时序特征，采用空洞卷积扩大感受野；DCA（通道注意力机制）自适应调整各通道权重；Transformer 编码器捕捉长距离依赖关系；三个模块的输出通过融合层进行整合，实现多尺度特征融合。

### 聚类模块（src/clustering）

聚类模块实现了基于粒度球结构的无监督聚类算法。粒度球是一种自适应的数据聚合结构，能够有效捕捉数据的局部密度分布。

**核心组件**：

- `GranularBall`：粒度球核心类，表示一个粒度球及其属性
- `GranularBallCollection`：粒度球集合类，管理多个粒度球
- `SplittingStrategy`：分裂策略枚举，支持不同分裂方法

**核心函数**：

- `split_ball_by_distance()`：基于距离的球体分裂
- `calculate_density()`：计算球体密度
- `calculate_radius()`：计算球体半径
- `split_based_on_density()`：基于密度改进条件的球体分裂
- `calculate_detection_radius()`：计算检测半径
- `normalize_balls_by_radius()`：基于半径的球体归一化
- `normalize_balls_by_radius_iterative()`：迭代归一化
- `normalize_balls_by_density()`：基于密度的归一化
- `evaluate_clustering_quality()`：评估聚类质量

**算法流程**：

1. 初始化：将整个数据集作为一个粒度球
2. 密度分裂：迭代分裂球体，只在分裂后子球密度改进时才执行分裂
3. 半径归一化：基于检测半径对球体进行归一化处理
4. 聚类：将归一化后的球体映射回原始数据点，进行最终聚类

### 优化模块（src/optimization）

优化模块实现了黑翅鸢优化算法（BKOA），用于自动优化聚类超参数。BKOA 是一种基于生物行为的元启发式优化算法，模拟黑翅鸢的捕食行为。

**核心组件**：

- `BaseOptimizer`：基础优化器抽象类，定义了优化器的标准接口
- `OptimizerConfig`：优化器配置基类
- `BKOAConfig`：BKOA 配置类，包含种群大小、迭代次数等参数
- `BKOAOptimizer`：黑翅鸢优化算法实现
- `ObjectiveFunction`：目标函数抽象类
- `ClusteringObjective`：聚类目标函数，计算 ARI 分数

**BKOA 算法特点**：

- 自适应参数调整：根据搜索进度动态调整探索与开发平衡
- 早停机制：当优化效果不明显时提前终止
- 超时保护：防止优化过程无限运行
- 进度显示：集成 tqdm 进度条，实时显示优化进度

**默认参数配置**：

| 参数     | 默认值 | 说明           |
| -------- | ------ | -------------- |
| pop_size | 3      | 种群大小       |
| max_iter | 2      | 最大迭代次数   |
| timeout  | 30     | 超时时间（秒） |
| alpha    | 0.8    | 探索系数       |
| beta     | 0.5    | 开发系数       |
| gamma    | 0.1    | 扰动系数       |

### 工具模块（src/utils）

工具模块提供了项目常用的辅助功能，包括评估指标计算和可视化工具。

**评估指标**：

- `calculate_mse()`：均方误差
- `calculate_rmse()`：均方根误差
- `calculate_mae()`：平均绝对误差
- `calculate_mape()`：平均绝对百分比误差
- `calculate_r2()`：决定系数 R²
- `calculate_ari()`：调整兰德指数
- `calculate_nmi()`：归一化互信息
- `EvaluationMetrics`：综合评估指标类

**可视化工具**：

- `plot_predictions()`：绘制预测结果对比图
- `plot_loss_curve()`：绘制训练损失曲线
- `plot_clustering_results()`：绘制聚类结果可视化
- `save_evaluation_results()`：保存评估结果到文件

## 配置系统

项目使用`Config`类作为配置单例，整合所有配置参数：

```python
from src.config import Config, get_default_config

# 获取默认配置
config = get_default_config()

# 访问配置
print(config.training.learning_rate)  # 0.01
print(config.fusion.window_size)  # 300
print(config.clustering.min_points_for_split)  # 8

# 修改配置
config.training.num_epochs = 30
config.optimization.pop_size = 5

# 保存配置
config.save_config('my_config.json')

# 加载配置
config.load_config('my_config.json')
```

**配置类结构**：

| 配置类               | 说明                                           |
| -------------------- | ---------------------------------------------- |
| `TCNModelConfig`     | TCN 模型配置（通道数、卷积核大小、dropout 等） |
| `AttentionConfig`    | 注意力机制配置（压缩比）                       |
| `TransformerConfig`  | Transformer 配置（头数、层数、dropout 等）     |
| `FusionModelConfig`  | 融合模型配置（窗口大小、预测步长）             |
| `TrainingConfig`     | 训练配置（轮数、学习率、批大小等）             |
| `ClusteringConfig`   | 聚类配置（分裂参数、PCA 维度等）               |
| `OptimizationConfig` | 优化器配置（BKOA 参数）                        |
| `DataConfig`         | 数据路径配置                                   |

## 数据格式

### CSV 格式（用于预测任务）

CSV 文件应包含特征列和目标列，数据格式如下：

```csv
feature1,feature2,feature3,target
0.1,0.2,0.3,0.15
0.2,0.3,0.4,0.25
...
```

### MAT 格式（用于聚类任务）

MAT 文件应包含以下变量：

| 变量名  | 形状                    | 说明                       |
| ------- | ----------------------- | -------------------------- |
| `data`  | (n_samples, n_features) | 数据矩阵                   |
| `label` | (n_samples,)            | 真实标签（可选，用于评估） |

## 许可证

本项目仅供研究和学习使用。
