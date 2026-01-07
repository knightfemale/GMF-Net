"""
基础数据加载器抽象类

该模块定义了数据加载器的抽象基类，提供了通用的数据加载接口。
所有具体的数据加载器都应该继承此类并实现抽象方法。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class DataConfig:
    """
    数据配置类，用于存储数据加载和预处理的相关参数。
    
    Attributes:
        file_path (str): 数据文件路径
        features (List[str]): 特征列名列表
        target (str): 目标列名
        window_size (int): 滑动窗口大小
        horizon (int): 预测范围
        test_size (float): 测试集比例
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        scaler_type (str): 缩放器类型 ('standard' 或 'minmax')
        random_state (int): 随机种子
    """
    file_path: str
    features: List[str]
    target: str
    window_size: int = 300
    horizon: int = 50
    test_size: float = 0.2
    batch_size: int = 32
    shuffle: bool = True
    scaler_type: str = 'standard'
    random_state: int = 42
    
    def __post_init__(self) -> None:
        """
        初始化后验证配置参数的有效性。
        
        Raises:
            ValueError: 如果参数值无效
        """
        if self.window_size < 1:
            raise ValueError(f"window_size 必须大于0，当前值: {self.window_size}")
        if self.horizon < 1:
            raise ValueError(f"horizon 必须大于0，当前值: {self.horizon}")
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size 必须在(0,1)范围内，当前值: {self.test_size}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size 必须大于0，当前值: {self.batch_size}")


class TimeSeriesDataset(Dataset):
    """
    时间序列数据集类，用于PyTorch DataLoader。
    
    该类将时间序列数据封装为PyTorch Dataset格式，支持批量加载和数据转换。
    
    Attributes:
        X (torch.Tensor): 特征数据张量，形状为 [样本数, 窗口大小, 特征数]
        y (torch.Tensor): 目标数据张量，形状为 [样本数, 预测范围]
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        初始化时间序列数据集。
        
        Args:
            X: 特征数组，形状为 [样本数, 窗口大小, 特征数]
            y: 目标数组，形状为 [样本数, 预测范围]
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self) -> int:
        """
        返回数据集中样本数量。
        
        Returns:
            int: 样本数量
        """
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定索引的数据样本。
        
        Args:
            idx: 样本索引
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (特征张量, 目标张量)
        """
        return self.X[idx], self.y[idx]


class DataLoaderBase(ABC):
    """
    基础数据加载器抽象类。
    
    该类定义了数据加载器的通用接口，所有具体的数据加载器都需要实现以下方法：
    - load_data: 加载和预处理数据
    - create_windows: 创建滑动窗口
    - get_data_loaders: 获取训练和测试数据加载器
    
    Attributes:
        config (DataConfig): 数据配置对象
        scaler_X (StandardScaler): 特征缩放器
        scaler_y (StandardScaler): 目标缩放器
    """
    
    def __init__(self, config: DataConfig) -> None:
        """
        初始化基础数据加载器。
        
        Args:
            config: 数据配置对象
        """
        self.config = config
        self.scaler_X: Optional[StandardScaler] = None
        self.scaler_y: Optional[StandardScaler] = None
    
    @abstractmethod
    def load_data(self) -> Tuple[DataLoader, DataLoader, StandardScaler, Optional[StandardScaler]]:
        """
        加载并预处理数据。
        
        Returns:
            Tuple[DataLoader, DataLoader, StandardScaler, Optional[StandardScaler]]:
                (训练数据加载器, 测试数据加载器, 目标缩放器, 特征缩放器)
        """
        pass
    
    @abstractmethod
    def create_windows(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建滑动窗口。
        
        Args:
            data: 预处理后的特征数据
            target: 预处理后的目标数据
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (窗口化后的特征, 窗口化后的目标)
        """
        pass
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        验证数据文件的有效性和完整性。
        
        Args:
            data: 加载的数据DataFrame
            
        Raises:
            ValueError: 如果数据缺少必需的列或包含无效值
        """
        missing_cols = [col for col in self.config.features + [self.config.target] 
                       if col not in data.columns]
        if missing_cols:
            raise ValueError(f"CSV 文件中缺少以下列: {missing_cols}")
        
        for col in self.config.features + [self.config.target]:
            if not np.issubdtype(data[col].dtype, np.number):
                raise ValueError(f"列 '{col}' 不是数值类型")
    
    def _scale_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对特征和目标进行标准化处理。
        
        Args:
            X: 原始特征数据
            y: 原始目标数据
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (标准化后的特征, 标准化后的目标)
        """
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        return X_scaled, y_scaled
    
    def _check_data_sufficiency(self, data_length: int) -> None:
        """
        检查数据长度是否足够创建滑动窗口。
        
        Args:
            data_length: 原始数据长度
            
        Raises:
            ValueError: 如果数据长度不足
        """
        min_required = self.config.window_size + self.config.horizon - 1
        if data_length <= min_required:
            raise ValueError(
                f"数据不足（{data_length} 行），无法创建窗口大小={self.config.window_size} "
                f"和预测范围={self.config.horizon} 的窗口"
            )
