"""
时间序列数据加载器

该模块提供了时间序列数据的加载和预处理功能，
支持滑动窗口生成和批量数据加载。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from .base import DataLoaderBase, DataConfig, TimeSeriesDataset


@dataclass
class TimeSeriesConfig(DataConfig):
    """
    时间序列数据配置类。
    
    在基础DataConfig基础上添加了时间序列特定的配置参数。
    
    Attributes:
        user_id_col (Optional[str]): 用户ID列名，用于留一用户交叉验证
        timestamp_col (str): 时间戳列名
        datetime_index (bool): 是否将时间戳列作为DataFrame索引
    """
    user_id_col: Optional[str] = None
    timestamp_col: str = 'timestamp'
    datetime_index: bool = True
    
    def __post_init__(self) -> None:
        """
        初始化后进行额外验证。
        """
        super().__post_init__()
        if self.user_id_col is not None and not isinstance(self.user_id_col, str):
            raise ValueError(f"user_id_col 必须是字符串类型，当前值: {self.user_id_col}")


class TimeSeriesDataLoader(DataLoaderBase):
    """
    时间序列数据加载器类。
    
    该类实现了基础数据加载器接口，专门用于处理时间序列数据。
    支持从CSV文件加载数据，进行标准化处理，创建滑动窗口，并生成PyTorch DataLoader。
    
    Example:
        >>> config = TimeSeriesConfig(
        ...     file_path='data.csv',
        ...     features=['feature1', 'feature2', 'feature3'],
        ...     target='target',
        ...     window_size=100,
        ...     horizon=10
        ... )
        >>> loader = TimeSeriesDataLoader(config)
        >>> train_loader, test_loader, scaler_y, scaler_X = loader.load_data()
    
    Attributes:
        config (TimeSeriesConfig): 时间序列数据配置对象
    """
    
    def __init__(self, config: TimeSeriesConfig) -> None:
        """
        初始化时间序列数据加载器。
        
        Args:
            config: 时间序列数据配置对象
        """
        super().__init__(config)
        self.config = config
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, StandardScaler, Optional[StandardScaler]]:
        """
        加载并预处理时间序列数据。
        
        该方法执行以下步骤：
        1. 从CSV文件读取数据
        2. 验证数据完整性和有效性
        3. 提取特征和目标变量
        4. 进行标准化处理
        5. 创建滑动窗口
        6. 划分训练集和测试集
        7. 生成DataLoader对象
        
        Returns:
            Tuple[DataLoader, DataLoader, StandardScaler, Optional[StandardScaler]]:
                训练数据加载器、测试数据加载器、目标缩放器、特征缩放器
                
        Raises:
            FileNotFoundError: 如果数据文件不存在
            ValueError: 如果数据缺少必需的列或格式不正确
        """
        data = self._load_csv_file()
        self._validate_data(data)
        
        X, y = self._extract_features_and_target(data)
        X_scaled, y_scaled = self._scale_data(X, y)
        
        X_windows, y_windows = self.create_windows(X_scaled, y_scaled)
        
        if X_windows.shape[0] == 0:
            raise ValueError("无法创建滑动窗口，检查数据长度、window_size和horizon")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_windows, y_windows, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state,
            shuffle=False
        )
        
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=self.config.shuffle
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        
        return train_loader, test_loader, self.scaler_y, self.scaler_X
    
    def create_windows(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列滑动窗口。
        
        该方法将时间序列数据转换为监督学习格式，每个窗口包含：
        - 输入：连续的时间步数据
        - 输出：预测范围内的目标值
        
        Args:
            data: 预处理后的特征数据，形状为 [时间步数, 特征数]
            target: 预处理后的目标数据，形状为 [时间步数]
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                X_windows: 窗口化后的特征，形状为 [样本数, 窗口大小, 特征数]
                y_windows: 窗口化后的目标，形状为 [样本数, 预测范围]
                
        Example:
            >>> data = np.random.randn(1000, 5)
            >>> target = np.random.randn(1000)
            >>> X_windows, y_windows = create_windows(data, target)
            >>> X_windows.shape  # (651, 300, 5) 假设window_size=300, horizon=50
        """
        window_size = self.config.window_size
        horizon = self.config.horizon
        
        X_windows = []
        y_windows = []
        
        for i in range(len(data) - window_size - horizon + 1):
            window_input = data[i:(i + window_size)]
            window_output = target[(i + window_size):(i + window_size + horizon)]
            
            X_windows.append(window_input)
            y_windows.append(window_output)
        
        return np.array(X_windows), np.array(y_windows)
    
    def _load_csv_file(self) -> pd.DataFrame:
        """
        从CSV文件加载时间序列数据。
        
        Returns:
            pd.DataFrame: 加载的数据
            
        Raises:
            FileNotFoundError: 如果文件不存在
            KeyError: 如果缺少时间戳列
        """
        try:
            if self.config.datetime_index:
                data = pd.read_csv(
                    self.config.file_path, 
                    parse_dates=[self.config.timestamp_col], 
                    index_col=self.config.timestamp_col
                )
            else:
                data = pd.read_csv(self.config.file_path)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"在 {self.config.file_path} 未找到文件")
        except KeyError:
            raise KeyError(
                f"在 {self.config.file_path} 中找不到 '{self.config.timestamp_col}' 列 "
                "或无法解析为日期"
            )
    
    def _extract_features_and_target(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        从数据中提取特征和目标变量。
        
        Args:
            data: 加载的数据DataFrame
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (特征矩阵, 目标向量)
        """
        X = data[self.config.features].values
        y = data[self.config.target].values
        
        if not np.issubdtype(y.dtype, np.number):
            y = pd.to_numeric(y, errors='coerce')
            if np.isnan(y).any():
                raise ValueError(f"无法将目标列 '{self.config.target}' 转换为数值类型")
        
        return X, y


class LeaveOneUserOutDataLoader:
    """
    留一用户交叉验证数据加载器类。
    
    该类专门用于处理用户级别的时间序列数据，支持留一用户交叉验证。
    每次验证时保留一个用户作为测试集，其余用户作为训练集。
    
    Attributes:
        config (TimeSeriesConfig): 时间序列数据配置对象
        user_data (Dict): 按用户组织的数据字典
        scaler_X (StandardScaler): 特征缩放器
        scaler_y (StandardScaler): 目标缩放器
        users (List): 用户ID列表
    """
    
    def __init__(self, config: TimeSeriesConfig) -> None:
        """
        初始化留一用户交叉验证数据加载器。
        
        Args:
            config: 时间序列数据配置对象
            
        Raises:
            ValueError: 如果user_id_col未指定或用户数量不足
        """
        if config.user_id_col is None:
            raise ValueError("留一用户交叉验证需要指定 user_id_col")
        
        self.config = config
        self.user_data: Dict = {}
        self.scaler_X: Optional[StandardScaler] = None
        self.scaler_y: Optional[StandardScaler] = None
        self.users: Optional[List] = None
    
    def load_data(self) -> Tuple[Dict, StandardScaler, StandardScaler]:
        """
        加载并预处理用户级别的时间序列数据。
        
        该方法执行以下步骤：
        1. 从CSV文件读取数据
        2. 提取用户ID并获取唯一用户列表
        3. 对数据进行标准化处理
        4. 创建滑动窗口并按用户ID组织数据
        
        Returns:
            Tuple[Dict, StandardScaler, StandardScaler]:
                (按用户组织的数据字典, 目标缩放器, 特征缩放器)
                
        Raises:
            ValueError: 如果用户数量不足2个
        """
        data = self._load_csv_file()
        self._validate_user_data(data)
        
        X, y, user_ids = self._extract_features_and_target(data)
        X_scaled, y_scaled = self._scale_data(X, y)
        
        X_windows, y_windows, window_user_ids = self._create_windows_with_user_ids(
            X_scaled, y_scaled, user_ids
        )
        
        self._organize_user_data(X_windows, y_windows, window_user_ids)
        
        return self.user_data, self.scaler_y, self.scaler_X
    
    def get_num_users(self) -> int:
        """
        获取用户数量。
        
        Returns:
            int: 用户数量
        """
        return len(self.users) if self.users is not None else 0
    
    def get_user_ids(self) -> List:
        """
        获取所有用户ID列表。
        
        Returns:
            List: 用户ID列表
        """
        return self.users if self.users is not None else []
    
    def _load_csv_file(self) -> pd.DataFrame:
        """
        从CSV文件加载时间序列数据。
        
        Returns:
            pd.DataFrame: 加载的数据
        """
        try:
            if self.config.datetime_index:
                return pd.read_csv(
                    self.config.file_path,
                    parse_dates=[self.config.timestamp_col],
                    index_col=self.config.timestamp_col
                )
            else:
                return pd.read_csv(self.config.file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"在 {self.config.file_path} 未找到文件")
    
    def _validate_user_data(self, data: pd.DataFrame) -> None:
        """
        验证用户数据的有效性和完整性。
        
        Args:
            data: 加载的数据DataFrame
            
        Raises:
            ValueError: 如果缺少必需的列或用户数量不足
        """
        if self.config.user_id_col not in data.columns:
            raise ValueError(f"CSV文件中缺少用户ID列: {self.config.user_id_col}")
        
        missing_cols = [col for col in self.config.features + [self.config.target] 
                       if col not in data.columns]
        if missing_cols:
            raise ValueError(f"CSV文件中缺少以下列: {missing_cols}")
        
        unique_users = data[self.config.user_id_col].unique()
        if len(unique_users) < 2:
            raise ValueError(f"数据中只有 {len(unique_users)} 个用户，无法进行留一用户交叉验证")
        
        self.users = list(unique_users)
    
    def _extract_features_and_target(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从数据中提取特征、目标变量和用户ID。
        
        Args:
            data: 加载的数据DataFrame
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (特征矩阵, 目标向量, 用户ID数组)
        """
        X = data[self.config.features].values
        y = data[self.config.target].values
        user_ids = data[self.config.user_id_col].values
        
        if not np.issubdtype(y.dtype, np.number):
            y = pd.to_numeric(y, errors='coerce')
            if np.isnan(y).any():
                raise ValueError(f"无法将目标列 '{self.config.target}' 转换为数值类型")
        
        return X, y, user_ids
    
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
    
    def _create_windows_with_user_ids(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        user_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        创建滑动窗口并保留用户ID信息。
        
        确保每个窗口内的用户ID一致，跳过包含多个用户的窗口。
        
        Args:
            X: 标准化后的特征数据
            y: 标准化后的目标数据
            user_ids: 用户ID数组
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                (窗口化特征, 窗口化目标, 窗口用户ID)
        """
        window_size = self.config.window_size
        horizon = self.config.horizon
        
        X_windows = []
        y_windows = []
        window_user_ids = []
        
        for i in range(len(X) - window_size - horizon + 1):
            window_user_id_slice = user_ids[i:(i + window_size)]
            
            if len(set(window_user_id_slice)) != 1:
                continue
            
            X_windows.append(X[i:(i + window_size)])
            y_windows.append(y[(i + window_size):(i + window_size + horizon)])
            window_user_ids.append(window_user_id_slice[0])
        
        return np.array(X_windows), np.array(y_windows), np.array(window_user_ids)
    
    def _organize_user_data(
        self, 
        X_windows: np.ndarray, 
        y_windows: np.ndarray, 
        window_user_ids: np.ndarray
    ) -> None:
        """
        按用户ID组织窗口化后的数据。
        
        Args:
            X_windows: 窗口化后的特征
            y_windows: 窗口化后的目标
            window_user_ids: 窗口对应的用户ID
        """
        for user_id in self.users:
            mask = window_user_ids == user_id
            user_X = X_windows[mask]
            user_y = y_windows[mask]
            
            if len(user_X) == 0:
                continue
            
            self.user_data[user_id] = {
                'X': user_X,
                'y': user_y,
                'dataset': TimeSeriesDataset(user_X, user_y)
            }
