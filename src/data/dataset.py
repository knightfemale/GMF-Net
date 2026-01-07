"""
数据集管理器

该模块提供了数据集的统一管理和配置功能，
支持多种数据集的加载和切换。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass
class DatasetInfo:
    """
    数据集信息类，用于描述数据集的元信息。
    
    Attributes:
        name (str): 数据集名称
        file_path (str): 数据文件路径
        features (List[str]): 特征列名列表
        target (str): 目标列名
        format_type (str): 数据格式类型 ('csv' 或 'mat')
        description (str): 数据集描述
    """
    name: str
    file_path: str
    features: List[str]
    target: str
    format_type: str = 'csv'
    description: str = ''
    
    def __post_init__(self) -> None:
        """
        初始化后验证数据格式。
        """
        if self.format_type not in ['csv', 'mat']:
            raise ValueError(f"不支持的数据格式: {self.format_type}，仅支持 'csv' 或 'mat'")


class DatasetManager:
    """
    数据集管理器类。
    
    该类提供了数据集的统一管理和配置功能，
    支持注册、加载和切换多个数据集。
    
    Example:
        >>> manager = DatasetManager()
        >>> manager.register_dataset(DatasetInfo(
        ...     name='train_data',
        ...     file_path='data/train.csv',
        ...     features=['f1', 'f2', 'f3'],
        ...     target='target'
        ... ))
        >>> config = manager.get_dataset_config('train_data')
    
    Attributes:
        datasets (Dict[str, DatasetInfo]): 已注册的数据集字典
        default_dataset (Optional[str]): 默认数据集名称
    """
    
    def __init__(self) -> None:
        """
        初始化数据集管理器。
        """
        self.datasets: Dict[str, DatasetInfo] = {}
        self.default_dataset: Optional[str] = None
    
    def register_dataset(self, dataset_info: DatasetInfo) -> None:
        """
        注册一个新的数据集。
        
        Args:
            dataset_info: 数据集信息对象
        """
        self.datasets[dataset_info.name] = dataset_info
        if self.default_dataset is None:
            self.default_dataset = dataset_info.name
    
    def unregister_dataset(self, name: str) -> None:
        """
        注销指定名称的数据集。
        
        Args:
            name: 数据集名称
            
        Raises:
            KeyError: 如果数据集不存在
        """
        if name not in self.datasets:
            raise KeyError(f"数据集 '{name}' 不存在")
        
        del self.datasets[name]
        
        if self.default_dataset == name:
            self.default_dataset = next(iter(self.datasets)) if self.datasets else None
    
    def get_dataset_config(self, name: Optional[str] = None) -> Optional[DatasetInfo]:
        """
        获取指定名称的数据集配置。
        
        Args:
            name: 数据集名称，默认为默认数据集
            
        Returns:
            DatasetInfo: 数据集信息对象，不存在返回None
        """
        if name is None:
            name = self.default_dataset
        
        return self.datasets.get(name) if name else None
    
    def set_default_dataset(self, name: str) -> None:
        """
        设置默认数据集。
        
        Args:
            name: 数据集名称
            
        Raises:
            KeyError: 如果数据集不存在
        """
        if name not in self.datasets:
            raise KeyError(f"数据集 '{name}' 不存在")
        
        self.default_dataset = name
    
    def list_datasets(self) -> List[str]:
        """
        列出所有已注册的数据集名称。
        
        Returns:
            List[str]: 数据集名称列表
        """
        return list(self.datasets.keys())
    
    def check_dataset_exists(self, name: str) -> bool:
        """
        检查指定数据集是否存在。
        
        Args:
            name: 数据集名称
            
        Returns:
            bool: 数据集是否存在
        """
        return name in self.datasets
    
    def validate_dataset_file(self, name: str) -> Tuple[bool, str]:
        """
        验证数据集文件是否存在且可访问。
        
        Args:
            name: 数据集名称
            
        Returns:
            Tuple[bool, str]: (是否有效, 消息)
        """
        dataset_info = self.get_dataset_config(name)
        if dataset_info is None:
            return False, f"数据集 '{name}' 不存在"
        
        file_path = Path(dataset_info.file_path)
        if not file_path.exists():
            return False, f"数据文件不存在: {file_path}"
        
        return True, "数据集文件有效"
    
    def load_data(self, name: Optional[str] = None) -> Tuple[pd.DataFrame, DatasetInfo]:
        """
        加载指定数据集的数据。
        
        Args:
            name: 数据集名称，默认为默认数据集
            
        Returns:
            Tuple[pd.DataFrame, DatasetInfo]: (加载的数据, 数据集信息)
            
        Raises:
            ValueError: 如果数据集不存在或文件无效
        """
        dataset_info = self.get_dataset_config(name)
        if dataset_info is None:
            raise ValueError(f"数据集 '{name}' 不存在")
        
        is_valid, message = self.validate_dataset_file(name)
        if not is_valid:
            raise ValueError(message)
        
        if dataset_info.format_type == 'csv':
            data = pd.read_csv(dataset_info.file_path)
        else:
            from scipy.io import loadmat
            mat_data = loadmat(dataset_info.file_path)
            data = pd.DataFrame(mat_data.get('fea', []))
        
        return data, dataset_info
    
    def get_features_target(self, name: Optional[str] = None) -> Tuple[List[str], str]:
        """
        获取指定数据集的特征列名和目标列名。
        
        Args:
            name: 数据集名称，默认为默认数据集
            
        Returns:
            Tuple[List[str], str]: (特征列名列表, 目标列名)
        """
        dataset_info = self.get_dataset_config(name)
        if dataset_info is None:
            raise ValueError(f"数据集 '{name}' 不存在")
        
        return dataset_info.features, dataset_info.target
