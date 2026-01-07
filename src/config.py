"""
配置文件

该模块提供了项目的默认配置参数，
包括模型配置、训练配置、聚类配置等。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os


@dataclass
class TCNModelConfig:
    """
    TCN模型默认配置。
    """
    num_channels: List[int] = field(default_factory=lambda: [64, 128])
    kernel_size: int = 3
    dropout: float = 0.2
    weight_decay: float = 0.001
    batch_norm: bool = True


@dataclass
class AttentionConfig:
    """
    通道注意力机制默认配置。
    """
    reduction_ratio: int = 8


@dataclass
class TransformerConfig:
    """
    Transformer编码器默认配置。
    """
    num_heads: int = 4
    dropout: float = 0.1
    forward_expansion: int = 4
    num_layers: int = 2


@dataclass
class FusionModelConfig:
    """
    融合模型默认配置。
    """
    window_size: int = 300
    horizon: int = 50
    tcn: TCNModelConfig = field(default_factory=TCNModelConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)


@dataclass
class TrainingConfig:
    """
    训练过程默认配置。
    """
    num_epochs: int = 20
    learning_rate: float = 0.01
    batch_size: int = 32
    test_size: float = 0.2
    random_seed: int = 42
    device: str = 'cuda'
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-6


@dataclass
class ClusteringConfig:
    """
    粒度球聚类默认配置。
    """
    min_points_for_split: int = 8
    min_points_after_split: int = 4
    max_iterations: int = 50
    pca_components: int = 2
    normalize_range: tuple = (0, 1)


@dataclass
class OptimizationConfig:
    """
    BKOA优化器默认配置。
    """
    pop_size: int = 20
    max_iter: int = 50
    timeout: float = 300
    alpha: float = 0.8
    beta: float = 0.5
    gamma: float = 0.1
    early_stopping: bool = True
    early_stopping_threshold: float = 1e-6
    early_stopping_patience: int = 5


@dataclass
class DataConfig:
    """
    数据路径默认配置。
    """
    data_directory: str = "data"
    results_directory: str = "results"
    dataset_prefix: str = "data_"


class Config:
    """
    项目配置单例类。
    
    该类整合所有配置，提供统一的配置访问接口。
    
    Example:
        >>> config = Config()
        >>> config.training.learning_rate
        0.01
        >>> config.fusion.window_size
        300
    """
    
    _instance: Optional['Config'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.training = TrainingConfig()
        self.fusion = FusionModelConfig()
        self.clustering = ClusteringConfig()
        self.optimization = OptimizationConfig()
        self.data = DataConfig()
        
        self._initialized = True
    
    def get_tcn_params(self) -> Dict[str, Any]:
        """
        获取TCN模型参数字典。
        
        Returns:
            Dict[str, Any]: TCN参数
        """
        return {
            'num_channels': self.fusion.tcn.num_channels,
            'kernel_size': self.fusion.tcn.kernel_size,
            'dropout': self.fusion.tcn.dropout,
            'weight_decay': self.fusion.tcn.weight_decay
        }
    
    def get_transformer_params(self) -> Dict[str, Any]:
        """
        获取Transformer参数字典。
        
        Returns:
            Dict[str, Any]: Transformer参数
        """
        return {
            'num_heads': self.fusion.transformer.num_heads,
            'dropout': self.fusion.transformer.dropout,
            'forward_expansion': self.fusion.transformer.forward_expansion,
            'num_layers': self.fusion.transformer.num_layers
        }
    
    def get_optimization_params(self) -> Dict[str, Any]:
        """
        获取优化器参数字典。
        
        Returns:
            Dict[str, Any]: 优化器参数
        """
        return {
            'pop_size': self.optimization.pop_size,
            'max_iter': self.optimization.max_iter,
            'timeout': self.optimization.timeout,
            'alpha': self.optimization.alpha,
            'beta': self.optimization.beta,
            'gamma': self.optimization.gamma
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        从字典更新配置。
        
        Args:
            config_dict: 配置字典
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    current_obj = getattr(self, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(current_obj, sub_key):
                            setattr(current_obj, sub_key, sub_value)
                else:
                    setattr(self, key, value)
    
    def save_config(self, path: str) -> None:
        """
        保存配置到文件。
        
        Args:
            path: 保存路径
        """
        import json
        
        config_dict = {
            'training': self.training.__dict__,
            'fusion': {
                'window_size': self.fusion.window_size,
                'horizon': self.fusion.horizon,
                'tcn': self.fusion.tcn.__dict__,
                'attention': self.fusion.attention.__dict__,
                'transformer': self.fusion.transformer.__dict__
            },
            'clustering': self.clustering.__dict__,
            'optimization': self.optimization.__dict__,
            'data': self.data.__dict__
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"配置已保存到: {path}")
    
    def load_config(self, path: str) -> None:
        """
        从文件加载配置。
        
        Args:
            path: 配置文件路径
        """
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        self.update_from_dict(config_dict)
        print(f"配置已从 {path} 加载")


def get_default_config() -> Config:
    """
    获取默认配置单例。
    
    Returns:
        Config: 默认配置对象
    """
    return Config()


def reset_config() -> None:
    """
    重置配置为默认值。
    """
    Config._instance = None
