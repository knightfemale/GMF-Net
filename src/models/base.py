"""
基础模型抽象类

该模块定义了神经网络模型的抽象基类，提供了通用的模型接口。
所有具体的模型都应该继承此类并实现必要的方法。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """
    模型配置类，用于存储模型的通用参数。
    
    Attributes:
        input_dim (int): 输入特征维度
        output_dim (int): 输出维度
        device (str): 计算设备 ('cuda' 或 'cpu')
        random_seed (int): 随机种子
    """
    input_dim: int
    output_dim: int
    device: str = 'cuda'
    random_seed: int = 42
    
    def __post_init__(self) -> None:
        """
        初始化后验证配置参数。
        """
        if self.input_dim < 1:
            raise ValueError(f"input_dim 必须大于0，当前值: {self.input_dim}")
        if self.output_dim < 1:
            raise ValueError(f"output_dim 必须大于0，当前值: {self.output_dim}")
        if self.device not in ['cuda', 'cpu']:
            raise ValueError(f"device 必须是 'cuda' 或 'cpu'，当前值: {self.device}")


class BaseModel(nn.Module, ABC):
    """
    基础模型抽象类。
    
    该类定义了神经网络模型的通用接口，所有具体的模型都需要继承此类。
    提供了模型初始化、前向传播、参数管理等功能。
    
    Attributes:
        config (ModelConfig): 模型配置对象
        device (torch.device): 计算设备
        is_initialized (bool): 模型是否已初始化
    """
    
    def __init__(self, config: ModelConfig) -> None:
        """
        初始化基础模型。
        
        Args:
            config: 模型配置对象
        """
        super().__init__()
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu'
        )
        self.is_initialized = False
        
        self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self) -> None:
        """
        初始化模型结构。
        
        子类需要实现此方法来定义模型的网络结构。
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义模型的前向传播逻辑。
        
        Args:
            x: 输入张量，形状为 [batch_size, input_dim, ...]
            
        Returns:
            torch.Tensor: 模型输出，形状为 [batch_size, output_dim, ...]
        """
        pass
    
    def get_device(self) -> torch.device:
        """
        获取模型所在设备。
        
        Returns:
            torch.device: 计算设备
        """
        return self.device
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息字典。
        
        Returns:
            Dict[str, Any]: 包含模型信息的字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': self.__class__.__name__,
            'input_dim': self.config.input_dim,
            'output_dim': self.config.output_dim,
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'is_initialized': self.is_initialized
        }
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        统计模型参数数量。
        
        Returns:
            Tuple[int, int]: (总参数数量, 可训练参数数量)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def print_model_summary(self) -> None:
        """
        打印模型结构摘要信息。
        """
        info = self.get_model_info()
        print(f"\n{'='*50}")
        print(f"模型类型: {info['model_type']}")
        print(f"输入维度: {info['input_dim']}")
        print(f"输出维度: {info['output_dim']}")
        print(f"运行设备: {info['device']}")
        print(f"总参数数量: {info['total_parameters']:,}")
        print(f"可训练参数: {info['trainable_parameters']:,}")
        print(f"{'='*50}\n")
    
    def save_model(self, path: str) -> None:
        """
        保存模型权重到文件。
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_info': self.get_model_info()
        }, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path: str, map_location: Optional[str] = None) -> None:
        """
        从文件加载模型权重。
        
        Args:
            path: 模型文件路径
            map_device: 设备映射，默认为None
        """
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已从 {path} 加载")
    
    def freeze_layers(self, layer_names: Optional[List[str]] = None) -> None:
        """
        冻结指定层的参数，使其不可训练。
        
        Args:
            layer_names: 要冻结的层名称列表，None表示冻结所有层
        """
        if layer_names is None:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
    
    def unfreeze_layers(self, layer_names: Optional[List[str]] = None) -> None:
        """
        解冻指定层的参数，使其可训练。
        
        Args:
            layer_names: 要解冻的层名称列表，None表示解冻所有层
        """
        if layer_names is None:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for name, param in self.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
    
    def get_trainable_params(self) -> List[torch.Tensor]:
        """
        获取所有可训练的参数张量列表。
        
        Returns:
            List[torch.Tensor]: 可训练参数列表
        """
        return [param for param in self.parameters() if param.requires_grad]
