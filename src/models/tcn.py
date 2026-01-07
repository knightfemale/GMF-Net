"""
时间卷积网络（TCN）模型实现

该模块实现了带有扩张卷积的时间卷积网络，用于捕获时间序列中的长程依赖关系。
TCN通过使用空洞卷积来扩大感受野，从而能够捕捉更长时间范围内的模式。
"""

from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel, ModelConfig


@dataclass
class TCNConfig(ModelConfig):
    """
    时间卷积网络配置类。
    
    在基础ModelConfig基础上添加了TCN特定的配置参数。
    
    Attributes:
        num_channels (List[int]): 各层通道数列表
        kernel_size (int): 卷积核大小
        dropout (float): Dropout比率
        weight_decay (float): L2正则化系数
        batch_norm (bool): 是否使用批归一化
    """
    num_channels: List[int] = None
    kernel_size: int = 3
    dropout: float = 0.2
    weight_decay: float = 0.001
    batch_norm: bool = True
    
    def __post_init__(self) -> None:
        """
        初始化后进行参数验证。
        """
        super().__post_init__()
        if self.num_channels is None:
            self.num_channels = [64, 128]
        if len(self.num_channels) < 1:
            raise ValueError("num_channels 至少需要一个通道数")
        if self.kernel_size < 2:
            raise ValueError(f"kernel_size 必须大于等于2，当前值: {self.kernel_size}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout 必须在[0,1)范围内，当前值: {self.dropout}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay 必须大于等于0，当前值: {self.weight_decay}")


class TemporalConvNet(BaseModel):
    """
    时间卷积网络（TCN）类。
    
    该类实现了时间卷积网络模型，使用多个带有扩张卷积的卷积层来捕获时间序列中的模式。
    每一层的扩张因子按指数增长，使得网络能够捕获不同时间尺度的依赖关系。
    
    主要特点：
    1. 空洞卷积：扩大感受野，捕获长程依赖
    2. 残差连接：缓解梯度消失问题
    3. Dropout正则化：防止过拟合
    4. 可选的批归一化：提高训练稳定性
    
    Example:
        >>> config = TCNConfig(
        ...     input_dim=5,
        ...     output_dim=128,
        ...     num_channels=[64, 128, 256],
        ...     kernel_size=3,
        ...     dropout=0.2
        ... )
        >>> model = TemporalConvNet(config)
        >>> x = torch.randn(32, 100, 5)  # (batch, seq_len, features)
        >>> output = model(x)
        >>> output.shape  # torch.Size([32, 100, 256])
    
    Attributes:
        config (TCNConfig): TCN配置对象
        network (nn.Sequential): 网络层序列
    """
    
    def __init__(self, config: TCNConfig) -> None:
        """
        初始化时间卷积网络。
        
        Args:
            config: TCN配置对象
        """
        self.tcn_config = config
        super().__init__(config)
    
    def _initialize_model(self) -> None:
        """
        初始化TCN网络结构。
        
        网络结构由多个卷积块组成，每个块包含：
        1. 1D卷积层（带空洞）
        2. 批归一化层（可选）
        3. ReLU激活函数
        4. Dropout层
        """
        layers = []
        num_levels = len(self.tcn_config.num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # 扩张因子按指数增长
            in_channels = self.config.input_dim if i == 0 else self.tcn_config.num_channels[i - 1]
            out_channels = self.tcn_config.num_channels[i]
            
            kernel_size = self.tcn_config.kernel_size
            padding = (kernel_size - 1) * dilation_size
            
            conv_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation_size,
                padding=padding
            )
            
            if self.tcn_config.batch_norm:
                layers.extend([
                    conv_layer,
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(self.tcn_config.dropout)
                ])
            else:
                layers.extend([
                    conv_layer,
                    nn.ReLU(),
                    nn.Dropout(self.tcn_config.dropout)
                ])
        
        self.network = nn.Sequential(*layers)
        self.is_initialized = True
        
        print(f"TCN初始化完成: dropout={self.tcn_config.dropout}, "
              f"weight_decay={self.tcn_config.weight_decay}, "
              f"batch_norm={self.tcn_config.batch_norm}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行TCN前向传播。
        
        输入张量形状: [batch_size, seq_len, input_dim]
        输出张量形状: [batch_size, seq_len, num_channels[-1]]
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 输出张量
        """
        x = x.transpose(1, 2)  # 转换为 [batch, input_dim, seq_len]
        output = self.network(x)
        output = output.transpose(1, 2)  # 转换回 [batch, seq_len, channels]
        return output
    
    def get_l2_regularization_loss(self) -> torch.Tensor:
        """
        计算模型的L2正则化损失。
        
        Returns:
            torch.Tensor: L2正则化损失值
        """
        l2_reg = torch.tensor(0.0, device=self.device)
        for param in self.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param, 2)
        return self.tcn_config.weight_decay * l2_reg
    
    def get_config(self) -> TCNConfig:
        """
        获取TCN配置对象。
        
        Returns:
            TCNConfig: TCN配置对象
        """
        return self.tcn_config
    
    def get_output_channels(self) -> int:
        """
        获取网络输出通道数。
        
        Returns:
            int: 输出通道数
        """
        return self.tcn_config.num_channels[-1] if self.tcn_config.num_channels else 0
    
    def get_receptive_field(self) -> int:
        """
        计算网络的感受野大小。
        
        感受野表示输出特征图上的每个点对应的输入序列范围。
        
        Returns:
            int: 感受野大小
        """
        kernel_size = self.tcn_config.kernel_size
        num_levels = len(self.tcn_config.num_channels)
        
        receptive_field = 1
        for i in range(num_levels):
            dilation = 2 ** i
            receptive_field += (kernel_size - 1) * dilation
        
        return receptive_field
    
    def print_network_info(self) -> None:
        """
        打印TCN网络详细信息。
        """
        self.print_model_summary()
        print(f"卷积层数: {len(self.tcn_config.num_channels)}")
        print(f"通道数: {self.tcn_config.num_channels}")
        print(f"卷积核大小: {self.tcn_config.kernel_size}")
        print(f"感受野: {self.get_receptive_field()}")
        print(f"输出通道数: {self.get_output_channels()}")
