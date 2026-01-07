"""
通道注意力机制（DCA）实现

该模块实现了通道注意力机制（Dual Channel Attention），
通过自适应调整不同通道的权重来增强模型的特征表示能力。

通道注意力机制的核心思想是：
1. 使用全局平均池化和全局最大池化来聚合空间信息
2. 通过共享的全连接层学习通道间的依赖关系
3. 使用Sigmoid函数生成通道权重
4. 将权重应用于原始特征，实现自适应特征重标定
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel, ModelConfig


@dataclass
class ChannelAttentionConfig(ModelConfig):
    """
    通道注意力机制配置类。
    
    Attributes:
        channels (int): 输入通道数
        reduction_ratio (int): 降维比例，用于减少全连接层的参数数量
    """
    channels: int = None
    reduction_ratio: int = 16
    
    def __post_init__(self) -> None:
        """
        初始化后验证参数。
        """
        super().__post_init__()
        if self.channels is None:
            self.channels = self.input_dim
        if self.channels < 1:
            raise ValueError(f"channels 必须大于0，当前值: {self.channels}")
        if self.reduction_ratio < 1:
            raise ValueError(f"reduction_ratio 必须大于0，当前值: {self.reduction_ratio}")


class ChannelAttention(BaseModel):
    """
    通道注意力机制类。
    
    该类实现了通道注意力机制（也称为挤压-激励网络SE-Net的简化版本）。
    通过学习每个通道的重要性权重，自适应地调整特征图的通道权重。
    
    工作原理：
    1. 对输入特征分别进行全局平均池化和全局最大池化，得到两个通道描述向量
    2. 将两个描述向量输入共享的全连接层进行特征变换
    3. 将变换后的两个向量相加，通过Sigmoid函数生成通道权重
    4. 将权重应用于原始输入特征，实现通道重标定
    
    Example:
        >>> config = ChannelAttentionConfig(
        ...     channels=128,
        ...     reduction_ratio=16
        ... )
        >>> attention = ChannelAttention(config)
        >>> x = torch.randn(32, 128, 100)  # (batch, channels, seq_len)
        >>> weights = attention(x)
        >>> # weights.shape = (32, 128, 1)
        >>> output = x * weights
    
    Attributes:
        config (ChannelAttentionConfig): 通道注意力配置对象
        avg_pool (nn.AdaptiveAvgPool1d): 全局平均池化层
        max_pool (nn.AdaptiveMaxPool1d): 全局最大池化层
        fc (nn.Sequential): 共享全连接层
    """
    
    def __init__(self, config: ChannelAttentionConfig) -> None:
        """
        初始化通道注意力机制。
        
        Args:
            config: 通道注意力配置对象
        """
        self.ca_config = config
        super().__init__(config)
    
    def _initialize_model(self) -> None:
        """
        初始化通道注意力网络结构。
        
        网络结构包括：
        1. 全局平均池化层：将输入压缩为每个通道一个值
        2. 全局最大池化层：捕获每个通道的最大响应
        3. 共享全连接层：降低维度后恢复，学习通道间关系
        4. Sigmoid激活：生成归一化的通道权重
        """
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        reduced_channels = max(1, self.ca_config.channels // self.ca_config.reduction_ratio)
        
        self.fc = nn.Sequential(
            nn.Linear(self.ca_config.channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, self.ca_config.channels, bias=False),
            nn.Sigmoid()
        )
        
        self.is_initialized = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行通道注意力前向传播。
        
        输入张量形状: [batch_size, channels, seq_len]
        输出张量形状: [batch_size, channels, 1]（通道权重）
        
        Args:
            x: 输入特征图
            
        Returns:
            torch.Tensor: 通道注意力权重
        """
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        
        channel_attention_weights = self.fc(avg_out + max_out)
        
        return channel_attention_weights.unsqueeze(-1)
    
    def apply_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        将通道注意力权重应用于输入特征。
        
        Args:
            x: 输入特征图，形状为 [batch_size, channels, seq_len]
            
        Returns:
            torch.Tensor: 加权后的特征图
        """
        attention_weights = self.forward(x)
        return x * attention_weights
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取输入特征对应的通道注意力权重。
        
        Args:
            x: 输入特征图
            
        Returns:
            torch.Tensor: 通道注意力权重
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class SEBlock(nn.Module):
    """
    简化的Squeeze-and-Excitation（SE）注意力模块。
    
    这是通道注意力机制的一个简化版本，主要用于特征重标定。
    
    Attributes:
        fc (nn.Sequential): 特征变换层
        reduction (int): 降维比例
    """
    
    def __init__(self, channels: int, reduction: int = 16) -> None:
        """
        初始化SE模块。
        
        Args:
            channels: 输入通道数
            reduction: 降维比例
        """
        super().__init__()
        self.reduction = reduction
        reduced_channels = max(1, channels // reduction)
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行SE注意力计算。
        
        Args:
            x: 输入特征图
            
        Returns:
            torch.Tensor: 加权后的特征图
        """
        weights = self.fc(x).unsqueeze(-1)
        return x * weights


class ECAChannelAttention(nn.Module):
    """
    高效通道注意力（ECA）模块。
    
    ECA是通道注意力机制的高效变体，相比原始SE模块：
    1. 避免了降维带来的信息损失
    2. 使用1D卷积替代全连接层，减少参数数量
    3. 保持了通道间的局部交互
    
    Attributes:
        kernel_size (int): 1D卷积核大小
        gamma (float): 核大小计算参数
        b (float): 核大小偏移参数
        conv (nn.Conv1d): 1D卷积层
    """
    
    def __init__(self, channels: int, gamma: float = 2, b: float = 1) -> None:
        """
        初始化ECA注意力模块。
        
        Args:
            channels: 输入通道数
            gamma: 核大小计算参数
            b: 核大小偏移参数
        """
        super().__init__()
        kernel_size = int(abs((channels * gamma + b) / gamma) // 2 * 2 + 1)
        kernel_size = max(kernel_size, 3)
        self.kernel_size = kernel_size
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行ECA注意力计算。
        
        Args:
            x: 输入特征图
            
        Returns:
            torch.Tensor: 加权后的特征图
        """
        y = self.avg_pool(x)
        y = self.conv(y.unsqueeze(1)).squeeze(1)
        return self.sigmoid(y).unsqueeze(-1).expand_as(x)
