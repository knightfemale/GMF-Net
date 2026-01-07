"""
模型模块

该模块提供了神经网络模型的实现，包括：
- 基础模型架构
- 时间卷积网络 (TCN)
- 通道注意力机制 (DCA)
- Transformer编码器
- 融合模型 (TCN + DCA + Transformer)
"""

from .base import BaseModel, ModelConfig
from .tcn import TemporalConvNet, TCNConfig
from .attention import ChannelAttention, ChannelAttentionConfig
from .transformer import TransformerBlock, TransformerConfig
from .fusion import TCNDCATransformer, FusionModelConfig

__all__ = [
    'BaseModel',
    'ModelConfig',
    'TemporalConvNet',
    'TCNConfig',
    'ChannelAttention',
    'ChannelAttentionConfig',
    'TransformerBlock',
    'TransformerConfig',
    'TCNDCATransformer',
    'FusionModelConfig'
]
