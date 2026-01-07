"""
融合模型（TCN + DCA + Transformer）

该模块实现了TCN-DCA-Transformer融合模型，
将时间卷积网络、通道注意力和Transformer编码器结合用于时间序列预测。

模型架构：
1. TCN特征提取：使用带扩张卷积的时间卷积网络捕获时序模式
2. 通道注意力(DCA)：自适应调整不同通道的权重
3. Transformer编码：捕获全局依赖关系
4. 全连接输出层：生成多步预测结果
"""

from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn

from .base import BaseModel, ModelConfig
from .tcn import TemporalConvNet, TCNConfig
from .attention import ChannelAttention, ChannelAttentionConfig
from .transformer import TransformerBlock, TransformerConfig


@dataclass
class FusionModelConfig(ModelConfig):
    """
    融合模型配置类。
    
    在基础ModelConfig基础上添加了融合模型特定的配置参数。
    
    Attributes:
        window_size (int): 输入窗口大小
        num_channels (List[int]): TCN通道数列表
        tcn_kernel_size (int): TCN卷积核大小
        dca_reduction_ratio (int): 通道注意力降维比例
        transformer_heads (int): Transformer注意力头数
        transformer_dropout (float): Transformer Dropout比率
        transformer_forward_expansion (int): Transformer前馈网络扩展因子
        transformer_num_layers (int): Transformer块数量
    """
    window_size: int = 300
    num_channels: List[int] = None
    tcn_kernel_size: int = 3
    dca_reduction_ratio: int = 8
    transformer_heads: int = 4
    transformer_dropout: float = 0.1
    transformer_forward_expansion: int = 4
    transformer_num_layers: int = 2
    
    def __post_init__(self) -> None:
        """
        初始化后验证参数。
        """
        super().__post_init__()
        if self.num_channels is None:
            self.num_channels = [64, 128]
        if self.window_size < 1:
            raise ValueError(f"window_size 必须大于0，当前值: {self.window_size}")
        if self.tcn_kernel_size < 2:
            raise ValueError(f"tcn_kernel_size 必须大于等于2，当前值: {self.tcn_kernel_size}")
        if self.dca_reduction_ratio < 1:
            raise ValueError(f"dca_reduction_ratio 必须大于0，当前值: {self.dca_reduction_ratio}")
        if self.transformer_heads < 1:
            raise ValueError(f"transformer_heads 必须大于0，当前值: {self.transformer_heads}")
        if not 0 <= self.transformer_dropout < 1:
            raise ValueError(f"transformer_dropout 必须在[0,1)范围内，当前值: {self.transformer_dropout}")


class TCNDCATransformer(BaseModel):
    """
    TCN-DCA-Transformer融合模型类。
    
    该模型将三种不同的神经网络架构组合用于时间序列预测任务：
    
    1. 时间卷积网络(TCN)：
       - 使用空洞卷积扩大感受野
       - 捕获局部时间模式
       - 输出多尺度特征
    
    2. 通道注意力机制(DCA)：
       - 自适应调整通道权重
       - 增强重要特征，抑制噪声
       - 使用全局池化聚合空间信息
    
    3. Transformer编码器：
       - 捕获长程依赖关系
       - 建模序列中的全局上下文
       - 多头注意力关注不同方面
    
    4. 全连接输出层：
       - 将高维特征映射到预测空间
       - 支持多步预测
    
    Example:
        >>> config = FusionModelConfig(
        ...     input_dim=5,
        ...     output_dim=50,
        ...     window_size=300,
        ...     num_channels=[64, 128],
        ...     transformer_heads=4
        ... )
        >>> model = TCNDCATransformer(config)
        >>> x = torch.randn(32, 300, 5)  # (batch, window_size, features)
        >>> output = model(x)
        >>> output.shape  # torch.Size([32, 50])
    
    Attributes:
        config (FusionModelConfig): 融合模型配置对象
        tcn (TemporalConvNet): 时间卷积网络
        dca (ChannelAttention): 通道注意力机制
        transformer_blocks (nn.ModuleList): Transformer块列表
        fc (nn.Linear): 全连接输出层
    """
    
    def __init__(self, config: FusionModelConfig) -> None:
        """
        初始化TCN-DCA-Transformer融合模型。
        
        Args:
            config: 融合模型配置对象
        """
        self.fusion_config = config
        super().__init__(config)
    
    def _initialize_model(self) -> None:
        """
        初始化融合模型网络结构。
        
        网络结构组成：
        1. 时间卷积网络(TCN)
        2. 通道注意力机制(DCA)
        3. 两个Transformer编码器块
        4. 全连接输出层
        
        输入形状: [batch_size, window_size, input_dim]
        输出形状: [batch_size, output_dim]
        """
        self.tcn = TemporalConvNet(
            TCNConfig(
                input_dim=self.config.input_dim,
                output_dim=self.fusion_config.num_channels[-1],
                num_channels=self.fusion_config.num_channels,
                kernel_size=self.fusion_config.tcn_kernel_size
            )
        )
        
        tcn_output_channels = self.fusion_config.num_channels[-1]
        
        self.dca = ChannelAttention(
            ChannelAttentionConfig(
                input_dim=tcn_output_channels,
                output_dim=tcn_output_channels,
                channels=tcn_output_channels,
                reduction_ratio=self.fusion_config.dca_reduction_ratio
            )
        )
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                TransformerConfig(
                    embed_size=tcn_output_channels,
                    num_heads=self.fusion_config.transformer_heads,
                    dropout=self.fusion_config.transformer_dropout,
                    forward_expansion=self.fusion_config.transformer_forward_expansion
                )
            )
            for _ in range(self.fusion_config.transformer_num_layers)
        ])
        
        fc_input_size = tcn_output_channels * self.fusion_config.window_size
        self.fc = nn.Linear(fc_input_size, self.config.output_dim)
        
        self.is_initialized = True
        
        print(f"TCN-DCA-Transformer 模型初始化完成")
        print(f"  - 输入维度: {self.config.input_dim}")
        print(f"  - 窗口大小: {self.fusion_config.window_size}")
        print(f"  - TCN通道: {self.fusion_config.num_channels}")
        print(f"  - Transformer头数: {self.fusion_config.transformer_heads}")
        print(f"  - 输出维度: {self.config.output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行融合模型前向传播。
        
        输入张量形状: [batch_size, window_size, input_dim]
        输出张量形状: [batch_size, output_dim]
        
        处理流程：
        1. TCN特征提取: [batch, window, input] -> [batch, window, channels]
        2. 通道注意力: [batch, window, channels] -> [batch, window, channels]
        3. Transformer编码: [batch, window, channels] -> [batch, window, channels]
        4. 展平 + 全连接: [batch, window*channels] -> [batch, output]
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 预测输出
        """
        x = x.transpose(1, 2)
        
        x_tcn = self.tcn(x)
        
        attention_weights = self.dca(x_tcn)
        x_dca = x_tcn * attention_weights
        
        x_transformer_input = x_dca.permute(0, 2, 1)
        
        transformer_output = x_transformer_input
        for block in self.transformer_blocks:
            transformer_output = block(
                transformer_output, 
                transformer_output, 
                transformer_output, 
                None
            )
        
        x_flat = transformer_output.reshape(transformer_output.size(0), -1)
        
        if self.fc.in_features != x_flat.shape[1]:
            self._adjust_fc_layer(x_flat.shape[1])
        
        x_out = self.fc(x_flat)
        
        return x_out
    
    def _adjust_fc_layer(self, new_input_size: int) -> None:
        """
        动态调整全连接层的输入大小。
        
        当输入特征维度与预期不符时，重新初始化全连接层。
        
        Args:
            new_input_size: 新的输入特征维度
        """
        print(f"警告: 正在将FC层输入大小从 {self.fc.in_features} 调整为 {new_input_size}")
        self.fc = nn.Linear(new_input_size, self.config.output_dim).to(self.device)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取输入样本的通道注意力权重。
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 通道注意力权重
        """
        self.eval()
        with torch.no_grad():
            x = x.transpose(1, 2)
            x_tcn = self.tcn(x)
            attention_weights = self.dca(x_tcn)
        return attention_weights
    
    def get_transformer_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取Transformer层的输出表示。
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: Transformer输出
        """
        self.eval()
        with torch.no_grad():
            x = x.transpose(1, 2)
            x_tcn = self.tcn(x)
            attention_weights = self.dca(x_tcn)
            x_dca = x_tcn * attention_weights
            x_transformer_input = x_dca.permute(0, 2, 1)
            
            transformer_output = x_transformer_input
            for block in self.transformer_blocks:
                transformer_output = block(
                    transformer_output,
                    transformer_output,
                    transformer_output,
                    None
                )
        return transformer_output
    
    def get_config(self) -> FusionModelConfig:
        """
        获取融合模型配置对象。
        
        Returns:
            FusionModelConfig: 融合模型配置对象
        """
        return self.fusion_config
    
    def print_model_architecture(self) -> None:
        """
        打印融合模型架构详细信息。
        """
        self.print_model_summary()
        print(f"\n{'='*50}")
        print("TCN-DCA-Transformer 模型架构")
        print(f"{'='*50}")
        print(f"1. TCN特征提取器:")
        print(f"   - 输入: [batch, window_size={self.fusion_config.window_size}, input_dim={self.config.input_dim}]")
        print(f"   - 输出: [batch, window_size, channels={self.fusion_config.num_channels[-1]}]")
        print(f"   - 通道数: {self.fusion_config.num_channels}")
        print(f"   - 卷积核大小: {self.fusion_config.tcn_kernel_size}")
        print(f"\n2. 通道注意力(DCA):")
        print(f"   - 输入: [batch, channels, window_size]")
        print(f"   - 输出: [batch, channels, 1] (注意力权重)")
        print(f"   - 降维比例: {self.fusion_config.dca_reduction_ratio}")
        print(f"\n3. Transformer编码器:")
        print(f"   - 输入: [batch, window_size, channels]")
        print(f"   - 输出: [batch, window_size, channels]")
        print(f"   - 头数: {self.fusion_config.transformer_heads}")
        print(f"   - 层数: {self.fusion_config.transformer_num_layers}")
        print(f"   - Dropout: {self.fusion_config.transformer_dropout}")
        print(f"\n4. 全连接输出层:")
        print(f"   - 输入: [batch, window_size * channels]")
        print(f"   - 输出: [batch, output_dim={self.config.output_dim}]")
        print(f"{'='*50}\n")
