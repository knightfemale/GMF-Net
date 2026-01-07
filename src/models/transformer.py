"""
Transformer编码器块实现

该模块实现了Transformer编码器的核心组件，包括：
1. 多头自注意力机制
2. 前馈神经网络
3. 层归一化和残差连接
4. Transformer编码器块

Transformer是一种基于注意力机制的神经网络架构，
最初设计用于自然语言处理任务，现已广泛应用于各种序列建模任务。
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel, ModelConfig


@dataclass
class TransformerConfig(ModelConfig):
    """
    Transformer配置类。
    
    在基础ModelConfig基础上添加了Transformer特定的配置参数。
    
    Attributes:
        embed_size (int): 嵌入维度
        num_heads (int): 注意力头数
        forward_expansion (int): 前馈网络扩展因子
        dropout (float): Dropout比率
        num_layers (int): Transformer块的数量
    """
    embed_size: int = None
    num_heads: int = 8
    forward_expansion: int = 4
    dropout: float = 0.1
    num_layers: int = 2
    
    def __post_init__(self) -> None:
        """
        初始化后验证参数。
        """
        super().__post_init__()
        if self.embed_size is None:
            self.embed_size = self.input_dim
        if self.num_heads < 1:
            raise ValueError(f"num_heads 必须大于0，当前值: {self.num_heads}")
        if self.forward_expansion < 1:
            raise ValueError(f"forward_expansion 必须大于0，当前值: {self.forward_expansion}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout 必须在[0,1)范围内，当前值: {self.dropout}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers 必须大于0，当前值: {self.num_layers}")


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制类。
    
    该类实现了Transformer中的多头自注意力机制。
    通过将注意力计算分成多个头并行执行，可以同时关注不同位置的不同表示子空间的信息。
    
    工作原理：
    1. 将输入投影到查询(Query)、键(Key)、值(Value)三个空间
    2. 将每个头分裂成多个子空间
    3. 在每个子空间计算注意力分数
    4. 拼接所有头的输出并线性变换
    
    Attributes:
        embed_size (int): 嵌入维度
        num_heads (int): 注意力头数
        head_dim (int): 每个头的维度
        qkv (nn.Linear): 查询、键、值的线性投影层
        fc_out (nn.Linear): 输出线性变换层
        dropout (nn.Dropout): Dropout层
    """
    
    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.1) -> None:
        """
        初始化多头自注意力机制。
        
        Args:
            embed_size: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        if embed_size % num_heads != 0:
            raise ValueError(f"embed_size ({embed_size}) 必须能被 num_heads ({num_heads}) 整除")
        
        self.qkv = nn.Linear(embed_size, embed_size * 3)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        执行多头自注意力计算。
        
        输入张量形状: [batch_size, seq_len, embed_size]
        输出张量形状: [batch_size, seq_len, embed_size]
        
        Args:
            query: 查询张量
            key: 键张量
            value: 值张量
            mask: 注意力掩码，None表示不使用掩码
            
        Returns:
            torch.Tensor: 注意力输出
        """
        batch_size = query.size(0)
        
        qkv = self.qkv(query).reshape(batch_size, -1, 3, self.embed_size)
        q, k, v = qkv.chunk(3, dim=2)
        
        q = q.reshape(batch_size, q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, v.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        
        energy = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.embed_size)
        
        return self.fc_out(out)
    
    def get_attention_scores(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        获取注意力分数矩阵（用于可视化）。
        
        Args:
            query: 查询张量
            key: 键张量
            
        Returns:
            torch.Tensor: 注意力分数矩阵
        """
        batch_size = query.size(0)
        
        q = query.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = key.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        energy = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = F.softmax(energy, dim=-1)
        
        return attention


class TransformerBlock(BaseModel):
    """
    Transformer编码器块类。
    
    该类实现了Transformer的一个编码器块，包含：
    1. 多头自注意力层
    2. 前馈神经网络层
    3. 层归一化和残差连接
    
    每个子层都使用残差连接和层归一化：
    Output = LayerNorm(x + Sublayer(x))
    
    Example:
        >>> config = TransformerConfig(
        ...     embed_size=128,
        ...     num_heads=8,
        ...     dropout=0.1,
        ...     num_layers=2
        ... )
        >>> transformer_block = TransformerBlock(config)
        >>> x = torch.randn(32, 100, 128)  # (batch, seq_len, embed_size)
        >>> output = transformer_block(x)
        >>> output.shape  # torch.Size([32, 100, 128])
    
    Attributes:
        config (TransformerConfig): Transformer配置对象
        attention (MultiHeadAttention): 多头自注意力层
        norm1 (nn.LayerNorm): 第一层归一化
        norm2 (nn.LayerNorm): 第二层归一化
        feed_forward (nn.Sequential): 前馈神经网络
        dropout (nn.Dropout): Dropout层
    """
    
    def __init__(self, config: TransformerConfig) -> None:
        """
        初始化Transformer编码器块。
        
        Args:
            config: Transformer配置对象
        """
        self.t_config = config
        super().__init__(config)
    
    def _initialize_model(self) -> None:
        """
        初始化Transformer块结构。
        
        网络结构包括：
        1. 多头自注意力层
        2. 第一层归一化 + 残差连接
        3. 前馈神经网络（两个线性变换 + ReLU激活）
        4. 第二层归一化 + 残差连接
        """
        self.attention = MultiHeadAttention(
            embed_size=self.t_config.embed_size,
            num_heads=self.t_config.num_heads,
            dropout=self.t_config.dropout
        )
        
        self.norm1 = nn.LayerNorm(self.t_config.embed_size)
        self.norm2 = nn.LayerNorm(self.t_config.embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(self.t_config.embed_size, self.t_config.forward_expansion * self.t_config.embed_size),
            nn.ReLU(),
            nn.Linear(self.t_config.forward_expansion * self.t_config.embed_size, self.t_config.embed_size)
        )
        
        self.dropout = nn.Dropout(self.t_config.dropout)
        
        self.is_initialized = True
    
    def forward(
        self, 
        value: torch.Tensor, 
        key: torch.Tensor, 
        query: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        执行Transformer块的前向传播。
        
        输入张量形状: [batch_size, seq_len, embed_size]
        输出张量形状: [batch_size, seq_len, embed_size]
        
        Args:
            value: 值张量
            key: 键张量
            query: 查询张量
            mask: 注意力掩码
            
        Returns:
            torch.Tensor: Transformer块输出
        """
        attention_output = self.attention(query, key, value, mask)
        
        x = self.dropout(self.norm1(attention_output + query))
        
        forward_output = self.feed_forward(x)
        
        out = self.dropout(self.norm2(forward_output + x))
        
        return out


class TransformerEncoder(nn.Module):
    """
    Transformer编码器类。
    
    该类堆叠多个Transformer编码器块，形成完整的Transformer编码器。
    
    Attributes:
        layers (nn.ModuleList): Transformer块列表
        num_layers (int): Transformer块数量
        norm (nn.LayerNorm): 最终归一化层
        dropout (nn.Dropout): Dropout层
    """
    
    def __init__(
        self, 
        embed_size: int, 
        num_heads: int, 
        forward_expansion: int,
        dropout: float, 
        num_layers: int
    ) -> None:
        """
        初始化Transformer编码器。
        
        Args:
            embed_size: 嵌入维度
            num_heads: 注意力头数
            forward_expansion: 前馈网络扩展因子
            dropout: Dropout比率
            num_layers: Transformer块数量
        """
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                TransformerConfig(
                    embed_size=embed_size,
                    num_heads=num_heads,
                    forward_expansion=forward_expansion,
                    dropout=dropout
                )
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        执行Transformer编码器的前向传播。
        
        Args:
            x: 输入序列
            mask: 注意力掩码
            
        Returns:
            torch.Tensor: 编码后的序列表示
        """
        for layer in self.layers:
            x = layer(x, x, x, mask)
        
        return self.norm(x)
