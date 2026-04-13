"""
LSTM-Transformer混合故障预测模型
结合LSTM的时序建模能力和Transformer的注意力机制
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    为Transformer提供位置信息，使模型感知序列中各token的位置
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        """
        初始化位置编码
        :param d_model: 模型维度（与嵌入维度一致）
        :param max_len: 支持的最大序列长度
        :param dropout: Dropout概率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 构建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # 频率项：以10000^(2i/d_model)为底
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        # 偶数维度使用sin，奇数维度使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        # 注册为buffer（不参与梯度计算，但随模型保存）
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将位置编码加到输入上
        :param x: 输入张量 (B, seq_len, d_model)
        :return: 加入位置编码后的张量
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LSTMTransformerModel(nn.Module):
    """
    LSTM-Transformer混合故障预测模型
    架构: LSTM → 投影线性层 → 位置编码 → TransformerEncoder → 双输出头
    - 分类头：预测故障类别（num_classes个类）
    - 回归头：预测故障严重程度（0~1）
    """

    def __init__(
        self,
        input_dim: int = 22,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        num_classes: int = 8,
        seq_len: int = 30,
        dropout: float = 0.2,
    ):
        """
        初始化模型
        :param input_dim: 输入特征维度（传感器数量）
        :param hidden_dim: 隐藏层维度
        :param num_layers: LSTM和Transformer的层数
        :param num_heads: 多头注意力头数
        :param num_classes: 故障类别数
        :param seq_len: 时序窗口长度
        :param dropout: Dropout概率
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # ── 第一阶段：LSTM时序特征提取 ──
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,        # 输入格式 (B, seq, features)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # ── 第二阶段：投影到Transformer维度 ──
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.proj_norm = nn.LayerNorm(hidden_dim)

        # ── 第三阶段：位置编码 ──
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len + 10, dropout=dropout)

        # ── 第四阶段：Transformer编码器 ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,        # 输入格式 (B, seq, d_model)
            norm_first=True,         # Pre-LayerNorm更稳定
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ── 输出头 ──
        # 分类头：故障类型预测
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # 回归头：故障严重程度预测（0~1）
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),           # 输出范围限制在0~1
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """使用Kaiming初始化线性层权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        :param x: 传感器时序输入 (B, seq_len, input_dim)
        :return: 包含以下键的字典:
                 - fault_logits: 故障类型logits (B, num_classes)
                 - fault_probs: 故障类型概率 (B, num_classes)
                 - severity: 故障严重程度 (B, 1)
                 - feature_repr: 特征表示 (B, hidden_dim)
        """
        # LSTM前向传播，获取所有时间步的隐藏状态
        lstm_out, _ = self.lstm(x)  # (B, seq_len, hidden_dim)

        # 线性投影 + LayerNorm
        proj_out = self.proj_norm(self.projection(lstm_out))  # (B, seq_len, hidden_dim)

        # 添加位置编码
        pos_out = self.pos_encoding(proj_out)  # (B, seq_len, hidden_dim)

        # Transformer编码器（全序列注意力）
        transformer_out = self.transformer_encoder(pos_out)  # (B, seq_len, hidden_dim)

        # 取最后一个时间步作为序列表示
        feature_repr = transformer_out[:, -1, :]  # (B, hidden_dim)

        # 分类头：故障类型
        fault_logits = self.classifier(feature_repr)           # (B, num_classes)
        fault_probs = F.softmax(fault_logits, dim=-1)          # (B, num_classes)

        # 回归头：故障严重程度
        severity = self.severity_head(feature_repr)            # (B, 1)

        return {
            "fault_logits": fault_logits,
            "fault_probs": fault_probs,
            "severity": severity,
            "feature_repr": feature_repr,
        }
