"""
对抗训练模块
通过生成对抗样本提升模型鲁棒性
"""

import logging
import torch
import torch.nn as nn
from typing import Tuple

logger = logging.getLogger(__name__)


class AdversarialTrainer:
    """
    对抗训练器
    通过多种扰动方式生成对抗样本，提高模型对传感器噪声的鲁棒性
    """

    def __init__(
        self,
        noise_std: float = 0.01,      # 高斯噪声标准差
        dropout_rate: float = 0.1,    # 特征丢弃率
        drift_scale: float = 0.05,    # 系统漂移幅度
        spike_prob: float = 0.01,     # 随机尖峰概率
        spike_scale: float = 3.0,     # 尖峰幅度（倍标准差）
    ):
        """
        初始化对抗训练器
        :param noise_std: 高斯噪声标准差（相对于信号幅度）
        :param dropout_rate: 随机特征丢弃比例
        :param drift_scale: 传感器系统性漂移幅度
        :param spike_prob: 每个点出现尖峰的概率
        :param spike_scale: 尖峰幅度缩放因子
        """
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate
        self.drift_scale = drift_scale
        self.spike_prob = spike_prob
        self.spike_scale = spike_scale

    def add_noise(self, x: torch.Tensor, std: float = None) -> torch.Tensor:
        """
        添加高斯白噪声（模拟传感器测量误差）
        :param x: 输入张量 (B, seq_len, features)
        :param std: 噪声标准差（None时使用初始化值）
        :return: 添加噪声后的张量
        """
        if std is None:
            std = self.noise_std
        noise = torch.randn_like(x) * std
        return x + noise

    def add_dropout(self, x: torch.Tensor, rate: float = None) -> torch.Tensor:
        """
        随机置零特征值（模拟传感器断线或数据丢失）
        :param x: 输入张量 (B, seq_len, features)
        :param rate: 丢弃率（None时使用初始化值）
        :return: 特征随机丢弃后的张量
        """
        if rate is None:
            rate = self.dropout_rate
        # 生成随机mask，为1的位置保留，为0的位置清零
        mask = torch.bernoulli(torch.ones_like(x) * (1.0 - rate))
        return x * mask

    def add_drift(self, x: torch.Tensor, drift: float = None) -> torch.Tensor:
        """
        添加系统性漂移（模拟传感器老化或校准偏差）
        :param x: 输入张量 (B, seq_len, features)
        :param drift: 漂移幅度（None时使用初始化值）
        :return: 添加漂移后的张量
        """
        if drift is None:
            drift = self.drift_scale
        # 时间线性漂移：越靠后的时间步漂移越大
        batch_size, seq_len, n_features = x.shape
        time_steps = torch.linspace(0, 1, seq_len, device=x.device)
        # (seq_len,) → (1, seq_len, 1) 广播到整个批次
        drift_pattern = (time_steps.unsqueeze(0).unsqueeze(-1) * drift)
        return x + drift_pattern

    def add_spike(self, x: torch.Tensor, prob: float = None, scale: float = None) -> torch.Tensor:
        """
        添加随机尖峰（模拟传感器瞬间干扰）
        :param x: 输入张量 (B, seq_len, features)
        :param prob: 尖峰出现概率
        :param scale: 尖峰幅度缩放
        :return: 添加尖峰后的张量
        """
        if prob is None:
            prob = self.spike_prob
        if scale is None:
            scale = self.spike_scale
        # 生成稀疏尖峰：以prob概率触发，触发时幅度为scale
        spike_mask = torch.bernoulli(torch.ones_like(x) * prob)
        spike_values = torch.randn_like(x) * scale
        return x + spike_mask * spike_values

    def generate_adversarial_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        strategy: str = "random",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成对抗样本批次
        :param x: 原始输入张量 (B, seq_len, features)
        :param y: 标签张量 (B,)
        :param strategy: 扰动策略 ('random'/'all'/'noise'/'dropout')
        :return: (扰动后的张量, 原始标签)
        """
        x_perturbed = x.clone()

        if strategy == "all":
            # 依次应用所有扰动类型
            x_perturbed = self.add_noise(x_perturbed)
            x_perturbed = self.add_dropout(x_perturbed)
            x_perturbed = self.add_drift(x_perturbed)
            x_perturbed = self.add_spike(x_perturbed)

        elif strategy == "noise":
            x_perturbed = self.add_noise(x_perturbed)

        elif strategy == "dropout":
            x_perturbed = self.add_dropout(x_perturbed)

        elif strategy == "random":
            # 随机选择一种扰动
            import random
            strategies = ["noise", "dropout", "drift", "spike"]
            chosen = random.choice(strategies)
            if chosen == "noise":
                x_perturbed = self.add_noise(x_perturbed)
            elif chosen == "dropout":
                x_perturbed = self.add_dropout(x_perturbed)
            elif chosen == "drift":
                x_perturbed = self.add_drift(x_perturbed)
            elif chosen == "spike":
                x_perturbed = self.add_spike(x_perturbed)

        return x_perturbed, y

    def adversarial_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module = None,
    ) -> torch.Tensor:
        """
        计算对抗样本上的分类损失
        :param model: 待评估模型
        :param x: 原始输入 (B, seq_len, features)
        :param y: 真实标签 (B,)
        :param criterion: 损失函数（None时使用交叉熵）
        :return: 标量损失张量
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # 生成对抗样本
        x_adv, y_adv = self.generate_adversarial_batch(x, y, strategy="random")

        # 前向传播（保持梯度图）
        outputs = model(x_adv)
        loss = criterion(outputs["fault_logits"], y_adv)

        return loss
