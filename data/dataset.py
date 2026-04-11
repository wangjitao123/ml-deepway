"""
车辆故障数据集模块
提供 PyTorch Dataset 和 DataLoader 的封装实现
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional


class VehicleFaultDataset(Dataset):
    """
    车辆故障预测数据集
    封装时序传感器序列、故障标签和严重程度
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        severity: Optional[np.ndarray] = None,
    ):
        """
        初始化数据集
        :param X: 传感器序列，形状 (N, seq_len, n_features)
        :param y: 故障类别标签，形状 (N,)
        :param severity: 故障严重程度，范围 0~1，形状 (N,)；若为None则按标签自动生成
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        if severity is not None:
            self.severity = torch.tensor(severity, dtype=torch.float32)
        else:
            # 根据标签自动生成严重程度：正常=0，其他故障线性分配
            sev = np.where(y == 0, 0.0, 0.3 + 0.1 * (y - 1))
            self.severity = torch.tensor(sev, dtype=torch.float32)

    def __len__(self) -> int:
        """返回数据集样本数量"""
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        :return: (传感器序列, 故障标签, 严重程度)
        """
        return self.X[idx], self.y[idx], self.severity[idx]


def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    severity: Optional[np.ndarray] = None,
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练/验证/测试数据加载器
    :param X: 特征序列 (N, seq_len, n_features)
    :param y: 标签 (N,)
    :param severity: 严重程度 (N,)
    :param batch_size: 批次大小
    :param train_ratio: 训练集比例 (默认0.70)
    :param val_ratio: 验证集比例 (默认0.15)，测试集为剩余部分
    :param num_workers: DataLoader工作进程数
    :param random_seed: 随机种子，保证可重复性
    :return: (train_loader, val_loader, test_loader)
    """
    # 构建完整数据集
    dataset = VehicleFaultDataset(X, y, severity)
    total = len(dataset)

    # 按比例计算各分割大小（确保和为total）
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    # 确保每个分割至少有一个样本
    if test_size <= 0:
        test_size = 1
        val_size = max(1, total - train_size - test_size)
        train_size = total - val_size - test_size

    # 使用固定随机种子拆分数据集
    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # 创建 DataLoader 实例
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,                    # 训练集打乱顺序
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,                   # 验证集不打乱
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,                   # 测试集不打乱
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
