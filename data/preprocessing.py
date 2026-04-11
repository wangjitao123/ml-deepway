"""
数据预处理模块
提供异常值检测、标准化和滑动窗口序列生成功能
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class DataPreprocessor:
    """
    车辆传感器数据预处理器
    提供异常值检测、归一化及时序窗口切割功能
    """

    def __init__(self, method: str = "zscore", seq_len: int = 30):
        """
        初始化预处理器
        :param method: 归一化方法，支持 'zscore' 和 'minmax'
        :param seq_len: 滑动窗口长度
        """
        self.method = method          # 归一化方法
        self.seq_len = seq_len        # 时序窗口长度
        self.is_fitted = False        # 是否已拟合

        # zscore 所需统计量
        self.mean_ = None
        self.std_ = None

        # minmax 所需统计量
        self.min_ = None
        self.max_ = None

        # IQR 异常值裁剪边界
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def _detect_outliers_iqr(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用四分位距(IQR)检测异常值边界
        :param X: 输入特征矩阵 (N, D)
        :return: 下界和上界数组
        """
        Q1 = np.percentile(X, 25, axis=0)   # 第一四分位数
        Q3 = np.percentile(X, 75, axis=0)   # 第三四分位数
        IQR = Q3 - Q1                         # 四分位距

        # 计算裁剪边界，系数1.5为标准IQR规则
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return lower, upper

    def fit(self, X: np.ndarray) -> "DataPreprocessor":
        """
        在训练数据上拟合预处理参数
        :param X: 训练数据矩阵 (N, D)
        :return: 自身实例（支持链式调用）
        """
        X = np.array(X, dtype=np.float32)

        # 计算IQR异常值边界
        self.lower_bounds_, self.upper_bounds_ = self._detect_outliers_iqr(X)

        # 先裁剪异常值，再计算统计量
        X_clipped = np.clip(X, self.lower_bounds_, self.upper_bounds_)

        if self.method == "zscore":
            # 计算均值和标准差
            self.mean_ = np.mean(X_clipped, axis=0)
            self.std_ = np.std(X_clipped, axis=0)
            # 避免除以零
            self.std_ = np.where(self.std_ == 0, 1.0, self.std_)

        elif self.method == "minmax":
            # 计算最小值和最大值
            self.min_ = np.min(X_clipped, axis=0)
            self.max_ = np.max(X_clipped, axis=0)
            # 避免除以零
            diff = self.max_ - self.min_
            diff = np.where(diff == 0, 1.0, diff)
            self.max_ = self.min_ + diff

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        对数据应用预处理变换
        :param X: 输入特征矩阵 (N, D)
        :return: 预处理后的特征矩阵
        """
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit() 方法进行拟合")

        X = np.array(X, dtype=np.float32)

        # 第一步：IQR异常值裁剪
        X = np.clip(X, self.lower_bounds_, self.upper_bounds_)

        # 第二步：归一化
        if self.method == "zscore":
            X = (X - self.mean_) / self.std_
        elif self.method == "minmax":
            X = (X - self.min_) / (self.max_ - self.min_)

        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        拟合并变换数据（合并操作）
        :param X: 输入特征矩阵 (N, D)
        :return: 预处理后的特征矩阵
        """
        return self.fit(X).transform(X)

    def create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建滑动窗口时序序列
        :param X: 特征矩阵 (N, D)
        :param y: 标签数组 (N,)
        :return: 序列矩阵 (N-seq_len, seq_len, D) 和标签 (N-seq_len,)
        """
        N, D = X.shape
        if N < self.seq_len:
            raise ValueError(f"样本数 {N} 小于窗口长度 {self.seq_len}")

        n_windows = N - self.seq_len + 1   # 窗口数量

        # 预分配内存以提高效率
        X_seq = np.zeros((n_windows, self.seq_len, D), dtype=np.float32)
        y_seq = np.zeros(n_windows, dtype=np.int64)

        for i in range(n_windows):
            X_seq[i] = X[i: i + self.seq_len]      # 提取窗口特征
            y_seq[i] = y[i + self.seq_len - 1]      # 取窗口末尾标签

        return X_seq, y_seq

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        逆变换，将归一化数据还原
        :param X: 归一化后的特征矩阵
        :return: 原始尺度的特征矩阵
        """
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit() 方法进行拟合")

        X = np.array(X, dtype=np.float32)

        if self.method == "zscore":
            X = X * self.std_ + self.mean_
        elif self.method == "minmax":
            X = X * (self.max_ - self.min_) + self.min_

        return X
