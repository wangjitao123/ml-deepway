"""
DataPreprocessor单元测试
测试异常值检测、归一化和滑动窗口功能
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """DataPreprocessor功能测试类"""

    def setUp(self):
        """测试前初始化：生成正态分布测试数据"""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 22
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.y = np.random.randint(0, 8, self.n_samples)
        self.seq_len = 30

    def test_zscore_normalization(self):
        """测试ZScore标准化：输出应接近均值0、标准差1"""
        preprocessor = DataPreprocessor(method="zscore", seq_len=self.seq_len)
        preprocessor.fit(self.X)
        X_norm = preprocessor.transform(self.X)

        # 均值接近0
        self.assertAlmostEqual(float(np.abs(np.mean(X_norm)).mean()), 0.0, places=0)
        # 检查形状不变
        self.assertEqual(X_norm.shape, self.X.shape)

    def test_minmax_normalization(self):
        """测试MinMax归一化：输出应在[0, 1]范围内"""
        preprocessor = DataPreprocessor(method="minmax", seq_len=self.seq_len)
        X_norm = preprocessor.fit_transform(self.X)

        self.assertTrue(np.all(X_norm >= -0.01))   # 允许极小的浮点误差
        self.assertTrue(np.all(X_norm <= 1.01))
        self.assertEqual(X_norm.shape, self.X.shape)

    def test_outlier_detection_iqr(self):
        """测试IQR异常值检测：注入极端异常值后应被裁剪"""
        preprocessor = DataPreprocessor(method="zscore", seq_len=self.seq_len)
        preprocessor.fit(self.X)

        # 注入极端异常值
        X_outlier = self.X.copy()
        X_outlier[0, 0] = 1000.0    # 极大异常值
        X_outlier[1, 0] = -1000.0   # 极小异常值

        X_transformed = preprocessor.transform(X_outlier)

        # 变换后异常值应被限制
        self.assertLess(np.abs(X_transformed[0, 0]), 100.0)
        self.assertLess(np.abs(X_transformed[1, 0]), 100.0)

    def test_sliding_window(self):
        """测试滑动窗口序列生成：输出形状应正确"""
        preprocessor = DataPreprocessor(method="zscore", seq_len=self.seq_len)
        X_norm = preprocessor.fit_transform(self.X)
        X_seq, y_seq = preprocessor.create_sequences(X_norm, self.y)

        expected_n_windows = self.n_samples - self.seq_len + 1
        self.assertEqual(X_seq.shape[0], expected_n_windows)
        self.assertEqual(X_seq.shape[1], self.seq_len)
        self.assertEqual(X_seq.shape[2], self.n_features)
        self.assertEqual(len(y_seq), expected_n_windows)

    def test_sliding_window_label_alignment(self):
        """测试滑动窗口标签对齐：窗口标签应为窗口末尾的标签"""
        preprocessor = DataPreprocessor(method="zscore", seq_len=5)
        X_small = np.random.randn(20, 4).astype(np.float32)
        y_small = np.arange(20)

        preprocessor.fit(X_small)
        X_seq, y_seq = preprocessor.create_sequences(X_small, y_small)

        # 第一个窗口的标签应等于索引4（seq_len - 1 = 4）
        self.assertEqual(y_seq[0], 4)
        # 第二个窗口的标签应等于索引5
        self.assertEqual(y_seq[1], 5)

    def test_fit_transform_consistency(self):
        """测试fit_transform与分开调用的一致性"""
        preprocessor1 = DataPreprocessor(method="zscore", seq_len=self.seq_len)
        X1 = preprocessor1.fit_transform(self.X)

        preprocessor2 = DataPreprocessor(method="zscore", seq_len=self.seq_len)
        preprocessor2.fit(self.X)
        X2 = preprocessor2.transform(self.X)

        np.testing.assert_array_almost_equal(X1, X2, decimal=5)

    def test_not_fitted_raises_error(self):
        """测试未fit时调用transform应抛出RuntimeError"""
        preprocessor = DataPreprocessor(method="zscore", seq_len=self.seq_len)
        with self.assertRaises(RuntimeError):
            preprocessor.transform(self.X)

    def test_insufficient_samples_raises_error(self):
        """测试样本数不足时应抛出ValueError"""
        preprocessor = DataPreprocessor(method="zscore", seq_len=50)
        X_small = np.random.randn(10, self.n_features).astype(np.float32)
        preprocessor.fit(X_small)
        with self.assertRaises(ValueError):
            preprocessor.create_sequences(X_small, np.zeros(10, dtype=np.int64))


if __name__ == "__main__":
    unittest.main(verbosity=2)
