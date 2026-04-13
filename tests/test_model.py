"""
LSTMTransformerModel单元测试
测试模型前向传播输出形状和键值
"""

import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_transformer import LSTMTransformerModel, PositionalEncoding


class TestPositionalEncoding(unittest.TestCase):
    """位置编码测试"""

    def test_output_shape(self):
        """测试位置编码输出形状不变"""
        pe = PositionalEncoding(d_model=64, max_len=100)
        x = torch.randn(4, 30, 64)
        out = pe(x)
        self.assertEqual(out.shape, x.shape)

    def test_different_seq_lengths(self):
        """测试不同序列长度的位置编码"""
        pe = PositionalEncoding(d_model=64, max_len=200)
        for seq_len in [1, 10, 30, 50]:
            x = torch.randn(2, seq_len, 64)
            out = pe(x)
            self.assertEqual(out.shape, (2, seq_len, 64))


class TestLSTMTransformerModel(unittest.TestCase):
    """LSTMTransformerModel功能测试类"""

    def setUp(self):
        """初始化测试配置"""
        self.input_dim = 22
        self.hidden_dim = 64    # 使用小维度加速测试
        self.num_layers = 1
        self.num_heads = 4
        self.num_classes = 8
        self.seq_len = 30
        self.batch_size = 4

        self.model = LSTMTransformerModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_classes=self.num_classes,
            seq_len=self.seq_len,
            dropout=0.0,    # 测试时关闭dropout
        )
        self.model.eval()

    def test_forward_pass_keys(self):
        """测试前向传播输出包含所有必要键"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        with torch.no_grad():
            outputs = self.model(x)

        required_keys = ["fault_logits", "fault_probs", "severity", "feature_repr"]
        for key in required_keys:
            self.assertIn(key, outputs, f"输出缺少键: {key}")

    def test_fault_logits_shape(self):
        """测试故障分类logits输出形状"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        with torch.no_grad():
            outputs = self.model(x)
        self.assertEqual(
            outputs["fault_logits"].shape,
            (self.batch_size, self.num_classes)
        )

    def test_fault_probs_sum_to_one(self):
        """测试故障概率输出之和应为1（softmax输出）"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        with torch.no_grad():
            outputs = self.model(x)
        probs_sum = outputs["fault_probs"].sum(dim=-1)
        torch.testing.assert_close(
            probs_sum,
            torch.ones(self.batch_size),
            atol=1e-5, rtol=1e-5,
        )

    def test_severity_range(self):
        """测试严重程度输出在[0, 1]范围内（Sigmoid输出）"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        with torch.no_grad():
            outputs = self.model(x)
        severity = outputs["severity"]
        self.assertTrue(torch.all(severity >= 0.0))
        self.assertTrue(torch.all(severity <= 1.0))

    def test_feature_repr_shape(self):
        """测试特征表示输出形状"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        with torch.no_grad():
            outputs = self.model(x)
        self.assertEqual(
            outputs["feature_repr"].shape,
            (self.batch_size, self.hidden_dim)
        )

    def test_batch_size_one(self):
        """测试批次大小为1时的正常工作"""
        x = torch.randn(1, self.seq_len, self.input_dim)
        with torch.no_grad():
            outputs = self.model(x)
        self.assertEqual(outputs["fault_logits"].shape[0], 1)

    def test_gradient_flow(self):
        """测试反向传播梯度能正常流动"""
        self.model.train()
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        y = torch.randint(0, self.num_classes, (self.batch_size,))

        outputs = self.model(x)
        loss = torch.nn.CrossEntropyLoss()(outputs["fault_logits"], y)
        loss.backward()

        # 检查至少有一个参数的梯度不为None
        has_gradient = any(
            p.grad is not None
            for p in self.model.parameters()
            if p.requires_grad
        )
        self.assertTrue(has_gradient, "模型参数应有梯度")

    def test_model_parameter_count(self):
        """测试模型参数量在合理范围内"""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 10000, "模型参数量太少")
        self.assertLess(total_params, 50_000_000, "模型参数量过大")


if __name__ == "__main__":
    unittest.main(verbosity=2)
