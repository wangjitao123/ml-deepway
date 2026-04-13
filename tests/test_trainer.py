"""
FaultTrainer单元测试
测试单个训练步骤和验证步骤
"""

import unittest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_transformer import LSTMTransformerModel
from models.trainer import FaultTrainer, EarlyStopping
from data.dataset import VehicleFaultDataset
from torch.utils.data import DataLoader


class TestEarlyStopping(unittest.TestCase):
    """早停机制测试"""

    def test_stops_when_no_improvement(self):
        """超过patience轮无改善时应触发停止"""
        es = EarlyStopping(patience=3, mode="min")
        self.assertFalse(es(1.0))
        self.assertFalse(es(1.0))  # 无改善
        self.assertFalse(es(1.0))  # 无改善
        self.assertTrue(es(1.0))   # 第3次无改善，应停止

    def test_resets_on_improvement(self):
        """有改善时计数器应重置"""
        es = EarlyStopping(patience=2, mode="min")
        es(1.0)
        es(0.9)  # 改善，计数器重置
        self.assertFalse(es(0.9))  # 第1次无改善
        self.assertTrue(es(0.9))   # 第2次无改善，停止

    def test_max_mode(self):
        """最大化模式（如准确率）的早停"""
        es = EarlyStopping(patience=2, mode="max")
        self.assertFalse(es(0.8))
        self.assertFalse(es(0.85))  # 改善
        self.assertFalse(es(0.85))  # 第1次无改善
        self.assertTrue(es(0.85))   # 第2次无改善，停止

    def test_reset(self):
        """重置后状态应恢复初始"""
        es = EarlyStopping(patience=2, mode="min")
        es(1.0)
        es(1.0)
        es.reset()
        self.assertIsNone(es.best_value)
        self.assertEqual(es.counter, 0)
        self.assertFalse(es.should_stop)


class TestFaultTrainer(unittest.TestCase):
    """FaultTrainer功能测试"""

    def setUp(self):
        """创建小规模测试用模型和数据"""
        self.input_dim = 22
        self.seq_len = 10       # 使用短序列加速测试
        self.batch_size = 8
        self.num_classes = 8

        # 构建小型测试模型
        self.model = LSTMTransformerModel(
            input_dim=self.input_dim,
            hidden_dim=32,          # 小隐藏层
            num_layers=1,
            num_heads=4,
            num_classes=self.num_classes,
            seq_len=self.seq_len,
            dropout=0.0,
        )

        # 创建测试DataLoader
        n_samples = 64
        X = np.random.randn(n_samples, self.seq_len, self.input_dim).astype(np.float32)
        y = np.random.randint(0, self.num_classes, n_samples)
        severity = np.random.uniform(0, 1, n_samples).astype(np.float32)

        dataset = VehicleFaultDataset(X, y, severity)
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # 创建训练器
        self.trainer = FaultTrainer(
            model=self.model,
            device=torch.device("cpu"),
            learning_rate=0.001,
            checkpoint_dir="test_checkpoints",
            use_tensorboard=False,
        )

    def tearDown(self):
        """清理测试检查点文件"""
        import shutil
        if os.path.exists("test_checkpoints"):
            shutil.rmtree("test_checkpoints")

    def test_train_epoch_returns_metrics(self):
        """训练一轮后应返回包含必要键的指标字典"""
        metrics = self.trainer.train_epoch(self.train_loader)

        required_keys = ["loss", "accuracy", "ce_loss"]
        for key in required_keys:
            self.assertIn(key, metrics, f"训练指标缺少键: {key}")

    def test_train_epoch_loss_is_positive(self):
        """训练损失应为正数"""
        metrics = self.trainer.train_epoch(self.train_loader)
        self.assertGreater(metrics["loss"], 0.0)

    def test_train_epoch_accuracy_in_range(self):
        """准确率应在[0, 1]范围内"""
        metrics = self.trainer.train_epoch(self.train_loader)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_validate_returns_metrics(self):
        """验证应返回包含必要键的指标字典"""
        val_metrics = self.trainer.validate(self.val_loader)

        self.assertIn("val_loss", val_metrics)
        self.assertIn("val_accuracy", val_metrics)

    def test_save_and_load_checkpoint(self):
        """测试检查点保存和加载"""
        # 训练一步
        self.trainer.train_epoch(self.train_loader)

        # 保存检查点
        self.trainer.save_checkpoint(1, {"test": True}, filename="test_ckpt.pth")

        ckpt_path = os.path.join("test_checkpoints", "test_ckpt.pth")
        self.assertTrue(os.path.exists(ckpt_path), "检查点文件应存在")

        # 加载检查点
        checkpoint = self.trainer.load_checkpoint(ckpt_path)
        self.assertIn("model_state_dict", checkpoint)
        self.assertEqual(checkpoint["epoch"], 1)

    def test_model_updates_after_training(self):
        """训练后模型参数应更新"""
        # 记录初始参数
        initial_params = [p.clone() for p in self.model.parameters()]

        # 训练一步
        self.trainer.train_epoch(self.train_loader)

        # 检查参数是否更新
        updated = any(
            not torch.equal(initial, current)
            for initial, current in zip(initial_params, self.model.parameters())
        )
        self.assertTrue(updated, "训练后模型参数应发生变化")

    def test_short_training_loop(self):
        """测试完整训练循环（少量轮次）"""
        history = self.trainer.train(
            self.train_loader,
            self.val_loader,
            epochs=3,
            early_stopping_patience=5,
            save_best=False,
        )

        self.assertIn("train_loss", history)
        self.assertIn("val_accuracy", history)
        self.assertEqual(len(history["train_loss"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
