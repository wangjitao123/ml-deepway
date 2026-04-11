"""
模型训练器模块
包含早停机制、混合损失函数和完整训练流程
"""

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Tuple

# TensorBoard为可选依赖
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    早停机制
    当验证集指标在patience轮内不改善时停止训练
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        """
        初始化早停
        :param patience: 容忍的无改善轮次
        :param min_delta: 最小改善阈值
        :param mode: 'min' 表示越小越好（如loss），'max' 表示越大越好（如accuracy）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0          # 无改善计数器
        self.best_value = None    # 最优值
        self.should_stop = False  # 是否应该停止

    def __call__(self, value: float) -> bool:
        """
        检查是否应该早停
        :param value: 当前轮次的监控指标值
        :return: True表示应该停止训练
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


class FaultTrainer:
    """
    车辆故障预测模型训练器
    支持混合损失函数、学习率调度和TensorBoard日志
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        physics_loss_fn=None,      # 物理约束损失函数（可选）
        checkpoint_dir: str = "checkpoints",
        tensorboard_dir: str = "runs",
        use_tensorboard: bool = True,
    ):
        """
        初始化训练器
        :param model: 待训练的PyTorch模型
        :param device: 计算设备（None时自动检测GPU/CPU）
        :param learning_rate: 学习率
        :param weight_decay: L2正则化权重衰减
        :param physics_loss_fn: 物理约束损失函数实例
        :param checkpoint_dir: 检查点保存目录
        :param tensorboard_dir: TensorBoard日志目录
        :param use_tensorboard: 是否使用TensorBoard
        """
        # 自动检测计算设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = model.to(self.device)
        self.physics_loss_fn = physics_loss_fn
        self.checkpoint_dir = checkpoint_dir

        # 确保检查点目录存在
        os.makedirs(checkpoint_dir, exist_ok=True)

        # AdamW优化器（比Adam更好的权重衰减）
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # 余弦退火学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )

        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss()       # 交叉熵：分类损失
        self.mse_loss = nn.MSELoss()               # 均方误差：严重程度损失

        # TensorBoard日志
        self.writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                os.makedirs(tensorboard_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=tensorboard_dir)
                logger.info(f"TensorBoard已启动，日志目录: {tensorboard_dir}")
            except Exception as e:
                logger.warning(f"TensorBoard启动失败: {e}")

        logger.info(f"训练器初始化完成，使用设备: {self.device}")

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        执行一个训练轮次
        :param train_loader: 训练数据加载器
        :return: 包含各损失指标的字典
        """
        self.model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_mse_loss = 0.0
        total_phys_loss = 0.0
        total_adv_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (X, y, severity) in enumerate(train_loader):
            # 移动数据到计算设备
            X = X.to(self.device)
            y = y.to(self.device)
            severity = severity.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(X)
            fault_logits = outputs["fault_logits"]
            pred_severity = outputs["severity"].squeeze(-1)

            # ── 损失计算 ──

            # 1. 分类交叉熵损失
            ce = self.ce_loss(fault_logits, y)

            # 2. 严重程度MSE损失
            mse = self.mse_loss(pred_severity, severity)

            # 3. 物理约束损失（若有）
            if self.physics_loss_fn is not None:
                try:
                    phys = self.physics_loss_fn(outputs, X)
                except Exception:
                    phys = torch.tensor(0.0, device=self.device)
            else:
                phys = torch.tensor(0.0, device=self.device)

            # 4. 对抗噪声损失（添加轻微高斯噪声后的预测稳定性）
            noise = torch.randn_like(X) * 0.01
            X_noisy = X + noise
            outputs_noisy = self.model(X_noisy)
            adv = self.ce_loss(outputs_noisy["fault_logits"], y)

            # 5. 组合损失（加权求和）
            loss = ce + 0.1 * mse + 0.05 * phys + 0.05 * adv

            # 反向传播和参数更新
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 统计指标
            total_loss += loss.item()
            total_ce_loss += ce.item()
            total_mse_loss += mse.item()
            total_phys_loss += phys.item()
            total_adv_loss += adv.item()

            # 分类准确率
            pred_classes = fault_logits.argmax(dim=1)
            correct += (pred_classes == y).sum().item()
            total += y.size(0)

        n_batches = max(len(train_loader), 1)
        return {
            "loss": total_loss / n_batches,
            "ce_loss": total_ce_loss / n_batches,
            "mse_loss": total_mse_loss / n_batches,
            "phys_loss": total_phys_loss / n_batches,
            "adv_loss": total_adv_loss / n_batches,
            "accuracy": correct / max(total, 1),
        }

    def validate(self, val_loader) -> Dict[str, float]:
        """
        在验证集上评估模型
        :param val_loader: 验证数据加载器
        :return: 包含验证指标的字典
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y, severity in val_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                severity = severity.to(self.device)

                outputs = self.model(X)
                fault_logits = outputs["fault_logits"]
                pred_severity = outputs["severity"].squeeze(-1)

                # 计算验证损失
                ce = self.ce_loss(fault_logits, y)
                mse = self.mse_loss(pred_severity, severity)
                loss = ce + 0.1 * mse

                total_loss += loss.item()
                pred_classes = fault_logits.argmax(dim=1)
                correct += (pred_classes == y).sum().item()
                total += y.size(0)

        n_batches = max(len(val_loader), 1)
        return {
            "val_loss": total_loss / n_batches,
            "val_accuracy": correct / max(total, 1),
        }

    def save_checkpoint(self, epoch: int, metrics: dict, filename: str = None):
        """
        保存训练检查点
        :param epoch: 当前轮次
        :param metrics: 指标字典
        :param filename: 文件名（None时自动生成）
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pth"

        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
        }, path)
        logger.info(f"检查点已保存: {path}")

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        """
        加载检查点
        :param checkpoint_path: 检查点文件路径
        :return: 检查点信息字典
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(f"检查点已加载: {checkpoint_path}")
        return checkpoint

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        save_best: bool = True,
    ) -> Dict[str, list]:
        """
        完整训练流程
        :param train_loader: 训练数据加载器
        :param val_loader: 验证数据加载器
        :param epochs: 总训练轮次
        :param early_stopping_patience: 早停耐心值
        :param save_best: 是否保存最优检查点
        :return: 训练历史字典
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode="min")
        history = {
            "train_loss": [], "train_accuracy": [],
            "val_loss": [], "val_accuracy": [],
        }
        best_val_loss = float("inf")
        start_time = time.time()

        logger.info(f"开始训练，共 {epochs} 轮，设备: {self.device}")

        for epoch in range(1, epochs + 1):
            # 训练一轮
            train_metrics = self.train_epoch(train_loader)
            # 验证
            val_metrics = self.validate(val_loader)
            # 更新学习率
            self.scheduler.step()

            # 记录历史
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_accuracy"].append(val_metrics["val_accuracy"])

            # TensorBoard记录
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
                self.writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
                self.writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
                self.writer.add_scalar("Accuracy/val", val_metrics["val_accuracy"], epoch)

            # 打印训练进度
            elapsed = time.time() - start_time
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch [{epoch:3d}/{epochs}] "
                f"Loss: {train_metrics['loss']:.4f} "
                f"Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} "
                f"Val Acc: {val_metrics['val_accuracy']:.4f} | "
                f"LR: {lr:.6f} | "
                f"时间: {elapsed:.1f}s"
            )

            # 保存最优检查点
            if save_best and val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                all_metrics = {**train_metrics, **val_metrics}
                self.save_checkpoint(epoch, all_metrics, filename="best_model.pth")

            # 早停检查
            if early_stopping(val_metrics["val_loss"]):
                logger.info(f"触发早停机制，在第 {epoch} 轮停止训练")
                break

        if self.writer is not None:
            self.writer.close()

        logger.info(f"训练完成！最优验证损失: {best_val_loss:.4f}")
        return history
