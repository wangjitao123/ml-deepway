"""
贝叶斯超参数优化模块
使用Optuna框架搜索最优超参数组合
"""

import logging
import sys
import os
from typing import Dict, Any, Optional, Callable

# Optuna为可选依赖
try:
    import optuna
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    MedianPruner = None
    OPTUNA_AVAILABLE = False
    logging.warning("optuna 未安装，贝叶斯优化不可用")

import torch
import numpy as np

logger = logging.getLogger(__name__)


class BayesianOptimizer:
    """
    贝叶斯超参数优化器
    使用Optuna的TPE算法搜索最优超参数
    """

    def __init__(
        self,
        train_loader,
        val_loader,
        n_trials: int = 50,
        study_name: str = "vehicle_fault_hpo",
        direction: str = "maximize",    # 最大化验证准确率
        pruning: bool = True,
        seed: int = 42,
    ):
        """
        初始化贝叶斯优化器
        :param train_loader: 训练数据加载器
        :param val_loader: 验证数据加载器
        :param n_trials: 超参数搜索次数
        :param study_name: Optuna研究名称
        :param direction: 优化方向（'maximize'/'minimize'）
        :param pruning: 是否启用剪枝（提前终止差的trial）
        :param seed: 随机种子
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_trials = n_trials
        self.study_name = study_name
        self.direction = direction
        self.pruning = pruning
        self.seed = seed

        self.study = None        # Optuna研究对象
        self.best_params = None  # 最优超参数

    def objective(self, trial) -> float:
        """
        Optuna目标函数：训练少量轮次，返回验证准确率
        :param trial: Optuna trial对象
        :return: 验证准确率（优化目标）
        """
        # 定义超参数搜索空间
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        # 确保num_heads能整除hidden_dim
        if hidden_dim % num_heads != 0:
            num_heads = 4

        # 动态导入，避免循环引用
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from models.lstm_transformer import LSTMTransformerModel
            from models.trainer import FaultTrainer
        except ImportError:
            logger.error("无法导入模型模块")
            return 0.0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 构建模型
        model = LSTMTransformerModel(
            input_dim=22,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_classes=8,
            seq_len=30,
            dropout=dropout,
        )

        # 创建训练器（快速评估只训练几轮）
        trainer = FaultTrainer(
            model=model,
            device=device,
            learning_rate=lr,
            weight_decay=weight_decay,
            use_tensorboard=False,
            checkpoint_dir="checkpoints_hpo",
        )

        best_val_acc = 0.0
        # 快速评估：只训练5轮
        for epoch in range(5):
            train_metrics = trainer.train_epoch(self.train_loader)
            val_metrics = trainer.validate(self.val_loader)
            val_acc = val_metrics["val_accuracy"]

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            # 报告中间值给Optuna（用于剪枝）
            trial.report(val_acc, epoch)

            # 如果当前trial表现差，提前终止
            if trial.should_prune():
                import optuna
                raise optuna.exceptions.TrialPruned()

        return best_val_acc

    def optimize(self) -> Dict[str, Any]:
        """
        运行超参数优化
        :return: 最优超参数字典
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("optuna未安装，返回默认超参数")
            return self._default_params()

        # 禁用optuna的冗余日志
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # 创建Optuna研究
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2) if self.pruning else None

        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            pruner=pruner,
            sampler=optuna.samplers.TPESampler(seed=self.seed),
        )

        logger.info(f"开始贝叶斯超参数优化，共 {self.n_trials} 次试验...")

        try:
            self.study.optimize(
                self.objective,
                n_trials=self.n_trials,
                show_progress_bar=False,
                catch=(Exception,),
            )

            self.best_params = self.study.best_params
            logger.info(f"最优超参数: {self.best_params}")
            logger.info(f"最优验证准确率: {self.study.best_value:.4f}")

        except Exception as e:
            logger.error(f"超参数优化失败: {e}")
            self.best_params = self._default_params()

        return self.best_params or self._default_params()

    def _default_params(self) -> Dict[str, Any]:
        """返回默认超参数"""
        return {
            "learning_rate": 0.001,
            "hidden_dim": 128,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.2,
            "weight_decay": 1e-4,
        }

    def get_best_params(self) -> Dict[str, Any]:
        """
        获取最优超参数
        :return: 最优超参数字典
        """
        if self.best_params is not None:
            return self.best_params
        return self._default_params()

    def print_summary(self):
        """打印优化结果摘要"""
        if self.study is None or not OPTUNA_AVAILABLE:
            logger.info("无优化结果可展示")
            return

        logger.info(f"\n{'='*50}")
        logger.info(f"超参数优化完成摘要")
        logger.info(f"总试验次数: {len(self.study.trials)}")
        logger.info(f"最优试验编号: {self.study.best_trial.number}")
        logger.info(f"最优验证准确率: {self.study.best_value:.4f}")
        logger.info(f"最优超参数:")
        for key, val in self.study.best_params.items():
            logger.info(f"  {key}: {val}")
        logger.info(f"{'='*50}")
