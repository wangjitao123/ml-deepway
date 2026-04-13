"""
模型评估模块
提供全面的故障预测性能评估和可视化功能
"""

import os
import json
import logging
import numpy as np
import torch
from typing import Dict, List, Optional

# matplotlib和sklearn为必要依赖
import matplotlib
matplotlib.use("Agg")   # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

logger = logging.getLogger(__name__)

# 故障类别名称（中文）
FAULT_CLASS_NAMES = [
    "正常",          # 0
    "发动机过热",     # 1
    "电池异常",       # 2
    "制动失效",       # 3
    "轮胎气压异常",   # 4
    "电机/控制器故障", # 5
    "润滑系统故障",   # 6
    "冷却系统故障",   # 7
]


class FaultEvaluator:
    """
    车辆故障预测评估器
    计算分类指标、绘制混淆矩阵和ROC曲线
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        output_dir: str = "evaluation_results",
        num_classes: int = 8,
    ):
        """
        初始化评估器
        :param model: 已训练的PyTorch模型
        :param device: 计算设备
        :param output_dir: 评估结果输出目录
        :param num_classes: 类别数量
        """
        self.model = model
        self.device = device or torch.device("cpu")
        self.output_dir = output_dir
        self.num_classes = num_classes

        os.makedirs(output_dir, exist_ok=True)

    def evaluate(self, test_loader) -> Dict:
        """
        在测试集上进行全面评估
        :param test_loader: 测试数据加载器
        :return: 包含所有评估指标的字典
        """
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for X, y, severity in test_loader:
                X = X.to(self.device)
                outputs = self.model(X)

                probs = outputs["fault_probs"].cpu().numpy()
                preds = outputs["fault_logits"].argmax(dim=1).cpu().numpy()
                labels = y.numpy()

                all_labels.extend(labels)
                all_preds.extend(preds)
                all_probs.extend(probs)

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        # 计算各项指标
        metrics = self._compute_metrics(all_labels, all_preds, all_probs)

        # 存储预测结果用于可视化
        self._last_labels = all_labels
        self._last_preds = all_preds
        self._last_probs = all_probs

        return metrics

    def _compute_metrics(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        probs: np.ndarray,
    ) -> Dict:
        """
        计算分类评估指标
        :param labels: 真实标签
        :param preds: 预测标签
        :param probs: 预测概率矩阵
        :return: 指标字典
        """
        accuracy = accuracy_score(labels, preds)

        # 宏平均指标（各类别等权重）
        precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
        recall_macro = recall_score(labels, preds, average="macro", zero_division=0)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)

        # 加权平均指标（按类别样本数加权）
        precision_weighted = precision_score(labels, preds, average="weighted", zero_division=0)
        recall_weighted = recall_score(labels, preds, average="weighted", zero_division=0)
        f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

        # 混淆矩阵
        cm = confusion_matrix(labels, preds, labels=list(range(self.num_classes)))

        # ROC-AUC（需要所有类别都出现）
        try:
            present_classes = np.unique(labels)
            if len(present_classes) >= 2:
                auc = roc_auc_score(
                    labels,
                    probs[:, :len(present_classes)] if probs.shape[1] > len(present_classes) else probs,
                    multi_class="ovr",
                    average="macro",
                    labels=present_classes,
                )
            else:
                auc = 0.0
        except Exception as e:
            logger.warning(f"ROC-AUC计算失败: {e}")
            auc = 0.0

        # 分类报告
        report = classification_report(
            labels, preds,
            labels=list(range(self.num_classes)),
            target_names=FAULT_CLASS_NAMES[:self.num_classes],
            zero_division=0,
            output_dict=True,
        )

        return {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
            "roc_auc": float(auc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "num_samples": len(labels),
        }

    def plot_confusion_matrix(
        self,
        labels: Optional[np.ndarray] = None,
        preds: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ):
        """
        绘制混淆矩阵热力图并保存到文件
        :param labels: 真实标签（None时使用上次evaluate的结果）
        :param preds: 预测标签
        :param save_path: 保存路径
        """
        if labels is None:
            labels = getattr(self, "_last_labels", None)
            preds = getattr(self, "_last_preds", None)

        if labels is None or preds is None:
            logger.warning("无可用预测结果，请先调用evaluate()")
            return

        cm = confusion_matrix(labels, preds, labels=list(range(self.num_classes)))
        class_names = FAULT_CLASS_NAMES[:self.num_classes]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(self.num_classes),
            yticks=np.arange(self.num_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            title="故障预测混淆矩阵",
            ylabel="真实标签",
            xlabel="预测标签",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)

        # 在格子中填写数字
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9,
                )

        fig.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "confusion_matrix.png")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"混淆矩阵已保存: {save_path}")

    def plot_roc_curve(
        self,
        labels: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ):
        """
        绘制多类别ROC曲线并保存到文件
        :param labels: 真实标签
        :param probs: 预测概率矩阵
        :param save_path: 保存路径
        """
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc

        if labels is None:
            labels = getattr(self, "_last_labels", None)
            probs = getattr(self, "_last_probs", None)

        if labels is None or probs is None:
            logger.warning("无可用预测结果，请先调用evaluate()")
            return

        class_names = FAULT_CLASS_NAMES[:self.num_classes]
        labels_bin = label_binarize(labels, classes=list(range(self.num_classes)))

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
        for i, (name, color) in enumerate(zip(class_names, colors)):
            if labels_bin[:, i].sum() == 0:
                continue
            try:
                fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=1.5,
                        label=f"{name} (AUC={roc_auc:.3f})")
            except Exception:
                pass

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="随机猜测")
        ax.set(
            xlim=[0.0, 1.0], ylim=[0.0, 1.05],
            xlabel="假阳性率 (FPR)",
            ylabel="真阳性率 (TPR)",
            title="各故障类型ROC曲线",
        )
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "roc_curves.png")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"ROC曲线已保存: {save_path}")

    def export_report(self, metrics: Dict, save_path: Optional[str] = None) -> str:
        """
        将评估报告导出为JSON文件
        :param metrics: 指标字典
        :param save_path: 保存路径
        :return: 保存路径
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "evaluation_report.json")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        logger.info(f"评估报告已导出: {save_path}")
        return save_path
