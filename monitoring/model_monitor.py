"""
模型监控模块
使用PSI（群体稳定性指数）检测数据漂移
"""

import numpy as np
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    模型性能监控器
    通过PSI（Population Stability Index）检测数据分布漂移
    """

    # PSI判断标准：
    # PSI < 0.1：分布稳定
    # 0.1 ≤ PSI < 0.2：轻微漂移，需要关注
    # PSI ≥ 0.2：显著漂移，需要重新训练模型
    PSI_THRESHOLDS = {
        "stable": 0.1,
        "moderate": 0.2,
    }

    def __init__(
        self,
        baseline_predictions: Optional[np.ndarray] = None,
        num_classes: int = 8,
        buckets: int = 10,
        window_size: int = 1000,
    ):
        """
        初始化模型监控器
        :param baseline_predictions: 基线预测概率（形状 N×num_classes）
        :param num_classes: 分类类别数
        :param buckets: PSI计算分箱数
        :param window_size: 滑动窗口大小
        """
        self.num_classes = num_classes
        self.buckets = buckets
        self.window_size = window_size

        # 基线分布（训练集/验证集预测）
        self.baseline: Optional[np.ndarray] = None
        if baseline_predictions is not None:
            self.update_baseline(baseline_predictions)

        # 最近预测历史（滑动窗口）
        self._prediction_history: List[np.ndarray] = []
        self._psi_history: List[float] = []

    def compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        buckets: int = None,
    ) -> float:
        """
        计算PSI（群体稳定性指数）
        PSI = Σ (实际占比 - 期望占比) × ln(实际占比 / 期望占比)
        :param expected: 期望（基线）分布，一维数组
        :param actual: 实际分布，一维数组
        :param buckets: 分箱数量
        :return: PSI值
        """
        if buckets is None:
            buckets = self.buckets

        # 将数组展平为一维
        expected = np.array(expected).flatten()
        actual = np.array(actual).flatten()

        # 过滤NaN
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]

        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # 确定分箱边界（基于期望分布的分位数）
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints[0] -= 1e-6   # 确保最小值被包含
        breakpoints[-1] += 1e-6  # 确保最大值被包含
        breakpoints = np.unique(breakpoints)

        if len(breakpoints) < 2:
            return 0.0

        # 计算各箱的频率
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        # 转换为比例，加小量避免除零和log(0)
        eps = 1e-6
        expected_pct = expected_counts / (len(expected) + eps) + eps
        actual_pct = actual_counts / (len(actual) + eps) + eps

        # PSI公式
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    def check_drift(self, new_predictions: np.ndarray) -> Dict:
        """
        检测数据漂移
        :param new_predictions: 新预测概率矩阵（N×num_classes）
        :return: 漂移检测结果字典
        """
        if self.baseline is None:
            logger.warning("基线未设置，无法检测漂移。请先调用update_baseline()")
            return {
                "drift_detected": False,
                "psi_scores": {},
                "overall_psi": 0.0,
                "status": "no_baseline",
            }

        new_predictions = np.array(new_predictions)
        if new_predictions.ndim == 1:
            new_predictions = new_predictions.reshape(-1, 1)

        # 按类别计算PSI
        psi_scores = {}
        n_cols = min(new_predictions.shape[1], self.baseline.shape[1])

        for cls_idx in range(n_cols):
            expected_col = self.baseline[:, cls_idx]
            actual_col = new_predictions[:, cls_idx]
            psi = self.compute_psi(expected_col, actual_col)
            psi_scores[f"class_{cls_idx}"] = round(psi, 4)

        overall_psi = np.mean(list(psi_scores.values())) if psi_scores else 0.0
        self._psi_history.append(overall_psi)

        # 判断漂移状态
        if overall_psi >= self.PSI_THRESHOLDS["moderate"]:
            status = "significant_drift"
            drift_detected = True
        elif overall_psi >= self.PSI_THRESHOLDS["stable"]:
            status = "moderate_drift"
            drift_detected = True
        else:
            status = "stable"
            drift_detected = False

        result = {
            "drift_detected": drift_detected,
            "psi_scores": psi_scores,
            "overall_psi": round(overall_psi, 4),
            "status": status,
            "threshold_moderate": self.PSI_THRESHOLDS["moderate"],
            "threshold_stable": self.PSI_THRESHOLDS["stable"],
        }

        if drift_detected:
            logger.warning(f"检测到数据漂移: PSI={overall_psi:.4f}, 状态={status}")
        else:
            logger.debug(f"数据分布稳定: PSI={overall_psi:.4f}")

        return result

    def update_baseline(self, predictions: np.ndarray):
        """
        更新基线分布（例如在模型重训后）
        :param predictions: 基线预测概率矩阵
        """
        predictions = np.array(predictions)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        self.baseline = predictions
        logger.info(f"基线已更新，共 {len(predictions)} 个样本")

    def get_psi_trend(self, n: int = 20) -> List[float]:
        """
        获取最近N次PSI值的趋势
        :param n: 返回的历史条数
        :return: PSI历史值列表
        """
        return self._psi_history[-n:]

    def add_predictions_to_window(self, predictions: np.ndarray):
        """
        将新预测加入滑动窗口
        :param predictions: 预测概率矩阵
        """
        predictions = np.array(predictions)
        self._prediction_history.append(predictions)

        # 保持滑动窗口大小
        if sum(len(p) for p in self._prediction_history) > self.window_size:
            self._prediction_history.pop(0)
