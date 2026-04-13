"""
数据质量监控模块
检测传感器数据缺失、异常和整体质量评分
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# 22个传感器特征名称
FEATURE_NAMES = [
    "engine_rpm", "engine_temp", "oil_pressure", "coolant_temp",
    "battery_voltage", "battery_temp", "battery_soc", "motor_current",
    "motor_temp", "brake_pressure", "tire_pressure_fl", "tire_pressure_fr",
    "tire_pressure_rl", "tire_pressure_rr", "vibration_level", "fuel_consumption",
    "vehicle_speed", "throttle_position", "ambient_temp", "humidity",
    "road_gradient", "altitude",
]


class DataQualityMonitor:
    """
    数据质量监控器
    提供缺失率检测、异常率检测和综合健康评分
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        初始化数据质量监控器
        :param feature_names: 特征名称列表（None时使用默认22个传感器名）
        """
        self.feature_names = feature_names or FEATURE_NAMES

    def check_missing_rate(
        self, data: np.ndarray, axis: int = 0
    ) -> float:
        """
        计算数据整体缺失率
        :param data: 输入数据矩阵 (N, D) 或 (N,)
        :param axis: 统计轴（0=按样本统计，None=全局）
        :return: 缺失率 (0~1)
        """
        data = np.array(data, dtype=float)
        total_elements = data.size

        if total_elements == 0:
            return 1.0

        # NaN和Inf都视为缺失
        missing_count = np.sum(~np.isfinite(data))
        return float(missing_count / total_elements)

    def check_missing_rate_per_feature(self, data: np.ndarray) -> Dict[str, float]:
        """
        计算每个特征的缺失率
        :param data: 输入矩阵 (N, D)
        :return: 特征名到缺失率的字典
        """
        data = np.array(data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_samples = data.shape[0]
        result = {}
        for i, name in enumerate(self.feature_names[:data.shape[1]]):
            missing = np.sum(~np.isfinite(data[:, i]))
            result[name] = float(missing / max(n_samples, 1))

        return result

    def check_anomaly_rate(
        self,
        data: np.ndarray,
        valid_ranges: Dict[str, Dict[str, float]],
    ) -> float:
        """
        计算数据整体异常率（超出有效范围的比例）
        :param data: 输入矩阵 (N, D)
        :param valid_ranges: 有效范围字典 {feature_name: {'min': x, 'max': y}}
        :return: 异常率 (0~1)
        """
        data = np.array(data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_samples, n_features = data.shape
        total_valid_checks = 0
        anomaly_count = 0

        for i, name in enumerate(self.feature_names[:n_features]):
            if name not in valid_ranges:
                continue

            min_val = valid_ranges[name].get("min", -np.inf)
            max_val = valid_ranges[name].get("max", np.inf)

            col = data[:, i]
            # 有效值（非NaN/Inf）才参与范围检查
            valid_mask = np.isfinite(col)
            valid_col = col[valid_mask]

            anomalies = np.sum((valid_col < min_val) | (valid_col > max_val))
            anomaly_count += anomalies
            total_valid_checks += len(valid_col)

        if total_valid_checks == 0:
            return 0.0

        return float(anomaly_count / total_valid_checks)

    def check_anomaly_rate_per_feature(
        self,
        data: np.ndarray,
        valid_ranges: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        计算每个特征的异常率
        :param data: 输入矩阵 (N, D)
        :param valid_ranges: 有效范围字典
        :return: 特征名到异常率的字典
        """
        data = np.array(data, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_samples = data.shape[0]
        result = {}

        for i, name in enumerate(self.feature_names[:data.shape[1]]):
            if name not in valid_ranges:
                result[name] = 0.0
                continue

            min_val = valid_ranges[name].get("min", -np.inf)
            max_val = valid_ranges[name].get("max", np.inf)
            col = data[:, i]

            valid_mask = np.isfinite(col)
            valid_col = col[valid_mask]
            if len(valid_col) == 0:
                result[name] = 1.0
                continue

            anomalies = np.sum((valid_col < min_val) | (valid_col > max_val))
            result[name] = float(anomalies / len(valid_col))

        return result

    def compute_health_score(
        self,
        data: np.ndarray,
        valid_ranges: Dict[str, Dict[str, float]],
        missing_weight: float = 0.4,
        anomaly_weight: float = 0.6,
    ) -> float:
        """
        计算综合数据健康评分（0~1，越高越好）
        健康评分 = 1 - (缺失权重×缺失率 + 异常权重×异常率)
        :param data: 输入数据矩阵
        :param valid_ranges: 有效范围字典
        :param missing_weight: 缺失率权重
        :param anomaly_weight: 异常率权重
        :return: 健康评分 (0~1)
        """
        missing_rate = self.check_missing_rate(data)
        anomaly_rate = self.check_anomaly_rate(data, valid_ranges)

        # 加权合成
        degradation = missing_weight * missing_rate + anomaly_weight * anomaly_rate
        health_score = max(0.0, 1.0 - degradation)

        return float(health_score)

    def generate_report(
        self,
        data: np.ndarray,
        valid_ranges: Dict[str, Dict[str, float]],
    ) -> Dict:
        """
        生成完整的数据质量报告
        :param data: 输入数据矩阵 (N, D)
        :param valid_ranges: 有效范围字典
        :return: 质量报告字典
        """
        data = np.array(data, dtype=float)

        # 整体指标
        overall_missing = self.check_missing_rate(data)
        overall_anomaly = self.check_anomaly_rate(data, valid_ranges)
        health_score = self.compute_health_score(data, valid_ranges)

        # 逐特征指标
        missing_per_feature = self.check_missing_rate_per_feature(data)
        anomaly_per_feature = self.check_anomaly_rate_per_feature(data, valid_ranges)

        # 识别问题特征（缺失率或异常率超过10%）
        problem_features = [
            name for name in self.feature_names[:data.shape[1] if data.ndim > 1 else 1]
            if missing_per_feature.get(name, 0) > 0.1
            or anomaly_per_feature.get(name, 0) > 0.1
        ]

        # 数据基本统计
        if data.ndim == 2 and data.shape[0] > 0:
            data_stats = {
                "n_samples": int(data.shape[0]),
                "n_features": int(data.shape[1]),
                "mean": float(np.nanmean(data)),
                "std": float(np.nanstd(data)),
            }
        else:
            data_stats = {"n_samples": 0, "n_features": 0}

        # 质量评级
        if health_score >= 0.95:
            quality_grade = "优秀"
        elif health_score >= 0.85:
            quality_grade = "良好"
        elif health_score >= 0.70:
            quality_grade = "一般"
        elif health_score >= 0.50:
            quality_grade = "较差"
        else:
            quality_grade = "很差"

        return {
            "health_score": round(health_score, 4),
            "quality_grade": quality_grade,
            "overall_missing_rate": round(overall_missing, 4),
            "overall_anomaly_rate": round(overall_anomaly, 4),
            "missing_per_feature": {k: round(v, 4) for k, v in missing_per_feature.items()},
            "anomaly_per_feature": {k: round(v, 4) for k, v in anomaly_per_feature.items()},
            "problem_features": problem_features,
            "data_stats": data_stats,
            "recommendations": self._generate_recommendations(
                health_score, overall_missing, overall_anomaly, problem_features
            ),
        }

    def _generate_recommendations(
        self,
        health_score: float,
        missing_rate: float,
        anomaly_rate: float,
        problem_features: List[str],
    ) -> List[str]:
        """生成改善建议"""
        recommendations = []

        if health_score >= 0.95:
            recommendations.append("数据质量优秀，可正常用于模型推理")
        elif health_score >= 0.85:
            recommendations.append("数据质量良好，建议关注问题特征")

        if missing_rate > 0.05:
            recommendations.append(f"缺失率偏高({missing_rate:.1%})，建议检查传感器连接和数据采集管道")

        if anomaly_rate > 0.05:
            recommendations.append(f"异常率偏高({anomaly_rate:.1%})，建议校准传感器")

        if problem_features:
            recommendations.append(f"问题传感器: {', '.join(problem_features[:5])}，建议重点检查")

        if not recommendations:
            recommendations.append("建议定期检查数据质量，保持采集设备正常运行")

        return recommendations
