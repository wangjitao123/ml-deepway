"""
告警引擎模块
基于预测结果生成和管理多级别告警
"""

import uuid
import logging
from datetime import datetime
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class AlertLevel(IntEnum):
    """告警级别枚举（数值越大越严重）"""
    INFO = 0        # 信息：轻微异常，记录观察
    WARNING = 1     # 警告：需要注意，建议检查
    CRITICAL = 2    # 严重：需要尽快处理
    EMERGENCY = 3   # 紧急：立即停车检查
    SHUTDOWN = 4    # 停机：立即关闭系统


# 告警级别中文描述
ALERT_LEVEL_NAMES = {
    AlertLevel.INFO: "信息",
    AlertLevel.WARNING: "警告",
    AlertLevel.CRITICAL: "严重",
    AlertLevel.EMERGENCY: "紧急",
    AlertLevel.SHUTDOWN: "停机",
}

# 故障类型名称
FAULT_TYPE_NAMES = [
    "正常运行",
    "发动机过热",
    "电池异常",
    "制动失效",
    "轮胎气压异常",
    "电机/控制器故障",
    "润滑系统故障",
    "冷却系统故障",
]


@dataclass
class Alert:
    """
    告警数据类
    包含告警的完整信息
    """
    id: str                         # 唯一告警ID
    level: AlertLevel               # 告警级别
    fault_type: int                 # 故障类型编号
    probability: float              # 故障概率
    severity: float                 # 故障严重程度 (0~1)
    message: str                    # 告警消息
    repair_advice: str              # 维修建议
    timestamp: datetime             # 告警时间
    vehicle_id: str = "UNKNOWN"     # 车辆ID
    extra: Dict[str, Any] = field(default_factory=dict)  # 额外信息

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "id": self.id,
            "level": self.level.value,
            "level_name": ALERT_LEVEL_NAMES[self.level],
            "fault_type": self.fault_type,
            "fault_name": FAULT_TYPE_NAMES[self.fault_type] if self.fault_type < len(FAULT_TYPE_NAMES) else "未知故障",
            "probability": round(self.probability, 4),
            "severity": round(self.severity, 4),
            "message": self.message,
            "repair_advice": self.repair_advice,
            "timestamp": self.timestamp.isoformat(),
            "vehicle_id": self.vehicle_id,
            "extra": self.extra,
        }


class AlertEngine:
    """
    告警引擎
    处理预测结果，生成标准化告警，维护告警历史
    """

    # 概率阈值映射到告警级别
    PROBABILITY_THRESHOLDS = [
        (0.95, AlertLevel.SHUTDOWN),
        (0.85, AlertLevel.EMERGENCY),
        (0.70, AlertLevel.CRITICAL),
        (0.50, AlertLevel.WARNING),
        (0.30, AlertLevel.INFO),
    ]

    # 各故障类型的维修建议（中文）
    REPAIR_ADVICE = {
        0: "车辆运行正常，定期保养即可",
        1: "检查冷却液液位和散热器，停车散热，严重时联系道路救援",
        2: "检查电池连接线路，测量电池电压和内阻，必要时更换电池组",
        3: "立即停车！检查制动液液位和制动管路，禁止行驶",
        4: "检查四轮胎压并充气至标准值(32-35 PSI)，检查是否漏气",
        5: "检查电机冷却系统，检查控制器散热，联系专业技师诊断",
        6: "立即停车！检查机油液位，更换机油滤芯，严重时更换机油泵",
        7: "检查冷却液液位，检查水泵和节温器，检查冷却风扇运转",
    }

    def __init__(
        self,
        vehicle_id: str = "VEHICLE_001",
        max_history: int = 1000,
    ):
        """
        初始化告警引擎
        :param vehicle_id: 车辆标识符
        :param max_history: 最大保留历史告警数量
        """
        self.vehicle_id = vehicle_id
        self.max_history = max_history
        self.alert_history: List[Alert] = []
        self._alert_count_by_level = {level: 0 for level in AlertLevel}

    def _determine_alert_level(self, probability: float) -> Optional[AlertLevel]:
        """
        根据故障概率确定告警级别
        :param probability: 故障概率 (0~1)
        :return: 告警级别，None表示无需告警
        """
        for threshold, level in self.PROBABILITY_THRESHOLDS:
            if probability >= threshold:
                return level
        return None   # 概率低于最低阈值，不生成告警

    def process_prediction(self, prediction_result: dict) -> Optional[Alert]:
        """
        处理模型预测结果，生成告警
        :param prediction_result: 预测结果字典，需包含:
               - fault_type: 故障类型编号
               - probability: 最高故障概率
               - severity: 严重程度 (0~1)
               - repair_advice (可选)
        :return: 生成的Alert对象，或None（无需告警时）
        """
        fault_type = prediction_result.get("fault_type", 0)
        probability = float(prediction_result.get("probability", 0.0))
        severity = float(prediction_result.get("severity", 0.0))

        # 正常状态（类型0且概率低）不生成告警
        if fault_type == 0 and probability < 0.3:
            return None

        # 确定告警级别
        alert_level = self._determine_alert_level(probability)
        if alert_level is None:
            return None

        # 构建告警消息
        fault_name = FAULT_TYPE_NAMES[fault_type] if fault_type < len(FAULT_TYPE_NAMES) else f"未知故障(类型{fault_type})"
        level_name = ALERT_LEVEL_NAMES[alert_level]
        message = (
            f"[{level_name}] 检测到{fault_name}，"
            f"概率: {probability:.1%}，"
            f"严重程度: {severity:.1%}"
        )

        repair_advice = prediction_result.get(
            "repair_advice",
            self.REPAIR_ADVICE.get(fault_type, "请联系专业技师检查")
        )

        # 创建告警对象
        alert = Alert(
            id=str(uuid.uuid4())[:8],
            level=alert_level,
            fault_type=fault_type,
            probability=probability,
            severity=severity,
            message=message,
            repair_advice=repair_advice,
            timestamp=datetime.now(),
            vehicle_id=self.vehicle_id,
            extra=prediction_result.get("extra", {}),
        )

        # 存入历史记录
        self.alert_history.append(alert)
        self._alert_count_by_level[alert_level] += 1

        # 超出上限时删除最旧的记录
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)

        # 根据告警级别输出日志
        log_msg = f"车辆 {self.vehicle_id}: {message}"
        if alert_level >= AlertLevel.EMERGENCY:
            logger.critical(log_msg)
        elif alert_level >= AlertLevel.CRITICAL:
            logger.error(log_msg)
        elif alert_level >= AlertLevel.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        return alert

    def get_recent_alerts(self, n: int = 10) -> List[Alert]:
        """
        获取最近N条告警记录
        :param n: 返回记录数量
        :return: 告警列表（最新的在前）
        """
        return self.alert_history[-n:][::-1]

    def get_statistics(self) -> dict:
        """
        获取告警统计信息
        :return: 统计数据字典
        """
        total = len(self.alert_history)
        by_level = {
            ALERT_LEVEL_NAMES[level]: count
            for level, count in self._alert_count_by_level.items()
        }

        # 故障类型分布统计
        fault_counts = {}
        for alert in self.alert_history:
            name = FAULT_TYPE_NAMES[alert.fault_type] if alert.fault_type < len(FAULT_TYPE_NAMES) else f"类型{alert.fault_type}"
            fault_counts[name] = fault_counts.get(name, 0) + 1

        # 最近24小时告警数
        recent_24h = sum(
            1 for a in self.alert_history
            if (datetime.now() - a.timestamp).total_seconds() < 86400
        )

        return {
            "total_alerts": total,
            "by_level": by_level,
            "by_fault_type": fault_counts,
            "recent_24h": recent_24h,
            "vehicle_id": self.vehicle_id,
        }

    def clear_history(self):
        """清空告警历史"""
        self.alert_history.clear()
        self._alert_count_by_level = {level: 0 for level in AlertLevel}
        logger.info("告警历史已清空")
