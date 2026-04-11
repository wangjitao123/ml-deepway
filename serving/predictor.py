"""
故障预测服务模块
提供模型加载、预处理、预测和告警级别判断功能
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

# 故障类型名称（中文）
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

# 各故障类型的维修建议（中文）
REPAIR_ADVICE = {
    0: "车辆运行正常，按计划进行定期保养",
    1: "⚠️ 发动机过热：立即靠边停车，关闭发动机冷却，检查冷却液液位和散热器，避免打开散热器盖",
    2: "⚠️ 电池异常：检查电池连接线路和接触点，测量电池电压，必要时更换电池或联系售后",
    3: "🚨 制动失效：立即停车！禁止行驶，检查制动液液位、制动管路和制动片，必须修复后方可上路",
    4: "⚠️ 轮胎气压异常：检查四轮气压并充气至标准值(32-35 PSI)，检查是否有轮胎损伤或漏气",
    5: "⚠️ 电机/控制器故障：停止行驶，检查电机冷却系统和控制器散热，联系专业新能源技师",
    6: "🚨 润滑系统故障：立即停车！检查机油液位，若机油不足严禁启动，更换机油滤芯",
    7: "⚠️ 冷却系统故障：停车冷却，检查冷却液液位、水泵运转和节温器状态，避免高速行驶",
}

# 告警级别名称
ALERT_LEVELS = {
    "normal": "正常",
    "info": "信息",
    "warning": "警告",
    "critical": "严重",
    "emergency": "紧急",
    "shutdown": "停机",
}


class FaultPredictor:
    """
    车辆故障预测器
    封装模型推理、预处理和告警生成的完整流程
    """

    # 概率阈值对应告警级别
    ALERT_THRESHOLDS = [
        (0.95, "shutdown"),
        (0.85, "emergency"),
        (0.70, "critical"),
        (0.50, "warning"),
        (0.30, "info"),
        (0.0, "normal"),
    ]

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        seq_len: int = 30,
        input_dim: int = 22,
        num_classes: int = 8,
        preprocessor=None,
    ):
        """
        初始化预测器
        :param model: 已加载的PyTorch模型（可为None）
        :param device: 计算设备
        :param seq_len: 时序窗口长度
        :param input_dim: 输入特征维度
        :param num_classes: 故障类别数
        :param preprocessor: 数据预处理器
        """
        self.device = device or torch.device("cpu")
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.preprocessor = preprocessor

        self.model = model
        if model is not None:
            self.model.to(self.device)
            self.model.eval()

        # 滑动窗口缓冲区（用于实时预测）
        self._sensor_buffer = []

    def load_model(self, checkpoint_path: str) -> bool:
        """
        从检查点文件加载模型
        :param checkpoint_path: 检查点路径
        :return: 加载成功返回True
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"检查点文件不存在: {checkpoint_path}")
            return False

        try:
            # 动态导入避免循环依赖
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from models.lstm_transformer import LSTMTransformerModel

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model = LSTMTransformerModel(
                input_dim=self.input_dim,
                hidden_dim=128,
                num_layers=2,
                num_heads=4,
                num_classes=self.num_classes,
                seq_len=self.seq_len,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"模型已从检查点加载: {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def preprocess(
        self,
        sensor_data: Union[dict, np.ndarray, list],
    ) -> torch.Tensor:
        """
        预处理传感器数据为模型输入格式
        :param sensor_data: 传感器数据（字典、numpy数组或列表）
        :return: 模型输入张量 (1, seq_len, input_dim)
        """
        # 转换字典格式
        if isinstance(sensor_data, dict):
            feature_names = [
                "engine_rpm", "engine_temp", "oil_pressure", "coolant_temp",
                "battery_voltage", "battery_temp", "battery_soc", "motor_current",
                "motor_temp", "brake_pressure", "tire_pressure_fl", "tire_pressure_fr",
                "tire_pressure_rl", "tire_pressure_rr", "vibration_level",
                "fuel_consumption", "vehicle_speed", "throttle_position",
                "ambient_temp", "humidity", "road_gradient", "altitude",
            ]
            arr = np.array([sensor_data.get(f, 0.0) for f in feature_names], dtype=np.float32)
        else:
            arr = np.array(sensor_data, dtype=np.float32)

        # 确保形状为 (input_dim,)
        arr = arr.flatten()[:self.input_dim]
        if len(arr) < self.input_dim:
            arr = np.pad(arr, (0, self.input_dim - len(arr)))

        # 应用预处理器（若有）
        if self.preprocessor is not None:
            try:
                arr = self.preprocessor.transform(arr.reshape(1, -1))[0]
            except Exception:
                pass

        # 更新滑动窗口缓冲区
        self._sensor_buffer.append(arr)
        if len(self._sensor_buffer) > self.seq_len:
            self._sensor_buffer.pop(0)

        # 填充不足seq_len的部分（用第一帧重复填充）
        buffer = list(self._sensor_buffer)
        while len(buffer) < self.seq_len:
            buffer.insert(0, buffer[0])

        # 构建序列张量 (1, seq_len, input_dim)
        seq = np.array(buffer[-self.seq_len:], dtype=np.float32)
        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        return tensor

    def _get_alert_level(self, probability: float) -> str:
        """
        根据故障概率确定告警级别
        :param probability: 故障概率 (0~1)
        :return: 告警级别字符串
        """
        for threshold, level in self.ALERT_THRESHOLDS:
            if probability >= threshold:
                return level
        return "normal"

    def predict(
        self,
        sensor_data: Union[dict, np.ndarray, torch.Tensor],
    ) -> Dict:
        """
        执行故障预测
        :param sensor_data: 传感器数据（支持字典/数组/张量）
        :return: 预测结果字典，包含:
                 - fault_type: 故障类型编号
                 - fault_name: 故障类型名称
                 - probability: 最高类别概率
                 - all_probs: 所有类别概率
                 - severity: 严重程度
                 - alert_level: 告警级别
                 - repair_advice: 维修建议
        """
        if self.model is None:
            logger.error("模型未加载，请先调用load_model()")
            return self._empty_result()

        # 预处理
        if isinstance(sensor_data, torch.Tensor):
            if sensor_data.dim() == 2:
                x = sensor_data.unsqueeze(0).to(self.device)
            elif sensor_data.dim() == 3:
                x = sensor_data.to(self.device)
            else:
                x = self.preprocess(sensor_data.numpy())
        else:
            x = self.preprocess(sensor_data)

        # 模型推理
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)

        fault_probs = outputs["fault_probs"][0].cpu().numpy()
        severity = float(outputs["severity"][0].cpu().item())
        fault_type = int(np.argmax(fault_probs))
        probability = float(fault_probs[fault_type])

        # 确定告警级别
        if fault_type == 0:
            alert_level = "normal" if probability > 0.7 else "info"
        else:
            alert_level = self._get_alert_level(probability)

        return {
            "fault_type": fault_type,
            "fault_name": FAULT_TYPE_NAMES[fault_type],
            "probability": round(probability, 4),
            "all_probs": {
                FAULT_TYPE_NAMES[i]: round(float(p), 4)
                for i, p in enumerate(fault_probs)
            },
            "severity": round(severity, 4),
            "alert_level": alert_level,
            "alert_level_name": ALERT_LEVELS.get(alert_level, alert_level),
            "repair_advice": REPAIR_ADVICE.get(fault_type, "请联系专业技师"),
        }

    def physics_check(self, sensor_data: dict) -> dict:
        """
        运行物理一致性验证
        :param sensor_data: 传感器数据字典
        :return: 验证结果字典
        """
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from physics.dynamics_model import VehicleDynamicsModel
            validator = VehicleDynamicsModel()
            return validator.validate_sensor_consistency(sensor_data)
        except Exception as e:
            logger.error(f"物理验证失败: {e}")
            return {"valid": True, "issues": [], "error": str(e)}

    def _empty_result(self) -> dict:
        """返回空结果（模型未加载时）"""
        return {
            "fault_type": -1,
            "fault_name": "模型未加载",
            "probability": 0.0,
            "all_probs": {},
            "severity": 0.0,
            "alert_level": "normal",
            "alert_level_name": "正常",
            "repair_advice": "请先加载模型",
        }

    def reset_buffer(self):
        """清空传感器缓冲区"""
        self._sensor_buffer.clear()
