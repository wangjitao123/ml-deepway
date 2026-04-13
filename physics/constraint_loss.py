"""
物理约束损失函数
基于车辆物理规律设计的可微分损失项
所有操作均使用PyTorch张量运算，确保支持反向传播
"""

import torch
import torch.nn as nn


class PhysicsConstraintLoss(nn.Module):
    """
    物理约束损失模块（完全可微分）

    实现3条基于物理规律的约束:
    1. 冷却规则：冷却液温度高且车速>0时，发动机温度应降低
    2. 电池规则：电池温度过高时，SOC不应异常偏高
    3. 制动规则：高速行驶时制动压力应能有效响应

    传感器特征索引:
    engine_temp=1, coolant_temp=3, vehicle_speed=16,
    battery_temp=5, battery_soc=6, brake_pressure=9
    """

    # 传感器特征索引常量
    IDX_ENGINE_TEMP = 1
    IDX_COOLANT_TEMP = 3
    IDX_BATTERY_TEMP = 5
    IDX_BATTERY_SOC = 6
    IDX_BRAKE_PRESSURE = 9
    IDX_VEHICLE_SPEED = 16

    def __init__(
        self,
        engine_temp_threshold: float = 100.0,  # 发动机温度过热阈值 (°C)
        coolant_temp_threshold: float = 90.0,  # 冷却液激活阈值 (°C)
        speed_threshold: float = 10.0,          # 最低有效车速 (km/h)
        battery_temp_threshold: float = 45.0,  # 电池过热阈值 (°C)
        battery_soc_threshold: float = 80.0,   # 电池过热时SOC阈值 (%)
        high_speed_threshold: float = 50.0,    # 高速制动触发阈值 (km/h)
        min_brake_pressure: float = 20.0,      # 高速时最低制动压力 (bar)
        weight_cooling: float = 1.0,
        weight_battery: float = 1.0,
        weight_brake: float = 1.0,
    ):
        """
        初始化物理约束损失
        :param engine_temp_threshold: 发动机过热温度阈值
        :param coolant_temp_threshold: 冷却液触发阈值
        :param speed_threshold: 最低有效车速
        :param battery_temp_threshold: 电池过热阈值
        :param battery_soc_threshold: 电池高SOC阈值
        :param high_speed_threshold: 高速触发阈值
        :param min_brake_pressure: 高速最低制动压力
        :param weight_cooling: 冷却约束权重
        :param weight_battery: 电池约束权重
        :param weight_brake: 制动约束权重
        """
        super().__init__()

        # 保存阈值为常量（不参与梯度）
        self.engine_temp_threshold = engine_temp_threshold
        self.coolant_temp_threshold = coolant_temp_threshold
        self.speed_threshold = speed_threshold
        self.battery_temp_threshold = battery_temp_threshold
        self.battery_soc_threshold = battery_soc_threshold
        self.high_speed_threshold = high_speed_threshold
        self.min_brake_pressure = min_brake_pressure

        self.weight_cooling = weight_cooling
        self.weight_battery = weight_battery
        self.weight_brake = weight_brake

    def forward(
        self,
        predictions: dict,
        sensor_sequences: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算物理约束损失（完全可微分）
        :param predictions: 模型输出字典（当前未使用，为接口扩展预留）
        :param sensor_sequences: 传感器时序张量，形状 (B, seq_len, 22)
        :return: 标量损失张量（支持.backward()）
        """
        # 取最后一个时间步的传感器读数 (B, 22)
        last = sensor_sequences[:, -1, :]

        # 提取各传感器值（均为连续张量，支持梯度）
        engine_temp = last[:, self.IDX_ENGINE_TEMP]       # (B,)
        coolant_temp = last[:, self.IDX_COOLANT_TEMP]     # (B,)
        vehicle_speed = last[:, self.IDX_VEHICLE_SPEED]   # (B,)
        battery_temp = last[:, self.IDX_BATTERY_TEMP]     # (B,)
        battery_soc = last[:, self.IDX_BATTERY_SOC]       # (B,)
        brake_pressure = last[:, self.IDX_BRAKE_PRESSURE] # (B,)

        # ── 约束1：冷却规则 ──
        # 当冷却液温度高 AND 车速>0 时，发动机温度不应过高
        # 使用 sigmoid 代替 if/else（保持可微分性）
        coolant_active = torch.sigmoid(coolant_temp - self.coolant_temp_threshold)
        speed_active = torch.sigmoid(vehicle_speed - self.speed_threshold)
        # 发动机超温惩罚：超过阈值的部分被惩罚，乘以冷却液和车速激活权重
        engine_overtemp = torch.relu(engine_temp - self.engine_temp_threshold)
        cooling_loss = (engine_overtemp * coolant_active * speed_active).mean()

        # ── 约束2：电池规则 ──
        # 电池过热时，SOC不应异常偏高（表明充电管理异常）
        battery_hot = torch.sigmoid(battery_temp - self.battery_temp_threshold)
        # SOC超出安全上限的惩罚
        soc_overhigh = torch.relu(battery_soc - self.battery_soc_threshold)
        battery_loss = (soc_overhigh * battery_hot).mean()

        # ── 约束3：制动规则 ──
        # 高速行驶时，制动压力不应严重不足（紧急制动能力约束）
        high_speed_active = torch.sigmoid(vehicle_speed - self.high_speed_threshold)
        # 制动压力不足惩罚：低于最低制动压力的部分
        brake_deficiency = torch.relu(self.min_brake_pressure - brake_pressure)
        brake_loss = (brake_deficiency * high_speed_active).mean()

        # 加权求和
        total_loss = (
            self.weight_cooling * cooling_loss
            + self.weight_battery * battery_loss
            + self.weight_brake * brake_loss
        ) / (self.weight_cooling + self.weight_battery + self.weight_brake)

        return total_loss
