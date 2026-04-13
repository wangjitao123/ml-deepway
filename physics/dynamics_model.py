"""
车辆动力学物理模型
基于经典物理方程计算阻力、功率及传感器一致性验证
"""

import math
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class VehicleDynamicsModel:
    """
    车辆动力学模型
    基于空气动力学和力学方程计算行驶阻力与功率需求
    """

    def __init__(
        self,
        mass: float = 1500.0,           # 整车质量 (kg)
        drag_coefficient: float = 0.3,  # 空气阻力系数 Cd
        rolling_resistance: float = 0.015,  # 滚动阻力系数 μ
        frontal_area: float = 2.2,      # 迎风面积 (m²)
        gravity: float = 9.81,          # 重力加速度 (m/s²)
        air_density: float = 1.225,     # 空气密度 (kg/m³)
    ):
        """
        初始化车辆动力学参数
        :param mass: 整车质量 (kg)
        :param drag_coefficient: 空气阻力系数
        :param rolling_resistance: 滚动阻力系数
        :param frontal_area: 迎风面积 (m²)
        :param gravity: 重力加速度 (m/s²)
        :param air_density: 空气密度 (kg/m³)
        """
        self.mass = mass
        self.drag_coefficient = drag_coefficient
        self.rolling_resistance = rolling_resistance
        self.frontal_area = frontal_area
        self.gravity = gravity
        self.air_density = air_density

    def calculate_air_resistance(self, speed_kmh: float) -> float:
        """
        计算空气阻力 (N)
        公式: F_drag = 0.5 × ρ × Cd × A × v²
        :param speed_kmh: 车速 (km/h)
        :return: 空气阻力 (N)
        """
        speed_ms = speed_kmh / 3.6    # 转换为 m/s
        f_drag = (
            0.5
            * self.air_density
            * self.drag_coefficient
            * self.frontal_area
            * speed_ms ** 2
        )
        return f_drag

    def calculate_rolling_resistance(self, mass: float = None, gradient_pct: float = 0.0) -> float:
        """
        计算滚动阻力 (N)
        公式: F_roll = μ × m × g × cos(θ)
        :param mass: 质量（None时使用默认值）
        :param gradient_pct: 道路坡度 (%)
        :return: 滚动阻力 (N)
        """
        if mass is None:
            mass = self.mass
        theta = math.atan(gradient_pct / 100.0)    # 坡角（弧度）
        f_roll = self.rolling_resistance * mass * self.gravity * math.cos(theta)
        return f_roll

    def calculate_gradient_resistance(self, mass: float = None, gradient_pct: float = 0.0) -> float:
        """
        计算坡道阻力 (N)
        公式: F_grade = m × g × sin(θ)
        :param mass: 质量（None时使用默认值）
        :param gradient_pct: 道路坡度 (%)，正值为上坡，负值为下坡
        :return: 坡道阻力 (N)，上坡为正值
        """
        if mass is None:
            mass = self.mass
        theta = math.atan(gradient_pct / 100.0)
        f_grade = mass * self.gravity * math.sin(theta)
        return f_grade

    def calculate_required_power(
        self, speed_kmh: float, mass: float = None, gradient_pct: float = 0.0
    ) -> float:
        """
        计算行驶所需功率 (kW)
        总阻力 = 空气阻力 + 滚动阻力 + 坡道阻力
        功率 P = F_total × v
        :param speed_kmh: 车速 (km/h)
        :param mass: 质量（None时使用默认值）
        :param gradient_pct: 道路坡度 (%)
        :return: 所需功率 (kW)
        """
        if mass is None:
            mass = self.mass

        speed_ms = speed_kmh / 3.6

        f_air = self.calculate_air_resistance(speed_kmh)
        f_roll = self.calculate_rolling_resistance(mass, gradient_pct)
        f_grade = self.calculate_gradient_resistance(mass, gradient_pct)

        f_total = f_air + f_roll + f_grade   # 总阻力 (N)
        power_kw = f_total * speed_ms / 1000.0  # 转换为 kW

        return max(0.0, power_kw)    # 功率不为负

    def validate_sensor_consistency(self, sensor_data: dict) -> dict:
        """
        验证传感器数据的物理一致性
        基于物理规律检测异常传感器读数组合
        :param sensor_data: 传感器数值字典
        :return: 包含 'valid' (bool) 和 'issues' (list) 的验证结果
        """
        issues: List[str] = []

        # 提取传感器值（带默认值）
        speed = sensor_data.get("vehicle_speed", 0.0)
        engine_rpm = sensor_data.get("engine_rpm", 0.0)
        engine_temp = sensor_data.get("engine_temp", 85.0)
        coolant_temp = sensor_data.get("coolant_temp", 85.0)
        oil_pressure = sensor_data.get("oil_pressure", 40.0)
        battery_voltage = sensor_data.get("battery_voltage", 12.6)
        battery_soc = sensor_data.get("battery_soc", 80.0)
        motor_current = sensor_data.get("motor_current", 50.0)
        brake_pressure = sensor_data.get("brake_pressure", 0.0)
        throttle = sensor_data.get("throttle_position", 0.0)
        gradient = sensor_data.get("road_gradient", 0.0)

        # 规则1：发动机超温检测
        if engine_temp > 105.0:
            issues.append(f"发动机温度过高: {engine_temp:.1f}°C (阈值105°C)")

        # 规则2：冷却液温度与发动机温度一致性
        if abs(engine_temp - coolant_temp) > 25.0:
            issues.append(
                f"发动机温度({engine_temp:.1f}°C)与冷却液温度({coolant_temp:.1f}°C)差值异常"
            )

        # 规则3：行驶时机油压力不应过低
        if speed > 10.0 and engine_rpm > 500 and oil_pressure < 15.0:
            issues.append(f"行驶中机油压力过低: {oil_pressure:.1f} psi")

        # 规则4：油门与刹车同时踩踏（物理矛盾）
        if throttle > 20.0 and brake_pressure > 30.0:
            issues.append(
                f"油门({throttle:.1f}%)和制动({brake_pressure:.1f} bar)同时施加，疑似传感器故障"
            )

        # 规则5：静止时发动机转速异常
        if speed < 2.0 and engine_rpm > 3000.0:
            issues.append(f"静止时发动机转速异常高: {engine_rpm:.0f} rpm")

        # 规则6：高速时制动压力仍高（可疑）
        if speed > 100.0 and brake_pressure > 80.0:
            issues.append(
                f"高速({speed:.0f} km/h)时制动压力过高({brake_pressure:.0f} bar)，疑似制动系统故障"
            )

        # 规则7：电池电压低于正常范围
        if battery_voltage < 11.5:
            issues.append(f"电池电压过低: {battery_voltage:.2f} V")

        # 规则8：估算功率与电机电流一致性
        required_power = self.calculate_required_power(speed, gradient_pct=gradient)
        estimated_current = required_power * 1000 / max(battery_voltage, 1.0)  # 简化估算
        if speed > 30.0 and abs(motor_current - estimated_current) > 200.0:
            issues.append(
                f"电机电流({motor_current:.0f}A)与估算值({estimated_current:.0f}A)差距过大"
            )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "issue_count": len(issues),
        }
