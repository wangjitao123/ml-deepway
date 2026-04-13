"""
虚拟样本生成器
基于物理规律生成各类故障场景的传感器数据
用于训练数据增强和测试
"""

import numpy as np
from typing import Tuple


class VirtualSampleGenerator:
    """
    虚拟传感器样本生成器
    生成正常和各类故障状态下的传感器读数

    传感器特征索引（共22个）:
    0:engine_rpm, 1:engine_temp, 2:oil_pressure, 3:coolant_temp,
    4:battery_voltage, 5:battery_temp, 6:battery_soc, 7:motor_current,
    8:motor_temp, 9:brake_pressure, 10:tire_pressure_fl, 11:tire_pressure_fr,
    12:tire_pressure_rl, 13:tire_pressure_rr, 14:vibration_level,
    15:fuel_consumption, 16:vehicle_speed, 17:throttle_position,
    18:ambient_temp, 19:humidity, 20:road_gradient, 21:altitude

    故障类型:
    0:正常, 1:发动机过热, 2:电池异常, 3:制动失效,
    4:轮胎气压, 5:电机/控制器故障, 6:润滑系统故障, 7:冷却系统故障
    """

    def __init__(self, seed: int = None):
        """
        初始化生成器
        :param seed: 随机种子，None表示不固定
        """
        self.rng = np.random.default_rng(seed)

        # 正常运行时各传感器的基准值（均值）
        self.normal_base = np.array([
            1800.0,   # engine_rpm      发动机转速（rpm）
            88.0,     # engine_temp     发动机温度（°C）
            45.0,     # oil_pressure    机油压力（psi）
            87.0,     # coolant_temp    冷却液温度（°C）
            13.2,     # battery_voltage 电池电压（V）
            28.0,     # battery_temp    电池温度（°C）
            75.0,     # battery_soc     荷电状态（%）
            120.0,    # motor_current   电机电流（A）
            45.0,     # motor_temp      电机温度（°C）
            5.0,      # brake_pressure  制动压力（bar）
            34.0,     # tire_pressure_fl 左前轮胎（psi）
            34.0,     # tire_pressure_fr 右前轮胎（psi）
            33.0,     # tire_pressure_rl 左后轮胎（psi）
            33.0,     # tire_pressure_rr 右后轮胎（psi）
            1.5,      # vibration_level 振动（m/s²）
            8.5,      # fuel_consumption 油耗（L/100km）
            75.0,     # vehicle_speed   车速（km/h）
            35.0,     # throttle_position 油门（%）
            22.0,     # ambient_temp    环境温度（°C）
            55.0,     # humidity        湿度（%）
            2.0,      # road_gradient   坡度（%）
            150.0,    # altitude        海拔（m）
        ], dtype=np.float32)

        # 正常噪声标准差
        self.normal_std = np.array([
            200.0, 4.0, 5.0, 4.0, 0.3, 3.0, 5.0, 20.0,
            5.0, 2.0, 1.5, 1.5, 1.5, 1.5, 0.5, 1.5,
            15.0, 8.0, 5.0, 10.0, 3.0, 50.0
        ], dtype=np.float32)

    def generate_normal_sample(self) -> np.ndarray:
        """
        生成正常工况传感器样本
        :return: 形状 (22,) 的传感器数值数组
        """
        sample = self.rng.normal(self.normal_base, self.normal_std).astype(np.float32)
        # 裁剪到物理合理范围
        sample = self._clip_to_valid_range(sample)
        return sample

    def generate_fault_sample(self, fault_type: int) -> np.ndarray:
        """
        生成特定故障类型的传感器样本
        :param fault_type: 故障类型 (1~7)
        :return: 形状 (22,) 的传感器数值数组（包含故障特征）
        """
        # 从正常样本出发，添加故障特征
        sample = self.generate_normal_sample()

        if fault_type == 1:
            # 故障类型1：发动机过热
            # 表现：发动机温度升高、冷却液温度高、机油压力下降、油耗增加
            sample[1] += self.rng.uniform(20.0, 40.0)   # engine_temp 升高20~40°C
            sample[3] += self.rng.uniform(15.0, 30.0)   # coolant_temp 升高
            sample[2] -= self.rng.uniform(5.0, 15.0)    # oil_pressure 下降
            sample[15] *= self.rng.uniform(1.2, 1.5)    # fuel_consumption 增加
            sample[0] = self.rng.uniform(1200, 1600)    # engine_rpm 降低（热保护）

        elif fault_type == 2:
            # 故障类型2：电池异常
            # 表现：电池电压降低、电池温度升高、SOC异常下降
            sample[4] -= self.rng.uniform(1.0, 2.5)     # battery_voltage 下降
            sample[5] += self.rng.uniform(10.0, 20.0)   # battery_temp 升高
            sample[6] -= self.rng.uniform(15.0, 35.0)   # battery_soc 下降
            sample[7] += self.rng.uniform(30.0, 80.0)   # motor_current 异常升高

        elif fault_type == 3:
            # 故障类型3：制动失效
            # 表现：制动压力异常、振动增加、ABS介入
            sample[9] -= self.rng.uniform(2.0, 4.0)     # brake_pressure 下降
            sample[14] += self.rng.uniform(2.0, 5.0)    # vibration_level 增加
            sample[16] = self.rng.uniform(50.0, 120.0)  # vehicle_speed 高速（危险场景）

        elif fault_type == 4:
            # 故障类型4：轮胎气压异常
            # 表现：某个或多个轮胎气压偏低
            affected_tires = self.rng.choice([0, 1, 2, 3], size=self.rng.integers(1, 3), replace=False)
            for tire in affected_tires:
                sample[10 + tire] -= self.rng.uniform(6.0, 15.0)  # 气压下降6~15 psi
            sample[14] += self.rng.uniform(0.5, 2.0)   # vibration_level 轻微增加
            sample[15] *= self.rng.uniform(1.05, 1.2)  # fuel_consumption 增加

        elif fault_type == 5:
            # 故障类型5：电机/控制器故障
            # 表现：电机电流异常、电机温度升高、动力异常
            sample[7] += self.rng.uniform(80.0, 200.0)  # motor_current 大幅升高
            sample[8] += self.rng.uniform(15.0, 30.0)   # motor_temp 升高
            sample[4] -= self.rng.uniform(0.5, 1.5)     # battery_voltage 下降
            sample[14] += self.rng.uniform(1.0, 3.0)    # vibration_level 增加

        elif fault_type == 6:
            # 故障类型6：润滑系统故障
            # 表现：机油压力大幅下降、发动机温度上升、振动增加
            sample[2] -= self.rng.uniform(15.0, 25.0)   # oil_pressure 大幅下降
            sample[1] += self.rng.uniform(8.0, 20.0)    # engine_temp 上升
            sample[14] += self.rng.uniform(1.5, 4.0)    # vibration_level 增加
            sample[0] = self.rng.uniform(800, 1400)     # engine_rpm 不稳定

        elif fault_type == 7:
            # 故障类型7：冷却系统故障
            # 表现：冷却液温度升高（但发动机温度因冷却不足更高）
            sample[3] += self.rng.uniform(18.0, 35.0)   # coolant_temp 大幅升高
            sample[1] += self.rng.uniform(12.0, 25.0)   # engine_temp 升高
            sample[15] *= self.rng.uniform(1.1, 1.3)    # fuel_consumption 增加
            # 冷却风扇可能故障，环境温度影响更大
            sample[0] = self.rng.uniform(1500, 2500)    # engine_rpm 降低到保护转速

        else:
            raise ValueError(f"无效故障类型: {fault_type}，应为1~7")

        # 裁剪到物理合理范围
        sample = self._clip_to_valid_range(sample)
        return sample

    def _clip_to_valid_range(self, sample: np.ndarray) -> np.ndarray:
        """
        将传感器值裁剪到物理合理范围内
        :param sample: 传感器数值数组
        :return: 裁剪后的数组
        """
        # 各传感器的最小值约束
        min_vals = np.array([
            0.0, 40.0, 0.0, 40.0, 9.0, -20.0, 0.0, 0.0,
            10.0, 0.0, 15.0, 15.0, 15.0, 15.0, 0.0, 0.0,
            0.0, 0.0, -45.0, 0.0, -40.0, 0.0
        ], dtype=np.float32)

        # 各传感器的最大值约束
        max_vals = np.array([
            8000.0, 140.0, 90.0, 130.0, 16.0, 65.0, 100.0, 700.0,
            100.0, 180.0, 55.0, 55.0, 55.0, 55.0, 15.0, 30.0,
            250.0, 100.0, 70.0, 100.0, 45.0, 5000.0
        ], dtype=np.float32)

        return np.clip(sample, min_vals, max_vals)

    def generate_dataset(
        self, n_samples: int = 1000, fault_ratio: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成完整数据集
        :param n_samples: 总样本数量
        :param fault_ratio: 故障样本比例（0~1）
        :return: (X, y) X形状 (n_samples, 22)，y形状 (n_samples,)
        """
        n_fault = int(n_samples * fault_ratio)
        n_normal = n_samples - n_fault

        samples = []
        labels = []

        # 生成正常样本
        for _ in range(n_normal):
            samples.append(self.generate_normal_sample())
            labels.append(0)

        # 生成故障样本（均匀分布在7种故障类型中）
        fault_types = list(range(1, 8))   # 故障类型1~7
        for i in range(n_fault):
            fault_type = fault_types[i % len(fault_types)]
            samples.append(self.generate_fault_sample(fault_type))
            labels.append(fault_type)

        X = np.array(samples, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)

        # 打乱数据顺序
        indices = self.rng.permutation(len(y))
        return X[indices], y[indices]

    def generate_per_class_dataset(
        self, n_per_class: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        每个类别生成相同数量的样本（均衡数据集）
        :param n_per_class: 每类样本数量
        :return: (X, y) 均衡数据集
        """
        samples = []
        labels = []

        # 正常样本
        for _ in range(n_per_class):
            samples.append(self.generate_normal_sample())
            labels.append(0)

        # 各故障类型样本
        for fault_type in range(1, 8):
            for _ in range(n_per_class):
                samples.append(self.generate_fault_sample(fault_type))
                labels.append(fault_type)

        X = np.array(samples, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)

        # 打乱顺序
        indices = self.rng.permutation(len(y))
        return X[indices], y[indices]
