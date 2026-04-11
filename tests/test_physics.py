"""
物理模型单元测试
测试VehicleDynamicsModel和VirtualSampleGenerator
"""

import unittest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.dynamics_model import VehicleDynamicsModel
from physics.virtual_sample_generator import VirtualSampleGenerator
from physics.constraint_loss import PhysicsConstraintLoss


class TestVehicleDynamicsModel(unittest.TestCase):
    """车辆动力学模型测试"""

    def setUp(self):
        """初始化动力学模型"""
        self.model = VehicleDynamicsModel(
            mass=1500.0,
            drag_coefficient=0.3,
            rolling_resistance=0.015,
            frontal_area=2.2,
        )

    def test_air_resistance_zero_speed(self):
        """零速度时空气阻力应为0"""
        f_drag = self.model.calculate_air_resistance(0.0)
        self.assertAlmostEqual(f_drag, 0.0, places=5)

    def test_air_resistance_increases_with_speed(self):
        """空气阻力应随速度增加（平方关系）"""
        f1 = self.model.calculate_air_resistance(60.0)
        f2 = self.model.calculate_air_resistance(120.0)
        self.assertGreater(f2, f1)
        # 速度翻倍，阻力应约为4倍
        self.assertAlmostEqual(f2 / f1, 4.0, delta=0.1)

    def test_rolling_resistance_positive(self):
        """滚动阻力应为正值"""
        f_roll = self.model.calculate_rolling_resistance()
        self.assertGreater(f_roll, 0.0)

    def test_gradient_resistance_uphill(self):
        """上坡时坡道阻力为正"""
        f_grade = self.model.calculate_gradient_resistance(gradient_pct=5.0)
        self.assertGreater(f_grade, 0.0)

    def test_gradient_resistance_downhill(self):
        """下坡时坡道阻力为负"""
        f_grade = self.model.calculate_gradient_resistance(gradient_pct=-5.0)
        self.assertLess(f_grade, 0.0)

    def test_required_power_zero_speed(self):
        """零速度时所需功率应接近0"""
        power = self.model.calculate_required_power(0.0)
        self.assertAlmostEqual(power, 0.0, places=3)

    def test_required_power_increases_with_speed(self):
        """功率需求随速度增加"""
        p1 = self.model.calculate_required_power(60.0)
        p2 = self.model.calculate_required_power(120.0)
        self.assertGreater(p2, p1)

    def test_validate_normal_sensor_data(self):
        """正常传感器数据应通过验证"""
        normal_data = {
            "vehicle_speed": 80.0,
            "engine_rpm": 2000.0,
            "engine_temp": 88.0,
            "coolant_temp": 87.0,
            "oil_pressure": 45.0,
            "battery_voltage": 13.2,
            "battery_soc": 75.0,
            "motor_current": 120.0,
            "brake_pressure": 5.0,
            "throttle_position": 35.0,
            "road_gradient": 2.0,
        }
        result = self.model.validate_sensor_consistency(normal_data)
        self.assertIn("valid", result)
        self.assertIn("issues", result)
        self.assertIsInstance(result["issues"], list)

    def test_validate_overheated_engine(self):
        """发动机过热应被检测到"""
        hot_data = {
            "engine_temp": 120.0,  # 超过阈值
            "coolant_temp": 88.0,
            "vehicle_speed": 80.0,
            "engine_rpm": 2000.0,
            "oil_pressure": 45.0,
            "battery_voltage": 13.2,
            "motor_current": 120.0,
            "brake_pressure": 5.0,
            "throttle_position": 35.0,
            "road_gradient": 0.0,
        }
        result = self.model.validate_sensor_consistency(hot_data)
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["issues"]), 0)


class TestVirtualSampleGenerator(unittest.TestCase):
    """虚拟样本生成器测试"""

    def setUp(self):
        """初始化生成器（固定种子）"""
        self.gen = VirtualSampleGenerator(seed=42)
        self.n_features = 22

    def test_normal_sample_shape(self):
        """正常样本形状应为(22,)"""
        sample = self.gen.generate_normal_sample()
        self.assertEqual(sample.shape, (self.n_features,))

    def test_normal_sample_dtype(self):
        """正常样本数据类型应为float32"""
        sample = self.gen.generate_normal_sample()
        self.assertEqual(sample.dtype, np.float32)

    def test_fault_sample_shape(self):
        """所有故障类型的样本形状应为(22,)"""
        for fault_type in range(1, 8):
            sample = self.gen.generate_fault_sample(fault_type)
            self.assertEqual(
                sample.shape, (self.n_features,),
                f"故障类型{fault_type}的样本形状不正确"
            )

    def test_invalid_fault_type(self):
        """无效故障类型应抛出ValueError"""
        with self.assertRaises(ValueError):
            self.gen.generate_fault_sample(0)
        with self.assertRaises(ValueError):
            self.gen.generate_fault_sample(8)

    def test_engine_overheat_sample(self):
        """发动机过热故障样本的发动机温度应高于正常值"""
        normal_temps = [self.gen.generate_normal_sample()[1] for _ in range(50)]
        fault_temps = [self.gen.generate_fault_sample(1)[1] for _ in range(50)]
        self.assertGreater(np.mean(fault_temps), np.mean(normal_temps))

    def test_battery_fault_voltage(self):
        """电池故障样本的电池电压应低于正常值"""
        normal_voltages = [self.gen.generate_normal_sample()[4] for _ in range(50)]
        fault_voltages = [self.gen.generate_fault_sample(2)[4] for _ in range(50)]
        self.assertLess(np.mean(fault_voltages), np.mean(normal_voltages))

    def test_generate_dataset(self):
        """测试数据集生成"""
        X, y = self.gen.generate_dataset(n_samples=100, fault_ratio=0.5)
        self.assertEqual(X.shape, (100, self.n_features))
        self.assertEqual(y.shape, (100,))
        self.assertTrue(0 in y)  # 应有正常样本
        self.assertTrue(any(y > 0))  # 应有故障样本

    def test_generate_per_class_dataset(self):
        """测试每类均衡数据集生成"""
        n_per_class = 20
        X, y = self.gen.generate_per_class_dataset(n_per_class=n_per_class)
        self.assertEqual(X.shape, (n_per_class * 8, self.n_features))
        # 每个类别应有约n_per_class个样本
        for cls in range(8):
            count = np.sum(y == cls)
            self.assertEqual(count, n_per_class, f"类别{cls}样本数不正确: {count}")


class TestPhysicsConstraintLoss(unittest.TestCase):
    """物理约束损失函数测试"""

    def setUp(self):
        """初始化物理约束损失"""
        self.loss_fn = PhysicsConstraintLoss()
        self.batch_size = 4
        self.seq_len = 30
        self.n_features = 22

    def test_forward_returns_scalar(self):
        """前向传播应返回标量损失"""
        sensor_seq = torch.randn(self.batch_size, self.seq_len, self.n_features)
        predictions = {"fault_probs": torch.softmax(torch.randn(self.batch_size, 8), dim=-1)}
        loss = self.loss_fn(predictions, sensor_seq)
        self.assertEqual(loss.dim(), 0)  # 应为标量

    def test_loss_non_negative(self):
        """损失值应为非负数"""
        sensor_seq = torch.randn(self.batch_size, self.seq_len, self.n_features)
        predictions = {"fault_probs": torch.softmax(torch.randn(self.batch_size, 8), dim=-1)}
        loss = self.loss_fn(predictions, sensor_seq)
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_backward_differentiable(self):
        """损失应支持反向传播（完全可微分）"""
        sensor_seq = torch.randn(
            self.batch_size, self.seq_len, self.n_features, requires_grad=True
        )
        predictions = {}
        loss = self.loss_fn(predictions, sensor_seq)
        loss.backward()
        # 梯度应不为None
        self.assertIsNotNone(sensor_seq.grad)

    def test_no_numpy_in_forward(self):
        """测试前向传播中不使用numpy（验证可微分性）"""
        # 只要backward()不报错，就证明是纯torch操作
        sensor_seq = torch.randn(
            self.batch_size, self.seq_len, self.n_features, requires_grad=True
        )
        predictions = {}
        loss = self.loss_fn(predictions, sensor_seq)
        try:
            loss.backward()
            backward_success = True
        except Exception as e:
            backward_success = False
            print(f"反向传播失败: {e}")
        self.assertTrue(backward_success, "物理约束损失的反向传播应成功")


if __name__ == "__main__":
    unittest.main(verbosity=2)
