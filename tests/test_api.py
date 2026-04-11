"""
FastAPI接口单元测试
使用TestClient测试所有HTTP端点
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# 标准传感器数据（正常工况）
NORMAL_SENSOR_DATA = {
    "engine_rpm": 1800.0,
    "engine_temp": 88.0,
    "oil_pressure": 45.0,
    "coolant_temp": 87.0,
    "battery_voltage": 13.2,
    "battery_temp": 28.0,
    "battery_soc": 75.0,
    "motor_current": 120.0,
    "motor_temp": 45.0,
    "brake_pressure": 5.0,
    "tire_pressure_fl": 34.0,
    "tire_pressure_fr": 34.0,
    "tire_pressure_rl": 33.0,
    "tire_pressure_rr": 33.0,
    "vibration_level": 1.5,
    "fuel_consumption": 8.5,
    "vehicle_speed": 75.0,
    "throttle_position": 35.0,
    "ambient_temp": 22.0,
    "humidity": 55.0,
    "road_gradient": 2.0,
    "altitude": 150.0,
}


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi未安装，跳过API测试")
class TestAPIHealth(unittest.TestCase):
    """健康检查接口测试"""

    @classmethod
    def setUpClass(cls):
        """初始化测试客户端"""
        from serving.api_server import app
        cls.client = TestClient(app)

    def test_health_endpoint_returns_200(self):
        """健康检查接口应返回200状态码"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_health_response_schema(self):
        """健康检查响应应包含必要字段"""
        response = self.client.get("/health")
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        self.assertIn("timestamp", data)

    def test_health_status_is_healthy(self):
        """健康检查状态应为'healthy'"""
        response = self.client.get("/health")
        data = response.json()
        self.assertEqual(data["status"], "healthy")


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi未安装，跳过API测试")
class TestModelInfo(unittest.TestCase):
    """模型信息接口测试"""

    @classmethod
    def setUpClass(cls):
        from serving.api_server import app
        cls.client = TestClient(app)

    def test_model_info_returns_200(self):
        """模型信息接口应返回200"""
        response = self.client.get("/model/info")
        self.assertEqual(response.status_code, 200)

    def test_model_info_schema(self):
        """模型信息响应应包含配置字段"""
        response = self.client.get("/model/info")
        data = response.json()
        self.assertIn("model_type", data)
        self.assertIn("input_dim", data)
        self.assertIn("num_classes", data)
        self.assertIn("seq_len", data)

    def test_model_info_correct_values(self):
        """模型配置值应正确"""
        response = self.client.get("/model/info")
        data = response.json()
        self.assertEqual(data["input_dim"], 22)
        self.assertEqual(data["num_classes"], 8)


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi未安装，跳过API测试")
class TestPredictEndpoint(unittest.TestCase):
    """预测接口测试"""

    @classmethod
    def setUpClass(cls):
        from serving.api_server import app
        cls.client = TestClient(app)

    def test_predict_returns_200(self):
        """预测接口应返回200"""
        response = self.client.post("/predict", json=NORMAL_SENSOR_DATA)
        self.assertEqual(response.status_code, 200)

    def test_predict_response_schema(self):
        """预测响应应包含必要字段"""
        response = self.client.post("/predict", json=NORMAL_SENSOR_DATA)
        data = response.json()
        required_fields = [
            "fault_type", "fault_name", "probability",
            "severity", "alert_level", "repair_advice",
        ]
        for field in required_fields:
            self.assertIn(field, data, f"响应缺少字段: {field}")

    def test_predict_fault_type_range(self):
        """预测的故障类型应在有效范围内"""
        response = self.client.post("/predict", json=NORMAL_SENSOR_DATA)
        data = response.json()
        self.assertGreaterEqual(data["fault_type"], -1)
        self.assertLess(data["fault_type"], 8)

    def test_predict_probability_range(self):
        """预测概率应在[0, 1]范围内"""
        response = self.client.post("/predict", json=NORMAL_SENSOR_DATA)
        data = response.json()
        self.assertGreaterEqual(data["probability"], 0.0)
        self.assertLessEqual(data["probability"], 1.0)

    def test_predict_severity_range(self):
        """严重程度应在[0, 1]范围内"""
        response = self.client.post("/predict", json=NORMAL_SENSOR_DATA)
        data = response.json()
        self.assertGreaterEqual(data["severity"], 0.0)
        self.assertLessEqual(data["severity"], 1.0)


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi未安装，跳过API测试")
class TestBatchPredictEndpoint(unittest.TestCase):
    """批量预测接口测试"""

    @classmethod
    def setUpClass(cls):
        from serving.api_server import app
        cls.client = TestClient(app)

    def test_batch_predict_returns_200(self):
        """批量预测接口应返回200"""
        batch = [NORMAL_SENSOR_DATA, NORMAL_SENSOR_DATA]
        response = self.client.post("/predict/batch", json=batch)
        self.assertEqual(response.status_code, 200)

    def test_batch_predict_count(self):
        """批量预测返回数量应与输入数量一致"""
        batch = [NORMAL_SENSOR_DATA] * 3
        response = self.client.post("/predict/batch", json=batch)
        data = response.json()
        self.assertEqual(data["count"], 3)
        self.assertEqual(len(data["predictions"]), 3)

    def test_empty_batch_returns_400(self):
        """空批次请求应返回400"""
        response = self.client.post("/predict/batch", json=[])
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
