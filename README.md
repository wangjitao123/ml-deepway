# 工业级智能车辆故障预测系统

## 系统概述

本系统是一套基于深度学习的工业级车辆故障预测解决方案，融合 **LSTM-Transformer 混合架构**、物理约束损失和贝叶斯超参数优化，实现对8类车辆故障的实时精准预测与告警。

### 支持故障类型

| 类别 | 故障名称 | 受影响传感器 |
|:---:|:---|:---|
| 0 | 正常运行 | - |
| 1 | 发动机过热 | engine_temp, coolant_temp, oil_pressure |
| 2 | 电池异常 | battery_voltage, battery_temp, battery_soc |
| 3 | 制动失效 | brake_pressure, vibration_level |
| 4 | 轮胎气压异常 | tire_pressure_fl/fr/rl/rr |
| 5 | 电机/控制器故障 | motor_current, motor_temp |
| 6 | 润滑系统故障 | oil_pressure, vibration_level, engine_temp |
| 7 | 冷却系统故障 | coolant_temp, engine_temp |

---

## 核心特性

- **LSTM-Transformer 混合模型**：先用 LSTM 提取局部时序特征，再用 Transformer 建模长程依赖，双头输出（分类 + 严重程度回归）
- **物理约束损失**：将冷却规则、电池规则、制动规则编码为可微分损失项，引导模型学习物理合理的预测
- **对抗鲁棒训练**：噪声注入、特征丢弃、漂移模拟和尖峰干扰，提升模型对传感器异常的鲁棒性
- **贝叶斯超参数优化**：基于 Optuna TPE 算法自动搜索最优超参数，配合中位数剪枝加速搜索
- **5级告警体系**：信息 → 警告 → 严重 → 紧急 → 停机，支持 Webhook 和邮件通知
- **PSI 数据漂移检测**：群体稳定性指数（PSI）持续监控预测分布漂移
- **多协议数据接入**：CAN 总线、MQTT、OBD-II，均含 Mock 模式
- **ONNX 跨平台部署**：支持导出 ONNX 模型供边缘设备推理
- **Docker 化部署**：包含 API 服务、InfluxDB、Mosquitto、Grafana 的完整 docker-compose

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                       数据采集层                              │
│   CAN 总线  │  MQTT  │  OBD-II  │  历史数据(.npy)            │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    数据处理层                                  │
│  IQR异常值裁剪 → ZScore/MinMax归一化 → 滑动窗口序列化          │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│              LSTM-Transformer 模型层                          │
│  LSTM → 投影 → 位置编码 → TransformerEncoder → 双输出头       │
│  损失：CE + 0.1·MSE + 0.05·PhysicsLoss + 0.05·AdvLoss        │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    服务与告警层                                │
│  FastAPI REST API  │  5级告警引擎  │  Webhook/邮件通知          │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    监控与存储层                                │
│  PSI漂移检测  │  数据质量评分  │  InfluxDB  │  Grafana         │
└─────────────────────────────────────────────────────────────┘
```

---

## 22个传感器特征

| 索引 | 特征名称 | 描述 | 单位 |
|:---:|:---|:---|:---|
| 0 | engine_rpm | 发动机转速 | rpm |
| 1 | engine_temp | 发动机温度 | °C |
| 2 | oil_pressure | 机油压力 | psi |
| 3 | coolant_temp | 冷却液温度 | °C |
| 4 | battery_voltage | 电池电压 | V |
| 5 | battery_temp | 电池温度 | °C |
| 6 | battery_soc | 电池荷电状态 | % |
| 7 | motor_current | 电机电流 | A |
| 8 | motor_temp | 电机温度 | °C |
| 9 | brake_pressure | 制动压力 | bar |
| 10 | tire_pressure_fl | 左前轮胎压力 | psi |
| 11 | tire_pressure_fr | 右前轮胎压力 | psi |
| 12 | tire_pressure_rl | 左后轮胎压力 | psi |
| 13 | tire_pressure_rr | 右后轮胎压力 | psi |
| 14 | vibration_level | 振动等级 | m/s² |
| 15 | fuel_consumption | 油耗 | L/100km |
| 16 | vehicle_speed | 车速 | km/h |
| 17 | throttle_position | 油门位置 | % |
| 18 | ambient_temp | 环境温度 | °C |
| 19 | humidity | 湿度 | % |
| 20 | road_gradient | 道路坡度 | % |
| 21 | altitude | 海拔高度 | m |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行演示

```bash
python main.py
```

演示流程：
1. 生成虚拟传感器数据（8类 × 300个样本）
2. 数据预处理 + 滑动窗口序列化
3. 训练 LSTM-Transformer 模型（5轮演示）
4. 在测试集上评估，生成混淆矩阵和ROC曲线
5. 运行告警系统演示
6. 数据质量检查

### 3. 完整训练

```bash
python scripts/train.py --config config/config.yaml --epochs 50 --output-dir checkpoints
```

### 4. 模型评估

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --output-dir evaluation_results
```

### 5. 超参数优化

```bash
python scripts/optimize.py --n-trials 50
```

### 6. 导出 ONNX

```bash
python scripts/export_onnx.py --checkpoint checkpoints/best_model.pth --output model.onnx
```

### 7. 启动 API 服务

```bash
uvicorn serving.api_server:app --host 0.0.0.0 --port 8000
```

### 8. Docker 部署

```bash
cd deploy
docker-compose up -d
```

---

## API 文档

服务启动后访问 `http://localhost:8000/docs` 查看交互式 API 文档。

### 主要接口

| 方法 | 路径 | 描述 |
|:---:|:---|:---|
| GET | `/health` | 服务健康检查 |
| GET | `/model/info` | 获取模型配置信息 |
| POST | `/predict` | 单次故障预测 |
| POST | `/predict/batch` | 批量故障预测（最多100条） |

### 预测请求示例

```json
POST /predict
{
  "engine_rpm": 1800.0,
  "engine_temp": 125.0,
  "oil_pressure": 30.0,
  "coolant_temp": 115.0,
  "battery_voltage": 13.0,
  "battery_temp": 28.0,
  "battery_soc": 75.0,
  "motor_current": 120.0,
  "motor_temp": 45.0,
  "brake_pressure": 5.0,
  "tire_pressure_fl": 34.0,
  "tire_pressure_fr": 34.0,
  "tire_pressure_rl": 33.0,
  "tire_pressure_rr": 33.0,
  "vibration_level": 2.0,
  "fuel_consumption": 12.5,
  "vehicle_speed": 60.0,
  "throttle_position": 30.0,
  "ambient_temp": 38.0,
  "humidity": 40.0,
  "road_gradient": 0.0,
  "altitude": 200.0
}
```

### 预测响应示例

```json
{
  "fault_type": 1,
  "fault_name": "发动机过热",
  "probability": 0.9234,
  "severity": 0.7812,
  "alert_level": "emergency",
  "alert_level_name": "紧急",
  "repair_advice": "立即靠边停车，关闭发动机...",
  "all_probs": {
    "正常运行": 0.0123,
    "发动机过热": 0.9234,
    ...
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

---

## 项目结构

```
ml-deepway/
├── main.py                          # 系统演示主入口
├── requirements.txt                 # Python依赖包
├── setup.py                         # 包安装配置
├── README.md                        # 本文档
│
├── config/
│   └── config.yaml                  # 系统配置文件（模型/训练/告警参数）
│
├── data/
│   ├── preprocessing.py             # 数据预处理（异常值、标准化、序列化）
│   ├── dataset.py                   # PyTorch Dataset和DataLoader
│   ├── connectors/
│   │   ├── can_bus_reader.py        # CAN总线数据读取
│   │   ├── mqtt_consumer.py         # MQTT消息消费
│   │   └── obd_reader.py            # OBD-II诊断读取
│   └── storage/
│       └── timeseries_db.py         # InfluxDB时序数据库封装
│
├── models/
│   ├── lstm_transformer.py          # LSTM-Transformer模型架构
│   ├── trainer.py                   # 训练器（混合损失、早停、TensorBoard）
│   ├── evaluator.py                 # 评估器（指标、混淆矩阵、ROC）
│   └── onnx_export.py               # ONNX模型导出
│
├── physics/
│   ├── dynamics_model.py            # 车辆动力学物理模型
│   ├── virtual_sample_generator.py  # 虚拟样本生成器
│   └── constraint_loss.py           # 可微分物理约束损失
│
├── optimizer/
│   ├── bayesian_optimizer.py        # 贝叶斯超参数优化（Optuna）
│   └── adversarial_trainer.py       # 对抗训练数据增强
│
├── serving/
│   ├── api_server.py                # FastAPI REST服务
│   └── predictor.py                 # 预测器（预处理+推理+告警）
│
├── alerting/
│   ├── alert_engine.py              # 5级告警引擎
│   └── notification.py              # Webhook/邮件通知
│
├── monitoring/
│   ├── model_monitor.py             # PSI数据漂移检测
│   └── data_quality.py              # 数据质量评分
│
├── deploy/
│   ├── Dockerfile                   # 多阶段Docker构建
│   └── docker-compose.yaml          # 完整服务编排
│
├── scripts/
│   ├── train.py                     # 训练脚本
│   ├── evaluate.py                  # 评估脚本
│   ├── optimize.py                  # 超参数优化脚本
│   ├── export_onnx.py               # ONNX导出脚本
│   └── generate_demo_data.py        # 演示数据生成脚本
│
└── tests/
    ├── test_preprocessing.py        # 预处理单元测试
    ├── test_model.py                # 模型单元测试
    ├── test_physics.py              # 物理模型单元测试
    ├── test_trainer.py              # 训练器单元测试
    └── test_api.py                  # API接口单元测试
```

---

## 运行单元测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行指定测试文件
python -m pytest tests/test_model.py -v
python -m pytest tests/test_physics.py -v

# 使用unittest运行
python -m unittest discover tests/
```

---

## 告警级别说明

| 级别 | 概率阈值 | 描述 | 建议行动 |
|:---:|:---:|:---|:---|
| 正常 | < 30% | 系统运行正常 | 定期保养 |
| 信息 | ≥ 30% | 轻微异常，记录观察 | 下次保养检查 |
| 警告 | ≥ 50% | 需要注意 | 尽快安排检查 |
| 严重 | ≥ 70% | 需要尽快处理 | 24小时内维修 |
| 紧急 | ≥ 85% | 立即停车检查 | 立即停车，联系救援 |
| 停机 | ≥ 95% | 立即关闭系统 | 禁止行驶，立即大修 |

---

## 许可证

MIT License - Copyright © 2024 ML DeepWay Team
