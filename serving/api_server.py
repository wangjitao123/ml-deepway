"""
FastAPI REST服务模块
提供车辆故障预测的HTTP API接口
"""

import os
import sys
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

# FastAPI和Pydantic
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ──────────────── Pydantic 请求/响应模型 ────────────────

class SensorData(BaseModel):
    """传感器数据请求模型（包含22个传感器字段）"""

    # 发动机相关
    engine_rpm: float = Field(default=1800.0, ge=0, le=8000, description="发动机转速(rpm)")
    engine_temp: float = Field(default=88.0, ge=40, le=150, description="发动机温度(°C)")
    oil_pressure: float = Field(default=45.0, ge=0, le=100, description="机油压力(psi)")
    coolant_temp: float = Field(default=87.0, ge=40, le=140, description="冷却液温度(°C)")

    # 电池相关
    battery_voltage: float = Field(default=13.2, ge=0, le=20, description="电池电压(V)")
    battery_temp: float = Field(default=28.0, ge=-30, le=80, description="电池温度(°C)")
    battery_soc: float = Field(default=75.0, ge=0, le=100, description="电池荷电状态(%)")

    # 电机相关
    motor_current: float = Field(default=120.0, ge=0, le=800, description="电机电流(A)")
    motor_temp: float = Field(default=45.0, ge=0, le=120, description="电机温度(°C)")

    # 制动相关
    brake_pressure: float = Field(default=5.0, ge=0, le=200, description="制动压力(bar)")

    # 轮胎气压
    tire_pressure_fl: float = Field(default=34.0, ge=10, le=60, description="左前轮胎压(psi)")
    tire_pressure_fr: float = Field(default=34.0, ge=10, le=60, description="右前轮胎压(psi)")
    tire_pressure_rl: float = Field(default=33.0, ge=10, le=60, description="左后轮胎压(psi)")
    tire_pressure_rr: float = Field(default=33.0, ge=10, le=60, description="右后轮胎压(psi)")

    # 其他传感器
    vibration_level: float = Field(default=1.5, ge=0, le=20, description="振动等级(m/s²)")
    fuel_consumption: float = Field(default=8.5, ge=0, le=40, description="油耗(L/100km)")
    vehicle_speed: float = Field(default=75.0, ge=0, le=300, description="车速(km/h)")
    throttle_position: float = Field(default=35.0, ge=0, le=100, description="油门位置(%)")
    ambient_temp: float = Field(default=22.0, ge=-50, le=80, description="环境温度(°C)")
    humidity: float = Field(default=55.0, ge=0, le=100, description="湿度(%)")
    road_gradient: float = Field(default=2.0, ge=-50, le=50, description="道路坡度(%)")
    altitude: float = Field(default=150.0, ge=0, le=6000, description="海拔高度(m)")

    # 可选字段
    seq_len: int = Field(default=30, ge=1, le=300, description="序列长度")
    vehicle_id: Optional[str] = Field(default=None, description="车辆ID")

    def to_feature_dict(self) -> dict:
        """转换为传感器特征字典"""
        return {
            "engine_rpm": self.engine_rpm,
            "engine_temp": self.engine_temp,
            "oil_pressure": self.oil_pressure,
            "coolant_temp": self.coolant_temp,
            "battery_voltage": self.battery_voltage,
            "battery_temp": self.battery_temp,
            "battery_soc": self.battery_soc,
            "motor_current": self.motor_current,
            "motor_temp": self.motor_temp,
            "brake_pressure": self.brake_pressure,
            "tire_pressure_fl": self.tire_pressure_fl,
            "tire_pressure_fr": self.tire_pressure_fr,
            "tire_pressure_rl": self.tire_pressure_rl,
            "tire_pressure_rr": self.tire_pressure_rr,
            "vibration_level": self.vibration_level,
            "fuel_consumption": self.fuel_consumption,
            "vehicle_speed": self.vehicle_speed,
            "throttle_position": self.throttle_position,
            "ambient_temp": self.ambient_temp,
            "humidity": self.humidity,
            "road_gradient": self.road_gradient,
            "altitude": self.altitude,
        }


class PredictionResponse(BaseModel):
    """预测响应模型"""
    fault_type: int
    fault_name: str
    probability: float
    severity: float
    alert_level: str
    alert_level_name: str
    repair_advice: str
    all_probs: Dict[str, float] = {}
    timestamp: str = ""
    vehicle_id: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """批量预测响应"""
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    timestamp: str
    version: str = "1.0.0"


class ModelInfoResponse(BaseModel):
    """模型信息响应"""
    model_type: str
    input_dim: int
    num_classes: int
    seq_len: int
    device: str
    is_loaded: bool


# ──────────────── FastAPI 应用 ────────────────

app = FastAPI(
    title="车辆故障预测API",
    description="工业级智能车辆故障预测系统 REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局预测器（启动时初始化）
_predictor = None


def get_predictor():
    """获取预测器单例"""
    global _predictor
    if _predictor is None:
        try:
            from serving.predictor import FaultPredictor
            _predictor = FaultPredictor()
            logger.info("FaultPredictor初始化完成")
        except Exception as e:
            logger.warning(f"FaultPredictor初始化失败: {e}")
    return _predictor


@app.on_event("startup")
async def startup_event():
    """服务启动事件"""
    logger.info("车辆故障预测API服务启动")
    get_predictor()


@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check():
    """
    服务健康检查接口
    返回服务状态和模型加载情况
    """
    predictor = get_predictor()
    model_loaded = predictor is not None and predictor.model is not None
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/model/info", response_model=ModelInfoResponse, summary="模型信息")
async def model_info():
    """获取模型配置和状态信息"""
    predictor = get_predictor()
    if predictor is None:
        return ModelInfoResponse(
            model_type="LSTMTransformer",
            input_dim=22,
            num_classes=8,
            seq_len=30,
            device="unknown",
            is_loaded=False,
        )
    return ModelInfoResponse(
        model_type="LSTMTransformerModel",
        input_dim=predictor.input_dim,
        num_classes=predictor.num_classes,
        seq_len=predictor.seq_len,
        device=str(predictor.device),
        is_loaded=predictor.model is not None,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="单次故障预测",
)
async def predict(sensor_data: SensorData):
    """
    提交传感器数据，获取故障预测结果
    """
    predictor = get_predictor()

    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="预测服务不可用，模型加载失败",
        )

    try:
        feature_dict = sensor_data.to_feature_dict()
        result = predictor.predict(feature_dict)
        result["timestamp"] = datetime.now().isoformat()
        result["vehicle_id"] = sensor_data.vehicle_id
        return PredictionResponse(**{k: v for k, v in result.items() if k in PredictionResponse.model_fields})
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"预测执行失败: {str(e)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="批量故障预测",
)
async def predict_batch(sensor_data_list: List[SensorData]):
    """
    批量提交传感器数据，获取批量故障预测结果
    """
    if len(sensor_data_list) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请求列表不能为空",
        )

    if len(sensor_data_list) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="单次批量请求不超过100条",
        )

    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="预测服务不可用",
        )

    predictions = []
    for sensor_data in sensor_data_list:
        try:
            feature_dict = sensor_data.to_feature_dict()
            result = predictor.predict(feature_dict)
            result["timestamp"] = datetime.now().isoformat()
            result["vehicle_id"] = sensor_data.vehicle_id
            pred = PredictionResponse(**{k: v for k, v in result.items() if k in PredictionResponse.model_fields})
            predictions.append(pred)
        except Exception as e:
            logger.error(f"批量预测单条失败: {e}")
            predictions.append(PredictionResponse(
                fault_type=-1,
                fault_name="预测失败",
                probability=0.0,
                severity=0.0,
                alert_level="normal",
                alert_level_name="正常",
                repair_advice=f"预测失败: {str(e)}",
                timestamp=datetime.now().isoformat(),
            ))

    return BatchPredictionResponse(predictions=predictions, count=len(predictions))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
