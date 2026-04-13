"""
OBD-II车载诊断数据读取模块
通过OBD接口读取车辆实时传感器数据
"""

import logging
import time
from typing import Optional, Dict, Any

# 优雅处理可选依赖
try:
    import obd
    OBD_AVAILABLE = True
except ImportError:
    obd = None
    OBD_AVAILABLE = False
    logging.warning("python-obd 未安装，OBD读取器将以Mock模式运行")

logger = logging.getLogger(__name__)


class OBDReader:
    """
    OBD-II数据读取器
    读取车辆标准诊断接口数据，支持Mock模式
    """

    # OBD-II标准命令列表（映射到传感器特征）
    OBD_COMMANDS = {
        "engine_rpm": "RPM",                # 发动机转速
        "engine_temp": "COOLANT_TEMP",      # 冷却液温度（即发动机温度）
        "vehicle_speed": "SPEED",           # 车速
        "throttle_position": "THROTTLE_POS",# 油门位置
        "fuel_consumption": "FUEL_RATE",    # 燃油消耗率
    }

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 38400,
        timeout: float = 10.0,
        mock_mode: bool = False,
    ):
        """
        初始化OBD读取器
        :param port: 串口设备路径（如 '/dev/ttyUSB0' 或 'COM3'）
        :param baudrate: 串口波特率
        :param timeout: 连接超时时间（秒）
        :param mock_mode: 是否使用Mock模式
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.mock_mode = mock_mode or not OBD_AVAILABLE

        self._connection = None    # OBD连接实例
        self.is_connected = False  # 连接状态

    def connect(self) -> bool:
        """
        连接OBD接口
        :return: 连接成功返回True
        """
        if self.mock_mode:
            logger.info("OBD Mock模式已激活")
            self.is_connected = True
            return True

        try:
            # 尝试连接OBD接口
            if self.port:
                self._connection = obd.OBD(
                    portstr=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                )
            else:
                # 自动检测端口
                self._connection = obd.OBD(timeout=self.timeout)

            if self._connection.is_connected():
                self.is_connected = True
                logger.info(f"OBD接口连接成功: {self._connection.port_name()}")
                return True
            else:
                logger.warning("OBD接口连接失败，切换到Mock模式")
                self.mock_mode = True
                self.is_connected = True
                return False

        except Exception as e:
            logger.error(f"OBD连接异常: {e}，切换到Mock模式")
            self.mock_mode = True
            self.is_connected = True
            return False

    def read_data(self, command_name: str) -> Optional[float]:
        """
        读取单个OBD数据项
        :param command_name: OBD命令名称（如 'RPM'）
        :return: 数值或None
        """
        if self.mock_mode:
            return self._mock_read(command_name)

        if not self.is_connected:
            raise RuntimeError("OBD未连接")

        try:
            cmd = getattr(obd.commands, command_name, None)
            if cmd is None:
                logger.warning(f"未知OBD命令: {command_name}")
                return None

            response = self._connection.query(cmd)
            if not response.is_null():
                return float(response.value.magnitude)
        except Exception as e:
            logger.error(f"OBD读取 {command_name} 失败: {e}")
        return None

    def _mock_read(self, command_name: str) -> float:
        """
        生成模拟OBD数据
        :param command_name: 命令名称
        :return: 模拟数值
        """
        import math
        t = time.time()
        mock_values = {
            "RPM": 1500 + 300 * math.sin(t * 0.5),
            "COOLANT_TEMP": 88 + 5 * math.sin(t * 0.1),
            "SPEED": 60 + 20 * math.sin(t * 0.2),
            "THROTTLE_POS": 30 + 10 * math.sin(t * 0.3),
            "FUEL_RATE": 8.5 + 1.5 * math.sin(t * 0.15),
        }
        return mock_values.get(command_name, 0.0)

    def get_sensor_values(self) -> Dict[str, Optional[float]]:
        """
        一次性读取所有支持的传感器值
        :return: 传感器名称到数值的字典
        """
        values = {}
        for sensor_name, obd_cmd in self.OBD_COMMANDS.items():
            values[sensor_name] = self.read_data(obd_cmd)
        return values

    def get_supported_commands(self) -> list:
        """
        获取当前车辆支持的OBD命令列表
        :return: 支持的命令名称列表
        """
        if self.mock_mode:
            return list(self.OBD_COMMANDS.values())

        if not self.is_connected or self._connection is None:
            return []

        try:
            supported = self._connection.supported_commands
            return [str(cmd) for cmd in supported]
        except Exception:
            return []

    def disconnect(self):
        """断开OBD连接，释放资源"""
        if self._connection is not None and not self.mock_mode:
            try:
                self._connection.close()
            except Exception as e:
                logger.error(f"关闭OBD连接失败: {e}")

        self.is_connected = False
        logger.info("OBD连接已断开")
