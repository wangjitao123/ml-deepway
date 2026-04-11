"""
CAN总线数据读取模块
支持真实CAN总线通信及Mock模式用于测试
"""

import time
import threading
import queue
import logging
from typing import Optional, Dict, Callable

# 优雅处理可选依赖
try:
    import can
    CAN_AVAILABLE = True
except ImportError:
    CAN_AVAILABLE = False
    logging.warning("python-can 未安装，将使用Mock模式")

logger = logging.getLogger(__name__)


class CANBusReader:
    """
    CAN总线读取器
    封装CAN总线消息读取、解码和分发功能
    支持Mock模式方便单元测试
    """

    # 常用CAN消息ID映射（示例）
    MESSAGE_MAP = {
        0x100: "engine_status",     # 发动机状态帧
        0x200: "battery_status",    # 电池状态帧
        0x300: "brake_status",      # 制动状态帧
        0x400: "tire_pressure",     # 轮胎压力帧
        0x500: "motor_status",      # 电机状态帧
    }

    def __init__(
        self,
        channel: str = "vcan0",
        bustype: str = "socketcan",
        mock_mode: bool = False,
    ):
        """
        初始化CAN总线读取器
        :param channel: CAN通道名称（如 'vcan0'）
        :param bustype: 总线类型（如 'socketcan', 'pcan'）
        :param mock_mode: 是否使用Mock模式（True时不需要真实硬件）
        """
        self.channel = channel
        self.bustype = bustype
        # 若无法导入can库，自动切换到Mock模式
        self.mock_mode = mock_mode or not CAN_AVAILABLE

        self.bus = None                    # CAN总线实例
        self.is_running = False            # 运行状态标志
        self._thread: Optional[threading.Thread] = None
        self._message_queue = queue.Queue(maxsize=1000)   # 消息缓冲队列
        self._callbacks: Dict[int, Callable] = {}         # 消息ID回调函数

    def connect(self) -> bool:
        """
        连接CAN总线
        :return: 连接成功返回True
        """
        if self.mock_mode:
            logger.info("CAN总线 Mock模式已激活")
            return True

        try:
            self.bus = can.interface.Bus(
                channel=self.channel,
                bustype=self.bustype,
            )
            logger.info(f"CAN总线连接成功: {self.channel} ({self.bustype})")
            return True
        except Exception as e:
            logger.error(f"CAN总线连接失败: {e}，切换到Mock模式")
            self.mock_mode = True
            return False

    def read_message(self, timeout: float = 1.0) -> Optional[dict]:
        """
        读取单条CAN消息
        :param timeout: 超时时间（秒）
        :return: 消息字典或None
        """
        if self.mock_mode:
            return self._generate_mock_message()

        if self.bus is None:
            raise RuntimeError("CAN总线未连接，请先调用 connect()")

        try:
            msg = self.bus.recv(timeout=timeout)
            if msg is not None:
                return self.decode_message(msg)
        except Exception as e:
            logger.error(f"读取CAN消息失败: {e}")
        return None

    def decode_message(self, msg) -> dict:
        """
        解码CAN消息为字典格式
        :param msg: can.Message 实例
        :return: 解码后的字典
        """
        decoded = {
            "arbitration_id": msg.arbitration_id,
            "data": list(msg.data),
            "timestamp": msg.timestamp,
            "channel": self.channel,
        }

        # 根据消息ID进行特定解码
        msg_type = self.MESSAGE_MAP.get(msg.arbitration_id, "unknown")
        decoded["message_type"] = msg_type

        if msg_type == "engine_status" and len(msg.data) >= 4:
            # 字节0-1：转速，字节2-3：温度
            decoded["engine_rpm"] = (msg.data[0] << 8 | msg.data[1]) * 0.25
            decoded["engine_temp"] = msg.data[2] - 40

        elif msg_type == "battery_status" and len(msg.data) >= 4:
            # 字节0-1：电压（0.01V精度），字节2：SOC
            decoded["battery_voltage"] = (msg.data[0] << 8 | msg.data[1]) * 0.01
            decoded["battery_soc"] = msg.data[2]

        return decoded

    def _generate_mock_message(self) -> dict:
        """
        生成模拟CAN消息（用于测试）
        :return: 模拟消息字典
        """
        import random
        import math

        # 模拟发动机状态帧
        t = time.time()
        return {
            "arbitration_id": 0x100,
            "message_type": "engine_status",
            "engine_rpm": 1200 + 200 * math.sin(t),
            "engine_temp": 85 + 5 * math.sin(t * 0.1),
            "timestamp": t,
            "channel": "mock",
        }

    def register_callback(self, arbitration_id: int, callback: Callable):
        """
        注册消息ID的回调函数
        :param arbitration_id: CAN消息仲裁ID
        :param callback: 收到消息时调用的回调函数
        """
        self._callbacks[arbitration_id] = callback

    def start_reading(self):
        """启动后台线程持续读取CAN消息"""
        self.is_running = True
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name="CAN-Reader"
        )
        self._thread.start()
        logger.info("CAN总线读取线程已启动")

    def _read_loop(self):
        """后台读取循环（在独立线程中运行）"""
        while self.is_running:
            msg = self.read_message(timeout=0.1)
            if msg is not None:
                # 放入队列
                try:
                    self._message_queue.put_nowait(msg)
                except queue.Full:
                    pass  # 队列满时丢弃旧消息

                # 触发注册的回调函数
                arb_id = msg.get("arbitration_id")
                if arb_id in self._callbacks:
                    try:
                        self._callbacks[arb_id](msg)
                    except Exception as e:
                        logger.error(f"回调函数执行失败: {e}")

            # Mock模式下增加延迟避免空转
            if self.mock_mode:
                time.sleep(0.05)

    def get_message(self, timeout: float = 1.0) -> Optional[dict]:
        """
        从队列中获取消息
        :param timeout: 等待超时时间（秒）
        :return: 消息字典或None
        """
        try:
            return self._message_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """停止CAN总线读取，释放资源"""
        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        if self.bus is not None and not self.mock_mode:
            try:
                self.bus.shutdown()
            except Exception as e:
                logger.error(f"关闭CAN总线失败: {e}")

        logger.info("CAN总线读取器已停止")
