"""
MQTT消息消费者模块
用于从MQTT Broker订阅并接收传感器数据
"""

import json
import time
import logging
import threading
import queue
from typing import Optional, Callable, Dict, Any

# 优雅处理可选依赖
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    mqtt = None
    MQTT_AVAILABLE = False
    logging.warning("paho-mqtt 未安装，MQTT消费者将以Mock模式运行")

logger = logging.getLogger(__name__)


class MQTTConsumer:
    """
    MQTT数据消费者
    订阅MQTT主题并接收车辆传感器数据
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        client_id: str = "vehicle_fault_consumer",
        username: Optional[str] = None,
        password: Optional[str] = None,
        keepalive: int = 60,
    ):
        """
        初始化MQTT消费者
        :param broker_host: MQTT Broker 主机地址
        :param broker_port: MQTT Broker 端口（默认1883）
        :param client_id: 客户端ID
        :param username: 认证用户名
        :param password: 认证密码
        :param keepalive: 心跳保活间隔（秒）
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        self.username = username
        self.password = password
        self.keepalive = keepalive

        self.is_connected = False       # 连接状态标志
        self.is_running = False         # 运行状态标志

        # 消息缓冲队列
        self._message_queue: queue.Queue = queue.Queue(maxsize=5000)
        # 主题回调函数映射
        self._topic_callbacks: Dict[str, Callable] = {}

        self._client = None
        if MQTT_AVAILABLE:
            self._client = mqtt.Client(client_id=self.client_id)
            self._setup_callbacks()

    def _setup_callbacks(self):
        """绑定MQTT事件回调函数"""
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self.on_message

    def _on_connect(self, client, userdata, flags, rc):
        """MQTT连接成功回调"""
        if rc == 0:
            self.is_connected = True
            logger.info(f"MQTT已连接到 {self.broker_host}:{self.broker_port}")
        else:
            logger.error(f"MQTT连接失败，返回码: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """MQTT断连回调"""
        self.is_connected = False
        logger.warning(f"MQTT连接断开，返回码: {rc}")

    def connect(self) -> bool:
        """
        连接MQTT Broker
        :return: 连接成功返回True
        """
        if not MQTT_AVAILABLE:
            logger.info("MQTT Mock模式：跳过连接")
            self.is_connected = True
            return True

        try:
            if self.username:
                self._client.username_pw_set(self.username, self.password)

            self._client.connect(
                self.broker_host, self.broker_port, self.keepalive
            )
            return True
        except Exception as e:
            logger.error(f"MQTT连接失败: {e}")
            return False

    def subscribe(self, topic: str, qos: int = 1, callback: Optional[Callable] = None):
        """
        订阅MQTT主题
        :param topic: 主题字符串（支持通配符 # 和 +）
        :param qos: 服务质量等级（0/1/2）
        :param callback: 收到消息时的回调函数
        """
        if callback:
            self._topic_callbacks[topic] = callback

        if self._client and self.is_connected:
            self._client.subscribe(topic, qos)
            logger.info(f"已订阅主题: {topic} (QoS={qos})")

    def on_message(self, client, userdata, message):
        """
        消息接收回调
        :param client: MQTT客户端
        :param userdata: 用户数据
        :param message: 接收到的MQTT消息
        """
        try:
            payload_str = message.payload.decode("utf-8")
            payload = json.loads(payload_str)

            msg_data = {
                "topic": message.topic,
                "payload": payload,
                "timestamp": time.time(),
                "qos": message.qos,
            }

            # 放入队列
            try:
                self._message_queue.put_nowait(msg_data)
            except queue.Full:
                logger.warning("MQTT消息队列已满，丢弃旧消息")
                self._message_queue.get_nowait()
                self._message_queue.put_nowait(msg_data)

            # 触发主题回调
            for topic_pattern, callback in self._topic_callbacks.items():
                if self._topic_matches(topic_pattern, message.topic):
                    callback(msg_data)

        except json.JSONDecodeError:
            logger.warning(f"无法解析JSON消息: {message.payload}")
        except Exception as e:
            logger.error(f"处理MQTT消息时出错: {e}")

    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """
        检查主题是否匹配模式（支持MQTT通配符）
        :param pattern: 主题模式（可含 # 和 +）
        :param topic: 实际主题
        :return: 是否匹配
        """
        if pattern == topic:
            return True
        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")
        for p, t in zip(pattern_parts, topic_parts):
            if p == "#":
                return True
            if p != "+" and p != t:
                return False
        return len(pattern_parts) == len(topic_parts)

    def get_message(self, timeout: float = 1.0) -> Optional[dict]:
        """
        从队列获取一条消息
        :param timeout: 超时时间（秒）
        :return: 消息字典或None
        """
        try:
            return self._message_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def start(self):
        """启动MQTT消息循环（非阻塞）"""
        self.is_running = True
        if not MQTT_AVAILABLE:
            logger.info("MQTT Mock模式：跳过消息循环")
            return

        if self._client:
            self._client.loop_start()
            logger.info("MQTT消息循环已启动")

    def stop(self):
        """停止MQTT消息循环并断开连接"""
        self.is_running = False
        if self._client and MQTT_AVAILABLE:
            self._client.loop_stop()
            self._client.disconnect()
        logger.info("MQTT消费者已停止")
