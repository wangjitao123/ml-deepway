"""
时序数据库模块
封装InfluxDB客户端，用于存储和查询车辆传感器时序数据
"""

import logging
import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

# 优雅处理可选依赖
try:
    from influxdb_client import InfluxDBClient as _InfluxDBClient
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    _InfluxDBClient = None
    SYNCHRONOUS = None
    INFLUXDB_AVAILABLE = False
    logging.warning("influxdb-client 未安装，时序数据库将以Mock模式运行")

logger = logging.getLogger(__name__)


class TimeSeriesDB:
    """
    InfluxDB时序数据库封装类
    提供传感器数据写入和查询功能
    支持无influxdb-client时的Mock模式
    """

    def __init__(
        self,
        url: str = "http://localhost:8086",
        token: str = "my-token",
        org: str = "vehicle-org",
        bucket: str = "vehicle-sensors",
    ):
        """
        初始化时序数据库连接
        :param url: InfluxDB地址
        :param token: 认证Token
        :param org: 组织名称
        :param bucket: 数据桶名称
        """
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket

        self._client = None
        self._write_api = None
        self._query_api = None

        # Mock存储（当influxdb不可用时）
        self._mock_store: List[dict] = []

        if INFLUXDB_AVAILABLE:
            self._init_client()

    def _init_client(self):
        """初始化InfluxDB客户端"""
        try:
            self._client = _InfluxDBClient(
                url=self.url, token=self.token, org=self.org
            )
            self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
            self._query_api = self._client.query_api()
            logger.info(f"InfluxDB客户端已初始化: {self.url}")
        except Exception as e:
            logger.error(f"InfluxDB初始化失败: {e}")
            self._client = None

    def write_sensor_data(
        self,
        measurement: str,
        tags: Dict[str, str],
        fields: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        写入传感器数据到时序数据库
        :param measurement: 测量名称（如 'vehicle_sensor'）
        :param tags: 标签字典（如 {'vehicle_id': 'V001'}）
        :param fields: 字段字典（传感器数值）
        :param timestamp: 时间戳，None时使用当前时间
        :return: 写入成功返回True
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        if not INFLUXDB_AVAILABLE or self._write_api is None:
            # Mock模式：存储到内存
            self._mock_store.append({
                "measurement": measurement,
                "tags": tags,
                "fields": fields,
                "timestamp": timestamp,
            })
            return True

        try:
            from influxdb_client import Point
            point = Point(measurement)

            for tag_key, tag_val in tags.items():
                point = point.tag(tag_key, tag_val)

            for field_key, field_val in fields.items():
                point = point.field(field_key, float(field_val))

            point = point.time(timestamp)
            self._write_api.write(bucket=self.bucket, org=self.org, record=point)
            return True

        except Exception as e:
            logger.error(f"写入InfluxDB失败: {e}")
            return False

    def query_recent(self, measurement: str, minutes: int = 10) -> List[Dict]:
        """
        查询最近N分钟的数据
        :param measurement: 测量名称
        :param minutes: 查询时间范围（分钟）
        :return: 数据点列表
        """
        if not INFLUXDB_AVAILABLE or self._query_api is None:
            # 返回Mock存储中的最近数据
            cutoff = time.time() - minutes * 60
            return [
                r for r in self._mock_store
                if r["measurement"] == measurement
            ][-100:]

        try:
            flux_query = f"""
            from(bucket: "{self.bucket}")
                |> range(start: -{minutes}m)
                |> filter(fn: (r) => r._measurement == "{measurement}")
            """
            tables = self._query_api.query(flux_query, org=self.org)
            records = []
            for table in tables:
                for record in table.records:
                    records.append({
                        "time": record.get_time(),
                        "field": record.get_field(),
                        "value": record.get_value(),
                    })
            return records
        except Exception as e:
            logger.error(f"InfluxDB查询失败: {e}")
            return []

    def query_range(
        self,
        measurement: str,
        start: datetime,
        stop: datetime,
    ) -> List[Dict]:
        """
        查询指定时间范围内的数据
        :param measurement: 测量名称
        :param start: 开始时间
        :param stop: 结束时间
        :return: 数据点列表
        """
        if not INFLUXDB_AVAILABLE or self._query_api is None:
            return [
                r for r in self._mock_store
                if r["measurement"] == measurement
                and start <= r["timestamp"] <= stop
            ]

        try:
            start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
            stop_str = stop.strftime("%Y-%m-%dT%H:%M:%SZ")
            flux_query = f"""
            from(bucket: "{self.bucket}")
                |> range(start: {start_str}, stop: {stop_str})
                |> filter(fn: (r) => r._measurement == "{measurement}")
            """
            tables = self._query_api.query(flux_query, org=self.org)
            records = []
            for table in tables:
                for record in table.records:
                    records.append({
                        "time": record.get_time(),
                        "field": record.get_field(),
                        "value": record.get_value(),
                    })
            return records
        except Exception as e:
            logger.error(f"InfluxDB范围查询失败: {e}")
            return []

    def close(self):
        """关闭数据库连接，释放资源"""
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.error(f"关闭InfluxDB连接失败: {e}")
        logger.info("InfluxDB连接已关闭")
