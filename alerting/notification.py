"""
通知管理模块
支持Webhook和邮件通知，根据告警级别分发通知
"""

import json
import time
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any
from urllib import request, error

logger = logging.getLogger(__name__)


class NotificationManager:
    """
    通知管理器
    支持HTTP Webhook、邮件等多种通知渠道
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        email_config: Optional[dict] = None,
        min_level_for_webhook: int = 1,   # WARNING及以上触发Webhook
        min_level_for_email: int = 2,     # CRITICAL及以上触发邮件
    ):
        """
        初始化通知管理器
        :param webhook_url: 默认Webhook URL
        :param email_config: 邮件配置字典，包含smtp_host/port/user/password/to
        :param min_level_for_webhook: 触发Webhook的最低告警级别
        :param min_level_for_email: 触发邮件通知的最低告警级别
        """
        self.webhook_url = webhook_url
        self.email_config = email_config or {}
        self.min_level_for_webhook = min_level_for_webhook
        self.min_level_for_email = min_level_for_email

        # 通知历史（记录发送结果）
        self.notification_history = []

    def send_webhook(
        self,
        url: str,
        payload: Dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> bool:
        """
        发送HTTP Webhook通知（带重试机制）
        :param url: Webhook目标URL
        :param payload: 通知载荷（字典格式，将被序列化为JSON）
        :param max_retries: 最大重试次数
        :param retry_delay: 重试间隔（秒）
        :return: 发送成功返回True
        """
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        for attempt in range(1, max_retries + 1):
            try:
                req = request.Request(
                    url,
                    data=payload_bytes,
                    headers={
                        "Content-Type": "application/json; charset=utf-8",
                        "User-Agent": "VehicleFaultPredictor/1.0",
                    },
                    method="POST",
                )
                with request.urlopen(req, timeout=10) as response:
                    if response.status < 400:
                        logger.info(f"Webhook发送成功: {url} (尝试{attempt}次)")
                        self._record_notification("webhook", url, True, attempt)
                        return True
                    else:
                        logger.warning(
                            f"Webhook返回错误状态码: {response.status} (尝试{attempt}/{max_retries})"
                        )

            except error.URLError as e:
                logger.warning(f"Webhook发送失败: {e} (尝试{attempt}/{max_retries})")
            except Exception as e:
                logger.error(f"Webhook发送异常: {e} (尝试{attempt}/{max_retries})")

            if attempt < max_retries:
                time.sleep(retry_delay * attempt)   # 指数退避

        logger.error(f"Webhook发送最终失败，已重试{max_retries}次: {url}")
        self._record_notification("webhook", url, False, max_retries)
        return False

    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        is_html: bool = False,
    ) -> bool:
        """
        发送邮件通知
        注意：当前为Mock实现，生产环境需配置SMTP服务器
        :param to: 收件人邮箱
        :param subject: 邮件主题
        :param body: 邮件正文
        :param is_html: 是否为HTML格式
        :return: 发送成功返回True
        """
        smtp_host = self.email_config.get("smtp_host")
        smtp_port = self.email_config.get("smtp_port", 587)
        username = self.email_config.get("username")
        password = self.email_config.get("password")
        from_addr = self.email_config.get("from", username or "noreply@vehicle.ai")

        if not smtp_host or not username:
            # Mock模式：仅记录日志
            logger.info(f"[Mock邮件] 收件人: {to} | 主题: {subject}")
            logger.debug(f"[Mock邮件正文] {body[:200]}")
            self._record_notification("email", to, True, 1)
            return True

        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = from_addr
            msg["To"] = to
            msg["Subject"] = subject

            content_type = "html" if is_html else "plain"
            msg.attach(MIMEText(body, content_type, "utf-8"))

            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.login(username, password)
                server.sendmail(from_addr, [to], msg.as_string())

            logger.info(f"邮件发送成功: {to} | {subject}")
            self._record_notification("email", to, True, 1)
            return True

        except Exception as e:
            logger.error(f"邮件发送失败: {e}")
            self._record_notification("email", to, False, 1)
            return False

    def notify(self, alert) -> Dict[str, bool]:
        """
        根据告警级别分发通知
        :param alert: Alert对象（来自alerting.alert_engine）
        :return: 各通知渠道的发送结果字典
        """
        results = {}
        alert_dict = alert.to_dict() if hasattr(alert, "to_dict") else dict(alert)
        alert_level = alert.level.value if hasattr(alert.level, "value") else int(alert.level)

        # Webhook通知（WARNING及以上）
        if alert_level >= self.min_level_for_webhook and self.webhook_url:
            payload = {
                "event": "vehicle_fault_alert",
                "alert": alert_dict,
            }
            results["webhook"] = self.send_webhook(self.webhook_url, payload)

        # 邮件通知（CRITICAL及以上）
        if alert_level >= self.min_level_for_email:
            to_email = self.email_config.get("to", "admin@vehicle.ai")
            subject = f"[车辆故障告警] {alert_dict.get('level_name', '')} - {alert_dict.get('fault_name', '')}"
            body = self._format_email_body(alert_dict)
            results["email"] = self.send_email(to_email, subject, body, is_html=True)

        return results

    def _format_email_body(self, alert_dict: dict) -> str:
        """
        格式化告警邮件正文（HTML格式）
        :param alert_dict: 告警字典
        :return: HTML格式邮件正文
        """
        level_colors = {
            "信息": "#17a2b8",
            "警告": "#ffc107",
            "严重": "#fd7e14",
            "紧急": "#dc3545",
            "停机": "#721c24",
        }
        level_name = alert_dict.get("level_name", "告警")
        color = level_colors.get(level_name, "#6c757d")

        return f"""
<html><body>
<h2 style="color:{color}">车辆故障告警 - {level_name}</h2>
<table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse">
  <tr><td><b>车辆ID</b></td><td>{alert_dict.get('vehicle_id', 'N/A')}</td></tr>
  <tr><td><b>故障类型</b></td><td>{alert_dict.get('fault_name', 'N/A')}</td></tr>
  <tr><td><b>故障概率</b></td><td>{alert_dict.get('probability', 0):.1%}</td></tr>
  <tr><td><b>严重程度</b></td><td>{alert_dict.get('severity', 0):.1%}</td></tr>
  <tr><td><b>告警时间</b></td><td>{alert_dict.get('timestamp', 'N/A')}</td></tr>
  <tr><td><b>维修建议</b></td><td style="color:red">{alert_dict.get('repair_advice', 'N/A')}</td></tr>
</table>
</body></html>
"""

    def _record_notification(
        self, channel: str, target: str, success: bool, attempts: int
    ):
        """记录通知历史"""
        self.notification_history.append({
            "channel": channel,
            "target": target,
            "success": success,
            "attempts": attempts,
            "timestamp": time.time(),
        })
        # 保持历史记录不超过500条
        if len(self.notification_history) > 500:
            self.notification_history.pop(0)
