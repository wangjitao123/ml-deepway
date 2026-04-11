"""
ONNX模型导出模块
将PyTorch模型导出为ONNX格式以便跨平台部署
"""

import os
import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class OnnxExporter:
    """
    ONNX模型导出器
    支持动态批次维度，方便生产环境部署
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        seq_len: int = 30,
        input_dim: int = 22,
        opset_version: int = 17,
    ):
        """
        初始化导出器
        :param model: 待导出的PyTorch模型
        :param device: 计算设备
        :param seq_len: 序列长度
        :param input_dim: 输入特征维度
        :param opset_version: ONNX算子集版本
        """
        self.model = model
        self.device = device or torch.device("cpu")
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.opset_version = opset_version

        self.model.to(self.device)
        self.model.eval()

    def export(
        self,
        output_path: str,
        batch_size: int = 1,
        verbose: bool = False,
    ) -> str:
        """
        导出模型为ONNX格式
        :param output_path: 输出文件路径（.onnx）
        :param batch_size: 示例批次大小（用于追踪）
        :param verbose: 是否输出详细日志
        :return: 导出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # 构造示例输入张量
        dummy_input = torch.randn(
            batch_size, self.seq_len, self.input_dim,
            device=self.device
        )

        # 动态轴配置：批次维度设为动态
        dynamic_axes = {
            "input": {0: "batch_size"},          # 输入批次维度动态
            "fault_logits": {0: "batch_size"},   # 分类输出批次维度动态
            "severity": {0: "batch_size"},        # 严重程度输出批次维度动态
        }

        # 自定义前向传播包装（ONNX只支持张量输出，不支持字典）
        class OnnxWrapper(nn.Module):
            """包装模型以输出元组（ONNX要求）"""
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                outputs = self.model(x)
                return outputs["fault_logits"], outputs["severity"]

        wrapper = OnnxWrapper(self.model).to(self.device)
        wrapper.eval()

        try:
            torch.onnx.export(
                wrapper,
                dummy_input,
                output_path,
                input_names=["input"],
                output_names=["fault_logits", "severity"],
                dynamic_axes=dynamic_axes,
                opset_version=self.opset_version,
                verbose=verbose,
                do_constant_folding=True,   # 常量折叠优化
                export_params=True,         # 导出训练参数
            )
            logger.info(f"ONNX模型已导出: {output_path}")

            # 验证导出的ONNX模型
            self._verify_onnx(output_path, dummy_input)
            return output_path

        except Exception as e:
            logger.error(f"ONNX导出失败: {e}")
            raise

    def _verify_onnx(self, onnx_path: str, dummy_input: torch.Tensor):
        """
        验证导出的ONNX模型的正确性
        :param onnx_path: ONNX文件路径
        :param dummy_input: 验证用示例输入
        """
        try:
            import onnx
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            logger.info("ONNX模型验证通过")
        except ImportError:
            logger.warning("onnx 未安装，跳过模型验证")
        except Exception as e:
            logger.warning(f"ONNX模型验证警告: {e}")

    def benchmark_onnx(self, onnx_path: str, n_runs: int = 100) -> dict:
        """
        对ONNX模型进行推理性能基准测试
        :param onnx_path: ONNX文件路径
        :param n_runs: 推理次数
        :return: 性能指标字典
        """
        try:
            import onnxruntime as ort
            import time

            session = ort.InferenceSession(onnx_path)
            dummy_input = np.random.randn(1, self.seq_len, self.input_dim).astype(np.float32)

            # 预热
            for _ in range(10):
                session.run(None, {"input": dummy_input})

            # 计时
            start = time.time()
            for _ in range(n_runs):
                session.run(None, {"input": dummy_input})
            elapsed = time.time() - start

            avg_latency_ms = elapsed / n_runs * 1000
            throughput = n_runs / elapsed

            logger.info(f"ONNX推理延迟: {avg_latency_ms:.2f}ms | 吞吐量: {throughput:.1f} 次/秒")
            return {
                "avg_latency_ms": avg_latency_ms,
                "throughput_per_sec": throughput,
                "n_runs": n_runs,
            }
        except ImportError:
            logger.warning("onnxruntime 未安装，跳过性能测试")
            return {}
        except Exception as e:
            logger.error(f"ONNX基准测试失败: {e}")
            return {}
