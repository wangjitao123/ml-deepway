"""
ONNX模型导出脚本
将PyTorch检查点导出为ONNX格式
"""

import os
import sys
import argparse
import logging
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="车辆故障预测模型ONNX导出脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="PyTorch检查点路径(.pth)"
    )
    parser.add_argument(
        "--output", type=str, default="model.onnx",
        help="ONNX输出文件路径(.onnx)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=30, dest="seq_len",
        help="输入序列长度"
    )
    parser.add_argument(
        "--input-dim", type=int, default=22, dest="input_dim",
        help="输入特征维度"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        logger.error(f"检查点文件不存在: {args.checkpoint}")
        sys.exit(1)

    from models.lstm_transformer import LSTMTransformerModel
    from models.onnx_export import OnnxExporter

    # 加载模型
    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = LSTMTransformerModel(
        input_dim=args.input_dim,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_classes=8,
        seq_len=args.seq_len,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"模型已加载: {args.checkpoint}")

    # 导出ONNX
    exporter = OnnxExporter(
        model=model,
        device=device,
        seq_len=args.seq_len,
        input_dim=args.input_dim,
    )

    output_path = exporter.export(args.output)
    logger.info(f"ONNX模型已导出: {output_path}")

    # 性能测试
    benchmark = exporter.benchmark_onnx(output_path, n_runs=50)
    if benchmark:
        logger.info(f"推理延迟: {benchmark.get('avg_latency_ms', 0):.2f}ms")


if __name__ == "__main__":
    main()
