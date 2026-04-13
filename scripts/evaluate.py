"""
模型评估脚本
加载检查点，在测试集上运行全面评估并保存报告
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="车辆故障预测模型评估脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="模型检查点路径(.pth)"
    )
    parser.add_argument(
        "--data-path", type=str, default=None, dest="data_path",
        help="测试数据X文件路径(.npy)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation_results", dest="output_dir",
        help="评估结果输出目录"
    )
    parser.add_argument(
        "--n-per-class", type=int, default=200, dest="n_per_class",
        help="每类评估样本数（无数据文件时使用虚拟样本）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 检查检查点文件
    if not os.path.exists(args.checkpoint):
        logger.error(f"检查点文件不存在: {args.checkpoint}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    from physics.virtual_sample_generator import VirtualSampleGenerator
    from data.preprocessing import DataPreprocessor
    from data.dataset import create_dataloaders
    from models.lstm_transformer import LSTMTransformerModel
    from models.evaluator import FaultEvaluator

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = LSTMTransformerModel(
        input_dim=22, hidden_dim=128, num_layers=2,
        num_heads=4, num_classes=8, seq_len=30,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"模型已加载: {args.checkpoint}")

    # 加载或生成数据
    if args.data_path and os.path.exists(args.data_path):
        X = np.load(args.data_path)
        y_path = args.data_path.replace("_X.npy", "_y.npy")
        y = np.load(y_path) if os.path.exists(y_path) else np.zeros(len(X), dtype=np.int64)
    else:
        logger.info("使用虚拟样本进行评估...")
        gen = VirtualSampleGenerator(seed=100)
        X, y = gen.generate_per_class_dataset(n_per_class=args.n_per_class)

    # 预处理
    preprocessor = DataPreprocessor(method="zscore", seq_len=30)
    X_norm = preprocessor.fit_transform(X)
    X_seq, y_seq = preprocessor.create_sequences(X_norm, y)

    _, _, test_loader = create_dataloaders(X_seq, y_seq, batch_size=32)
    logger.info(f"测试集规模: {len(test_loader.dataset)}")

    # 评估
    evaluator = FaultEvaluator(model, device=device, output_dir=args.output_dir)
    metrics = evaluator.evaluate(test_loader)

    # 打印结果
    logger.info(f"\n{'='*50}")
    logger.info(f"评估结果:")
    logger.info(f"  准确率: {metrics['accuracy']:.4f}")
    logger.info(f"  宏F1: {metrics['f1_macro']:.4f}")
    logger.info(f"  加权F1: {metrics['f1_weighted']:.4f}")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"{'='*50}")

    # 保存可视化
    evaluator.plot_confusion_matrix()
    evaluator.plot_roc_curve()

    # 导出报告
    report_path = evaluator.export_report(metrics)
    logger.info(f"评估报告已保存: {report_path}")


if __name__ == "__main__":
    main()
