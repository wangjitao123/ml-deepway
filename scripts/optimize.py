"""
超参数优化脚本
使用贝叶斯优化搜索最优模型超参数
"""

import os
import sys
import argparse
import logging
import yaml
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="车辆故障预测超参数优化脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-trials", type=int, default=30, dest="n_trials",
        help="优化试验次数"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--output", type=str, default="best_hyperparams.json",
        help="最优超参数输出文件"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from physics.virtual_sample_generator import VirtualSampleGenerator
    from data.preprocessing import DataPreprocessor
    from data.dataset import create_dataloaders
    from optimizer.bayesian_optimizer import BayesianOptimizer

    logger.info(f"开始超参数优化，共 {args.n_trials} 次试验...")

    # 生成训练数据（快速版）
    gen = VirtualSampleGenerator(seed=42)
    X, y = gen.generate_per_class_dataset(n_per_class=100)

    preprocessor = DataPreprocessor(method="zscore", seq_len=30)
    X_norm = preprocessor.fit_transform(X)
    X_seq, y_seq = preprocessor.create_sequences(X_norm, y)

    train_loader, val_loader, _ = create_dataloaders(X_seq, y_seq, batch_size=32)

    # 运行贝叶斯优化
    optimizer = BayesianOptimizer(
        train_loader=train_loader,
        val_loader=val_loader,
        n_trials=args.n_trials,
        study_name="vehicle_fault_hpo",
    )

    best_params = optimizer.optimize()
    optimizer.print_summary()

    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    logger.info(f"最优超参数已保存: {args.output}")
    logger.info(f"最优超参数: {best_params}")


if __name__ == "__main__":
    main()
