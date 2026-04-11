"""
模型训练脚本
支持命令行参数和YAML配置文件的完整训练流程
"""

import os
import sys
import argparse
import logging
import yaml
import numpy as np
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="车辆故障预测模型训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="训练轮次（覆盖配置文件）"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, dest="batch_size",
        help="批次大小（覆盖配置文件）"
    )
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints", dest="output_dir",
        help="模型检查点输出目录"
    )
    parser.add_argument(
        "--data-path", type=str, default=None, dest="data_path",
        help="预先生成的数据文件路径（npy格式）"
    )
    parser.add_argument(
        "--n-per-class", type=int, default=300, dest="n_per_class",
        help="每类别生成的样本数（使用虚拟样本时）"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"配置已加载: {config_path}")
    return config


def main():
    """完整训练流程"""
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 命令行参数覆盖配置文件
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    epochs = args.epochs or train_cfg.get("epochs", 50)
    batch_size = args.batch_size or train_cfg.get("batch_size", 32)
    lr = train_cfg.get("learning_rate", 0.001)
    patience = train_cfg.get("early_stopping_patience", 10)

    input_dim = model_cfg.get("input_dim", 22)
    hidden_dim = model_cfg.get("hidden_dim", 128)
    num_layers = model_cfg.get("num_layers", 2)
    num_heads = model_cfg.get("num_heads", 4)
    num_classes = model_cfg.get("num_classes", 8)
    seq_len = model_cfg.get("seq_len", 30)
    dropout = model_cfg.get("dropout", 0.2)

    os.makedirs(args.output_dir, exist_ok=True)

    # 导入模块
    from physics.virtual_sample_generator import VirtualSampleGenerator
    from data.preprocessing import DataPreprocessor
    from data.dataset import create_dataloaders
    from models.lstm_transformer import LSTMTransformerModel
    from models.trainer import FaultTrainer
    from physics.constraint_loss import PhysicsConstraintLoss

    # 数据加载或生成
    if args.data_path and os.path.exists(args.data_path):
        logger.info(f"加载数据: {args.data_path}")
        X = np.load(args.data_path)
        y_path = args.data_path.replace("_X.npy", "_y.npy")
        y = np.load(y_path) if os.path.exists(y_path) else np.zeros(len(X), dtype=np.int64)
    else:
        logger.info(f"生成虚拟数据，每类 {args.n_per_class} 个样本...")
        gen = VirtualSampleGenerator(seed=42)
        X, y = gen.generate_per_class_dataset(n_per_class=args.n_per_class)

    logger.info(f"数据规模: X={X.shape}, y={y.shape}")

    # 预处理
    preprocessor = DataPreprocessor(method="zscore", seq_len=seq_len)
    X_norm = preprocessor.fit_transform(X)
    X_seq, y_seq = preprocessor.create_sequences(X_norm, y)
    logger.info(f"序列数据: X={X_seq.shape}, y={y_seq.shape}")

    # 创建DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        X_seq, y_seq, batch_size=batch_size
    )
    logger.info(
        f"数据集: 训练{len(train_loader.dataset)}, 验证{len(val_loader.dataset)}, 测试{len(test_loader.dataset)}"
    )

    # 构建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMTransformerModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_classes=num_classes,
        seq_len=seq_len,
        dropout=dropout,
    )

    physics_loss = PhysicsConstraintLoss()

    trainer = FaultTrainer(
        model=model,
        device=device,
        learning_rate=lr,
        physics_loss_fn=physics_loss,
        checkpoint_dir=args.output_dir,
    )

    logger.info(f"开始训练：{epochs} 轮，设备: {device}")
    history = trainer.train(
        train_loader, val_loader,
        epochs=epochs,
        early_stopping_patience=patience,
    )

    # 保存最终检查点
    trainer.save_checkpoint(epochs, {"final": True}, filename="final_model.pth")
    logger.info(f"训练完成！最终模型已保存至: {args.output_dir}/final_model.pth")

    # 打印训练摘要
    best_val_acc = max(history["val_accuracy"]) if history["val_accuracy"] else 0.0
    logger.info(f"最优验证准确率: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
