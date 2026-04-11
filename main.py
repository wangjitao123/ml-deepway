"""
工业级智能车辆故障预测系统 - 主入口
演示完整流程：数据生成 → 训练 → 评估 → 告警演示
"""

import os
import sys
import logging
import numpy as np
import torch

# ── 路径配置：将项目根目录加入模块搜索路径 ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def print_banner():
    """打印系统横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║       工业级智能车辆故障预测系统 v1.0.0                       ║
║       Industrial Vehicle Fault Prediction System             ║
║       基于 LSTM-Transformer 混合架构                          ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def step1_generate_data(n_per_class: int = 300):
    """
    步骤1：生成演示数据集
    每类生成 n_per_class 个样本，共8类 × n_per_class 个样本
    :param n_per_class: 每个类别的样本数量
    :return: (X, y) 原始特征矩阵和标签数组
    """
    logger.info("=" * 60)
    logger.info(f"步骤1：生成虚拟传感器数据（每类 {n_per_class} 个样本）")

    from physics.virtual_sample_generator import VirtualSampleGenerator

    gen = VirtualSampleGenerator(seed=2024)
    X, y = gen.generate_per_class_dataset(n_per_class=n_per_class)

    logger.info(f"  数据形状: X={X.shape}, y={y.shape}")
    fault_names = [
        "正常", "发动机过热", "电池异常", "制动失效",
        "轮胎气压", "电机故障", "润滑故障", "冷却故障"
    ]
    for cls in range(8):
        count = int(np.sum(y == cls))
        logger.info(f"  类别 {cls} ({fault_names[cls]}): {count} 个样本")

    return X, y


def step2_preprocess(X: np.ndarray, y: np.ndarray, seq_len: int = 30):
    """
    步骤2：数据预处理和序列化
    :param X: 原始特征矩阵
    :param y: 标签数组
    :param seq_len: 滑动窗口长度
    :return: (preprocessor, X_seq, y_seq) 预处理器和序列数据
    """
    logger.info("=" * 60)
    logger.info("步骤2：数据预处理（IQR异常值裁剪 + ZScore标准化 + 滑动窗口）")

    from data.preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor(method="zscore", seq_len=seq_len)
    X_norm = preprocessor.fit_transform(X)
    X_seq, y_seq = preprocessor.create_sequences(X_norm, y)

    logger.info(f"  归一化后: mean≈{X_norm.mean():.3f}, std≈{X_norm.std():.3f}")
    logger.info(f"  序列数据: X={X_seq.shape}, y={y_seq.shape}")

    return preprocessor, X_seq, y_seq


def step3_create_dataloaders(X_seq: np.ndarray, y_seq: np.ndarray, batch_size: int = 32):
    """
    步骤3：创建训练/验证/测试数据加载器
    :return: (train_loader, val_loader, test_loader)
    """
    logger.info("=" * 60)
    logger.info(f"步骤3：创建DataLoader（批次大小={batch_size}）")

    from data.dataset import create_dataloaders

    train_loader, val_loader, test_loader = create_dataloaders(
        X_seq, y_seq, batch_size=batch_size
    )

    logger.info(f"  训练集: {len(train_loader.dataset)} 个样本，{len(train_loader)} 个批次")
    logger.info(f"  验证集: {len(val_loader.dataset)} 个样本")
    logger.info(f"  测试集: {len(test_loader.dataset)} 个样本")

    return train_loader, val_loader, test_loader


def step4_build_model(device: torch.device):
    """
    步骤4：构建LSTM-Transformer模型
    :return: 模型实例
    """
    logger.info("=" * 60)
    logger.info("步骤4：构建 LSTM-Transformer 混合模型")

    from models.lstm_transformer import LSTMTransformerModel

    model = LSTMTransformerModel(
        input_dim=22,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        num_classes=8,
        seq_len=30,
        dropout=0.2,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  总参数量: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")
    logger.info(f"  计算设备: {device}")

    return model


def step5_train(model, train_loader, val_loader, device, epochs: int = 5):
    """
    步骤5：训练模型
    :return: 训练历史字典
    """
    logger.info("=" * 60)
    logger.info(f"步骤5：开始训练（{epochs} 轮，物理约束损失已启用）")

    from models.trainer import FaultTrainer
    from physics.constraint_loss import PhysicsConstraintLoss

    physics_loss = PhysicsConstraintLoss()

    trainer = FaultTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        weight_decay=1e-4,
        physics_loss_fn=physics_loss,
        checkpoint_dir="checkpoints",
        use_tensorboard=False,
    )

    history = trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        early_stopping_patience=epochs + 1,  # 演示模式不早停
        save_best=True,
    )

    logger.info("  训练完成！")
    if history["train_accuracy"]:
        logger.info(f"  最终训练准确率: {history['train_accuracy'][-1]:.4f}")
        logger.info(f"  最终验证准确率: {history['val_accuracy'][-1]:.4f}")

    return trainer, history


def step6_evaluate(model, test_loader, device):
    """
    步骤6：在测试集上评估模型性能
    :return: 评估指标字典
    """
    logger.info("=" * 60)
    logger.info("步骤6：模型评估")

    from models.evaluator import FaultEvaluator

    os.makedirs("evaluation_results", exist_ok=True)
    evaluator = FaultEvaluator(
        model=model,
        device=device,
        output_dir="evaluation_results",
    )

    metrics = evaluator.evaluate(test_loader)

    logger.info(f"  准确率:     {metrics['accuracy']:.4f}")
    logger.info(f"  精确率(宏): {metrics['precision_macro']:.4f}")
    logger.info(f"  召回率(宏): {metrics['recall_macro']:.4f}")
    logger.info(f"  F1(宏):     {metrics['f1_macro']:.4f}")
    logger.info(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")

    # 保存混淆矩阵和ROC曲线图
    try:
        evaluator.plot_confusion_matrix()
        evaluator.plot_roc_curve()
        report_path = evaluator.export_report(metrics)
        logger.info(f"  评估报告: {report_path}")
        logger.info("  混淆矩阵: evaluation_results/confusion_matrix.png")
        logger.info("  ROC曲线:  evaluation_results/roc_curves.png")
    except Exception as e:
        logger.warning(f"  可视化保存失败（非关键）: {e}")

    return metrics


def step7_alert_demo(model, device):
    """
    步骤7：告警系统演示
    模拟各类故障传感器数据，展示告警生成流程
    """
    logger.info("=" * 60)
    logger.info("步骤7：告警系统演示")

    from serving.predictor import FaultPredictor
    from alerting.alert_engine import AlertEngine

    predictor = FaultPredictor(model=model, device=device)
    alert_engine = AlertEngine(vehicle_id="DEMO-V001")

    # 演示场景：模拟7种故障传感器数据
    demo_scenarios = [
        {
            "name": "正常运行",
            "data": {
                "engine_rpm": 1800.0, "engine_temp": 88.0, "oil_pressure": 45.0,
                "coolant_temp": 87.0, "battery_voltage": 13.2, "battery_temp": 28.0,
                "battery_soc": 75.0, "motor_current": 120.0, "motor_temp": 45.0,
                "brake_pressure": 5.0, "tire_pressure_fl": 34.0, "tire_pressure_fr": 34.0,
                "tire_pressure_rl": 33.0, "tire_pressure_rr": 33.0, "vibration_level": 1.5,
                "fuel_consumption": 8.5, "vehicle_speed": 75.0, "throttle_position": 35.0,
                "ambient_temp": 22.0, "humidity": 55.0, "road_gradient": 2.0, "altitude": 150.0,
            },
        },
        {
            "name": "发动机过热模拟",
            "data": {
                "engine_rpm": 1200.0, "engine_temp": 125.0, "oil_pressure": 30.0,
                "coolant_temp": 115.0, "battery_voltage": 13.0, "battery_temp": 28.0,
                "battery_soc": 75.0, "motor_current": 120.0, "motor_temp": 45.0,
                "brake_pressure": 5.0, "tire_pressure_fl": 34.0, "tire_pressure_fr": 34.0,
                "tire_pressure_rl": 33.0, "tire_pressure_rr": 33.0, "vibration_level": 2.0,
                "fuel_consumption": 12.5, "vehicle_speed": 60.0, "throttle_position": 30.0,
                "ambient_temp": 38.0, "humidity": 40.0, "road_gradient": 0.0, "altitude": 200.0,
            },
        },
        {
            "name": "轮胎气压异常",
            "data": {
                "engine_rpm": 1800.0, "engine_temp": 88.0, "oil_pressure": 45.0,
                "coolant_temp": 87.0, "battery_voltage": 13.2, "battery_temp": 28.0,
                "battery_soc": 75.0, "motor_current": 120.0, "motor_temp": 45.0,
                "brake_pressure": 5.0, "tire_pressure_fl": 18.0, "tire_pressure_fr": 34.0,
                "tire_pressure_rl": 17.0, "tire_pressure_rr": 33.0, "vibration_level": 3.0,
                "fuel_consumption": 10.5, "vehicle_speed": 75.0, "throttle_position": 40.0,
                "ambient_temp": 22.0, "humidity": 55.0, "road_gradient": 2.0, "altitude": 150.0,
            },
        },
    ]

    logger.info(f"\n  运行 {len(demo_scenarios)} 个演示场景:")

    for i, scenario in enumerate(demo_scenarios, 1):
        try:
            # 清空缓冲区确保每个场景独立
            predictor.reset_buffer()

            # 预测（重复30次填满滑动窗口）
            for _ in range(30):
                result = predictor.predict(scenario["data"])

            # 物理验证
            phys_result = predictor.physics_check(scenario["data"])

            # 生成告警
            alert = alert_engine.process_prediction(result)

            logger.info(f"\n  [{i}] 场景：{scenario['name']}")
            logger.info(f"      故障类型: {result['fault_name']} (类型{result['fault_type']})")
            logger.info(f"      故障概率: {result['probability']:.1%}")
            logger.info(f"      严重程度: {result['severity']:.1%}")
            logger.info(f"      告警级别: {result['alert_level_name']}")
            phys_msg = "✓ 正常" if phys_result["valid"] else f"✗ {len(phys_result['issues'])}个问题"
            logger.info(f"      物理一致性: {phys_msg}")

            if alert:
                logger.info(f"      告警消息: {alert.message}")
                logger.info(f"      维修建议: {result['repair_advice'][:60]}...")

        except Exception as e:
            logger.warning(f"  场景{i}执行失败（非关键）: {e}")

    # 打印告警统计
    stats = alert_engine.get_statistics()
    logger.info(f"\n  告警统计: 共 {stats['total_alerts']} 条告警")

    return alert_engine


def step8_data_quality_check(X: np.ndarray):
    """
    步骤8：数据质量检查演示
    """
    logger.info("=" * 60)
    logger.info("步骤8：数据质量检查")

    from monitoring.data_quality import DataQualityMonitor

    valid_ranges = {
        "engine_rpm": {"min": 0, "max": 6000},
        "engine_temp": {"min": 60, "max": 120},
        "oil_pressure": {"min": 15, "max": 85},
        "coolant_temp": {"min": 60, "max": 110},
        "battery_voltage": {"min": 10.5, "max": 15.0},
        "battery_temp": {"min": -10, "max": 55},
        "battery_soc": {"min": 0, "max": 100},
        "motor_current": {"min": 0, "max": 600},
        "motor_temp": {"min": 15, "max": 90},
        "brake_pressure": {"min": 0, "max": 160},
        "vehicle_speed": {"min": 0, "max": 220},
    }

    monitor = DataQualityMonitor()
    report = monitor.generate_report(X, valid_ranges)

    logger.info(f"  健康评分: {report['health_score']:.4f} ({report['quality_grade']})")
    logger.info(f"  缺失率:   {report['overall_missing_rate']:.4f}")
    logger.info(f"  异常率:   {report['overall_anomaly_rate']:.4f}")
    if report["recommendations"]:
        logger.info(f"  建议: {report['recommendations'][0]}")


def main():
    """
    系统演示主流程
    完整运行：数据生成 → 预处理 → 训练 → 评估 → 告警演示 → 数据质量检查
    """
    print_banner()

    # 设置随机种子保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 自动选择计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用计算设备: {device}")

    try:
        # 步骤1：生成数据（每类300个样本 = 2400个总样本）
        X, y = step1_generate_data(n_per_class=300)

        # 步骤2：预处理 + 滑动窗口序列化
        preprocessor, X_seq, y_seq = step2_preprocess(X, y, seq_len=30)

        # 步骤3：创建DataLoader
        train_loader, val_loader, test_loader = step3_create_dataloaders(X_seq, y_seq)

        # 步骤4：构建模型
        model = step4_build_model(device)

        # 步骤5：训练（演示模式只训练5轮）
        trainer, history = step5_train(model, train_loader, val_loader, device, epochs=5)

        # 步骤6：评估
        metrics = step6_evaluate(model, test_loader, device)

        # 步骤7：告警演示
        alert_engine = step7_alert_demo(model, device)

        # 步骤8：数据质量检查
        step8_data_quality_check(X)

        # 打印最终摘要
        logger.info("=" * 60)
        logger.info("🎉 系统演示完成！")
        logger.info(f"  测试准确率: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (宏平均): {metrics['f1_macro']:.4f}")
        logger.info("  检查点: checkpoints/best_model.pth")
        logger.info("  评估报告: evaluation_results/evaluation_report.json")
        logger.info("=" * 60)

        # 显示告警统计
        stats = alert_engine.get_statistics()
        recent_alerts = alert_engine.get_recent_alerts(n=3)
        logger.info(f"  共生成告警: {stats['total_alerts']} 条")
        logger.info(f"  告警分布: {stats['by_level']}")

    except KeyboardInterrupt:
        logger.info("用户中断运行")
    except Exception as e:
        logger.error(f"运行过程中出现错误: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
