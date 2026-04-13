"""
演示数据生成脚本
生成带有7次渐进式故障注入的3000步时序传感器数据
"""

import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_progressive_fault_sequence(
    generator,
    n_total: int = 3000,
    n_faults: int = 7,
) -> tuple:
    """
    生成带有渐进式故障注入的时序数据
    :param generator: VirtualSampleGenerator实例
    :param n_total: 总时间步数
    :param n_faults: 注入的故障事件数量
    :return: (X, y) 时序数据和标签
    """
    X = []
    y = []

    # 将总时序分段：每段约 n_total/n_faults 步
    segment_len = n_total // (n_faults + 1)

    # 记录故障注入点
    fault_events = []
    for i in range(n_faults):
        fault_start = (i + 1) * segment_len - segment_len // 3  # 每段最后1/3开始故障
        fault_type = (i % 7) + 1  # 循环覆盖7种故障类型
        fault_events.append((fault_start, fault_type))

    print(f"故障注入时间点: {[(t, f) for t, f in fault_events]}")

    # 生成数据
    for t in range(n_total):
        # 判断当前时间步是否处于故障阶段
        current_fault = 0
        for fault_start, fault_type in fault_events:
            if t >= fault_start:
                current_fault = fault_type

        if current_fault == 0:
            sample = generator.generate_normal_sample()
        else:
            # 渐进式故障：故障程度随时间线性增加
            fault_start_time = max(fs for fs, ft in fault_events if ft == current_fault)
            progress = min(1.0, (t - fault_start_time) / (segment_len // 3 + 1))

            # 混合正常样本和故障样本（渐进过渡）
            normal_sample = generator.generate_normal_sample()
            fault_sample = generator.generate_fault_sample(current_fault)
            sample = (1 - progress) * normal_sample + progress * fault_sample

        X.append(sample)
        y.append(current_fault)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main():
    """主函数：生成并保存演示数据"""
    from physics.virtual_sample_generator import VirtualSampleGenerator

    print("=" * 60)
    print("车辆故障预测演示数据生成器")
    print("=" * 60)

    # 初始化生成器（固定随机种子保证可重复性）
    generator = VirtualSampleGenerator(seed=2024)

    print("\n生成 3000 个时间步的时序数据（包含7次渐进式故障注入）...")
    X, y = generate_progressive_fault_sequence(generator, n_total=3000, n_faults=7)

    print(f"\n数据统计:")
    print(f"  X 形状: {X.shape}")
    print(f"  y 形状: {y.shape}")

    fault_names = [
        "正常", "发动机过热", "电池异常", "制动失效",
        "轮胎气压", "电机故障", "润滑故障", "冷却故障"
    ]
    for cls in range(8):
        count = np.sum(y == cls)
        pct = count / len(y) * 100
        print(f"  类别 {cls} ({fault_names[cls]}): {count} 个样本 ({pct:.1f}%)")

    # 保存数据文件
    output_dir = os.getcwd()
    X_path = os.path.join(output_dir, "demo_X.npy")
    y_path = os.path.join(output_dir, "demo_y.npy")

    np.save(X_path, X)
    np.save(y_path, y)

    print(f"\n数据已保存:")
    print(f"  {X_path}")
    print(f"  {y_path}")
    print("\n生成完成！")


if __name__ == "__main__":
    main()
