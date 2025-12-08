"""
快速测试脚本 - 异构模型知识蒸馏
用于快速验证代码是否正常工作
"""

import os
import sys
import torch
import random
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_runner import run_heterogeneous_kd_experiment
from common.data_utils import split_dataset_by_ratio


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class QuickTestArgs:
    """快速测试参数"""
    def __init__(self):
        # 数据参数
        self.num_large_nodes = 4
        self.large_to_small_ratio = 10
        self.batch_size = 128
        self.num_workers = 2
        
        # 训练参数 - 使用很少的epochs进行快速测试
        self.local_epochs = 2
        self.small_node_epochs = 2
        self.large_node_kd_epochs = 2
        self.num_cycles = 1  # 只运行1个循环
        
        # 学习率
        self.lr = 0.1
        
        # KD参数 - 固定值用于快速测试
        self.small_temperature = 4.0
        self.small_alpha = 0.7
        self.large_temperature = 4.0
        self.large_alpha = 0.7
        
        # 其他参数
        self.seed = 42
        self.output_dir = 'hetero_results'


def main():
    """快速测试主函数"""
    print("="*80)
    print("异构模型知识蒸馏 - 快速测试")
    print("="*80)
    print("这是一个快速测试，使用极少的epochs和1个循环")
    print("用于验证代码是否正常工作")
    print("="*80)
    
    # 设置参数
    args = QuickTestArgs()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 准备数据
    print("\n准备数据...")
    large_trainloaders, small_trainloader, testloader = split_dataset_by_ratio(
        num_large_nodes=args.num_large_nodes,
        large_to_small_ratio=args.large_to_small_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root='../data'  # 使用项目根目录的data文件夹
    )
    
    print(f"✓ 大节点数量: {len(large_trainloaders)} (2个ResNet8 + 2个ResNet18)")
    print(f"✓ 小节点数量: 1 (ResNet34)")
    print(f"✓ 批次大小: {args.batch_size}")
    
    # 运行实验
    print("\n开始快速测试...")
    print(f"配置: Small(T={args.small_temperature}, α={args.small_alpha}), "
          f"Large(T={args.large_temperature}, α={args.large_alpha})")
    print(f"循环次数: {args.num_cycles}")
    print(f"每个阶段训练epochs: Local={args.local_epochs}, Small={args.small_node_epochs}, "
          f"Large_KD={args.large_node_kd_epochs}")
    
    try:
        result = run_heterogeneous_kd_experiment(
            large_trainloaders, small_trainloader, testloader,
            args, device
        )
        
        print("\n" + "="*80)
        print("快速测试完成!")
        print("="*80)
        print(f"最终平均准确率: {result['final_avg_test_acc']:.2f}%")
        print(f"小节点(ResNet34)准确率: {result['final_small_test_acc']:.2f}%")
        print("大节点准确率:")
        for i, acc in enumerate(result['final_large_test_accs']):
            model_name = "ResNet8" if i < 2 else "ResNet18"
            print(f"  节点{i+1} ({model_name}): {acc:.2f}%")
        print(f"大节点平均准确率: {np.mean(result['final_large_test_accs']):.2f}%")
        print("="*80)
        print("\n✓ 代码运行正常!")
        print("\n提示: 若要运行完整实验，请使用 main.py 并设置合适的epochs和cycles参数")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
