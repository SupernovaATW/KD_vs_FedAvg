"""
快速测试脚本 - 验证KD参数研究实验代码
仅测试少量参数组合，用于快速验证代码正确性
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from common.models import ResNet18
from kd_params_study import (
    split_dataset_for_nodes,
    train_fedavg_large_nodes,
    run_single_experiment
)


def quick_test():
    """快速测试"""
    print("="*80)
    print("快速测试 - KD参数研究实验")
    print("="*80)
    
    # 设置
    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 加载数据（使用更小的batch size加快测试）
    print("\n加载数据...")
    
    # 数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 数据目录
    data_root = os.path.join(project_root, 'data')
    
    # 下载和加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test)
    
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
    
    # 分配数据
    large_nodes_indices, small_node_indices = split_dataset_for_nodes(
        trainset, num_large_nodes=4, large_to_small_ratio=10, seed=42
    )
    
    # 创建数据加载器
    large_trainloaders = [
        DataLoader(Subset(trainset, indices), batch_size=256, shuffle=True, num_workers=2)
        for indices in large_nodes_indices
    ]
    small_trainloader = DataLoader(
        Subset(trainset, small_node_indices), batch_size=256, shuffle=True, num_workers=2
    )
    
    # 测试参数 - 只测试几个组合
    test_temperatures = [3, 5]
    test_alphas = [0.5, 0.7]
    
    print(f"\n测试参数组合:")
    print(f"  Temperature: {test_temperatures}")
    print(f"  Alpha: {test_alphas}")
    print(f"  总测试次数: {len(test_temperatures) * len(test_alphas)}")
    
    # 训练教师模型（使用更少的轮次）
    print("\n" + "="*80)
    print("步骤1: 训练教师模型 (FedAvg)")
    print("="*80)
    teacher_model, _, teacher_acc = train_fedavg_large_nodes(
        ResNet18, large_trainloaders, testloader,
        num_rounds=3, local_epochs=2, lr=0.1, device=device, verbose=True
    )
    print(f"教师模型准确率: {teacher_acc:.2f}%")
    
    # 测试参数组合
    print("\n" + "="*80)
    print("步骤2: 测试KD参数组合")
    print("="*80)
    
    results = []
    for temp in test_temperatures:
        for alpha in test_alphas:
            print(f"\n{'─'*80}")
            print(f"测试 T={temp}, α={alpha}")
            print(f"{'─'*80}")
            
            result = run_single_experiment(
                temperature=temp,
                alpha=alpha,
                large_trainloaders=large_trainloaders,
                small_trainloader=small_trainloader,
                testloader=testloader,
                device=device,
                teacher_model=teacher_model
            )
            results.append(result)
            
            print(f"\n结果:")
            print(f"  基线准确率: {result['baseline_acc']:.2f}%")
            print(f"  教师准确率: {result['teacher_acc']:.2f}%")
            print(f"  学生准确率: {result['student_acc']:.2f}%")
            print(f"  相比基线提升: {result['improvement_over_baseline']:+.2f}%")
            print(f"  相比教师提升: {result['improvement_over_teacher']:+.2f}%")
    
    # 汇总结果
    print("\n" + "="*80)
    print("测试完成 - 结果汇总")
    print("="*80)
    
    print("\n所有测试结果:")
    print(f"{'Temperature':<12} {'Alpha':<8} {'Baseline':<10} {'Teacher':<10} {'Student':<10} {'Improve':<10}")
    print("─"*80)
    for r in results:
        print(f"{r['temperature']:<12} {r['alpha']:<8.1f} {r['baseline_acc']:<10.2f} "
              f"{r['teacher_acc']:<10.2f} {r['student_acc']:<10.2f} {r['improvement_over_baseline']:<+10.2f}")
    
    # 找出最佳组合
    best_result = max(results, key=lambda x: x['student_acc'])
    print(f"\n最佳参数组合:")
    print(f"  Temperature: {best_result['temperature']}")
    print(f"  Alpha: {best_result['alpha']}")
    print(f"  学生准确率: {best_result['student_acc']:.2f}%")
    
    print("\n✓ 快速测试完成! 代码运行正常。")
    print("\n提示: 运行完整实验请执行: python kd_params_study.py")


if __name__ == '__main__':
    quick_test()
