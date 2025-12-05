"""
知识迁移循环实验 - 主程序
场景：5个节点的联邦学习
- 4个大数据节点(性能有限): 每个节点有大量数据
- 1个小数据节点(性能很好): 数据量只有大数据节点的1/10

实验流程：
1. 阶段1: 4个大数据节点进行初始FedAvg训练，得到聚合模型
2. 阶段2: 小数据节点使用知识蒸馏从聚合模型学习
3. 阶段3: 使用学生模型作为初始模型，继续FedAvg训练
4. 重复阶段2和3，形成知识迁移循环
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from common.data_loader import get_cifar10_dataloaders
from common.data_utils import split_dataset_for_nodes, create_dataloaders
from config import parse_args, print_config
from experiment_runner import run_knowledge_transfer_experiment
from results_utils import plot_results, save_results


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 打印实验配置
    print_config(args)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 加载数据
    print("\n加载CIFAR-10数据集...")
    trainset, testset, testloader = get_cifar10_dataloaders(
        data_dir=os.path.join(project_root, 'data'),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 分配数据给各个节点
    large_nodes_indices, small_node_indices = split_dataset_for_nodes(
        trainset, 
        num_large_nodes=args.num_large_nodes, 
        large_to_small_ratio=args.large_to_small_ratio, 
        seed=args.seed
    )
    
    # 创建数据加载器
    large_trainloaders, small_trainloader = create_dataloaders(
        trainset, large_nodes_indices, small_node_indices,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # 运行实验
    print("\n" + "="*80)
    print("开始知识迁移循环实验")
    print("="*80)
    
    results = run_knowledge_transfer_experiment(
        large_trainloaders, small_trainloader, testloader,
        args, device=device
    )
    
    # 保存结果
    print("\n" + "="*80)
    print("保存实验结果")
    print("="*80)
    
    timestamp = save_results(results, args.output_dir)
    
    # 生成可视化
    if not args.no_visualize:
        print("\n生成可视化图表...")
        plot_results(results, args.output_dir, timestamp)
    
    # 打印最终摘要
    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)
    
    # 提取关键指标
    initial_fedavg_acc = results['stages'][0]['best_acc']
    baseline_acc = results['baseline']['best_acc']
    
    print(f"\n关键结果:")
    print(f"  初始FedAvg准确率: {initial_fedavg_acc:.2f}%")
    print(f"  小数据节点基线: {baseline_acc:.2f}%")
    
    for i, stage in enumerate(results['stages'][1:], 1):
        kd_acc = stage['kd_stage']['best_acc']
        fedavg_acc = stage['fedavg_stage']['best_acc']
        print(f"\n  循环 {i}:")
        print(f"    KD学生模型: {kd_acc:.2f}% (比基线 {kd_acc - baseline_acc:+.2f}%)")
        print(f"    新FedAvg模型: {fedavg_acc:.2f}% (比学生 {fedavg_acc - kd_acc:+.2f}%)")
    
    print(f"\n结果已保存到: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
