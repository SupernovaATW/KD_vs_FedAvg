"""
知识蒸馏参数研究实验 - 主程序（循环迭代版本）
场景：5个节点的联邦学习
- 4个大数据节点(性能有限): 每个节点有大量数据
- 1个小数据节点(性能很好): 数据量只有大数据节点的1/10

实验目的：
研究不同的KD参数(temperature T 和 alpha α)在循环迭代中对知识迁移效果的影响

实验设置：
- Temperature (T): [1, 2, 3, 4, 5, 6, 8, 10]
- Alpha (α): [0.1, 0.3, 0.5, 0.7, 0.9]

实验流程（针对每组参数）：
1. 阶段1: 4个大数据节点进行初始FedAvg训练
2. 阶段2: 小数据节点从聚合模型进行KD学习
3. 阶段3: 使用学生模型继续FedAvg训练
4. 重复阶段2和3，形成知识迁移循环
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from common.models import ResNet18
from common.data_utils import split_dataset_for_nodes, load_cifar10_data, create_dataloaders
from config import parse_args, print_config
from experiment_runner import run_single_experiment
from visualize_results import visualize_results


def save_results(all_results, output_dir):
    """
    保存实验结果
    
    Args:
        all_results: 所有实验结果列表
        output_dir: 输出目录
    
    Returns:
        results_df: DataFrame格式的结果
    """
    # 转换为DataFrame (只保存摘要信息)
    results_summary = []
    for r in all_results:
        summary = r['summary']
        results_summary.append({
            'temperature': r['temperature'],
            'alpha': r['alpha'],
            'initial_fedavg_acc': summary['initial_fedavg_acc'],
            'baseline_acc': summary['baseline_acc'],
            'final_kd_acc': summary['final_kd_acc'],
            'final_fedavg_acc': summary['final_fedavg_acc'],
            'total_improvement_over_baseline': summary['total_improvement_over_baseline'],
            'total_improvement_over_initial': summary['total_improvement_over_initial']
        })
    
    results_df = pd.DataFrame(results_summary)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存CSV
    csv_path = os.path.join(output_dir, f'kd_params_study_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n结果已保存到: {csv_path}")
    
    # 保存详细JSON
    json_path = os.path.join(output_dir, f'kd_params_study_detailed_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"详细结果已保存到: {json_path}")
    
    return results_df


def print_summary(results_df):
    """打印结果摘要"""
    print("\n" + "="*80)
    print("实验结果摘要")
    print("="*80)
    
    # 找出最佳参数组合（基于最终FedAvg准确率）
    best_idx = results_df['final_fedavg_acc'].idxmax()
    best_result = results_df.iloc[best_idx]
    
    print(f"\n最佳参数组合（基于最终FedAvg准确率）:")
    print(f"  Temperature: {best_result['temperature']}")
    print(f"  Alpha: {best_result['alpha']}")
    print(f"  初始FedAvg准确率: {best_result['initial_fedavg_acc']:.2f}%")
    print(f"  基线准确率: {best_result['baseline_acc']:.2f}%")
    print(f"  最终KD准确率: {best_result['final_kd_acc']:.2f}%")
    print(f"  最终FedAvg准确率: {best_result['final_fedavg_acc']:.2f}%")
    print(f"  总提升(vs基线): {best_result['total_improvement_over_baseline']:+.2f}%")
    print(f"  总提升(vs初始): {best_result['total_improvement_over_initial']:+.2f}%")
    
    # 统计分析
    print(f"\n统计信息:")
    print(f"  最终FedAvg准确率范围: {results_df['final_fedavg_acc'].min():.2f}% - {results_df['final_fedavg_acc'].max():.2f}%")
    print(f"  平均最终FedAvg准确率: {results_df['final_fedavg_acc'].mean():.2f}%")
    print(f"  相比基线平均提升: {results_df['total_improvement_over_baseline'].mean():+.2f}%")
    print(f"  相比初始平均提升: {results_df['total_improvement_over_initial'].mean():+.2f}%")
    
    print("\n" + "="*80)
    print("参数研究实验完成!")
    print("="*80)


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
    data_root = os.path.join(project_root, 'data')
    trainset, testset, testloader = load_cifar10_data(
        data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # 分配数据给各个节点
    large_nodes_indices, small_node_indices = split_dataset_for_nodes(
        trainset, num_large_nodes=args.num_large_nodes, 
        large_to_small_ratio=args.large_to_small_ratio, seed=args.seed
    )
    
    # 创建数据加载器
    large_trainloaders, small_trainloader = create_dataloaders(
        trainset, large_nodes_indices, small_node_indices,
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # 定义参数网格
    temperatures = args.temperatures
    alphas = args.alphas
    
    print(f"\n{'='*80}")
    print("参数研究实验设置")
    print(f"{'='*80}")
    print(f"Temperature (T) 范围: {temperatures}")
    print(f"Alpha (α) 范围: {alphas}")
    print(f"总实验次数: {len(temperatures) * len(alphas)}")
    print(f"每个实验循环次数: {args.num_cycles}")
    print(f"{'='*80}")
    
    # 运行所有实验
    all_results = []
    param_combinations = list(product(temperatures, alphas))
    
    print(f"\n{'='*80}")
    print("开始参数扫描实验...")
    print(f"{'='*80}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    for idx, (temp, alpha) in enumerate(param_combinations, 1):
        print(f"\n[{idx}/{len(param_combinations)}] 正在测试 T={temp}, α={alpha}")
        
        result = run_single_experiment(
            temperature=temp,
            alpha=alpha,
            large_trainloaders=large_trainloaders,
            small_trainloader=small_trainloader,
            testloader=testloader,
            args=args,
            device=device
        )
        all_results.append(result)
        
        # 保存中间结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_df = pd.DataFrame([r['summary'] for r in all_results])
        temp_df.insert(0, 'temperature', [r['temperature'] for r in all_results])
        temp_df.insert(1, 'alpha', [r['alpha'] for r in all_results])
        temp_df.to_csv(os.path.join(args.output_dir, f'intermediate_results_{timestamp}.csv'), 
                      index=False)
    
    # 保存最终结果
    results_df = save_results(all_results, args.output_dir)
    
    # 生成可视化
    if not args.no_visualize:
        print("\n生成可视化图表...")
        visualize_results(results_df, args.output_dir)
    
    # 打印结果摘要
    print_summary(results_df)


if __name__ == '__main__':
    main()
