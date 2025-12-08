"""
主程序 - 异构模型知识蒸馏参数搜索实验
"""

import os
import sys
import json
import csv
import torch
import random
import numpy as np
from datetime import datetime
from itertools import product

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import parse_args, print_config
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data(args):
    """准备数据加载器"""
    print("\n准备数据...")
    
    # 使用common中的数据分割工具，指定正确的数据路径
    large_trainloaders, small_trainloader, testloader = split_dataset_by_ratio(
        num_large_nodes=args.num_large_nodes,
        large_to_small_ratio=args.large_to_small_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root='../data'  # 使用项目根目录的data文件夹
    )
    
    print(f"✓ 大节点数量: {len(large_trainloaders)}")
    print(f"✓ 小节点数量: 1")
    print(f"✓ 批次大小: {args.batch_size}")
    
    return large_trainloaders, small_trainloader, testloader


def save_results(results, args, timestamp):
    """保存实验结果"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存完整JSON结果
    json_path = os.path.join(args.output_dir, f'hetero_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ 完整结果已保存: {json_path}")
    
    # 保存汇总CSV
    csv_path = os.path.join(args.output_dir, f'hetero_summary_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'small_temperature', 'small_alpha', 'large_temperature', 'large_alpha',
            'final_avg_acc', 'final_small_acc', 'final_large_avg_acc'
        ])
        
        for result in results:
            large_avg = np.mean(result['final_large_test_accs'])
            writer.writerow([
                result['config']['small_temperature'],
                result['config']['small_alpha'],
                result['config']['large_temperature'],
                result['config']['large_alpha'],
                f"{result['final_avg_test_acc']:.2f}",
                f"{result['final_small_test_acc']:.2f}",
                f"{large_avg:.2f}"
            ])
    
    print(f"✓ 汇总结果已保存: {csv_path}")
    
    return json_path, csv_path


def run_param_search(args):
    """运行参数搜索实验"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 准备数据
    large_trainloaders, small_trainloader, testloader = prepare_data(args)
    
    # 生成参数组合
    param_combinations = list(product(
        args.small_temperatures,
        args.small_alphas,
        args.large_temperatures,
        args.large_alphas
    ))
    
    total_experiments = len(param_combinations)
    print(f"\n总共需要运行 {total_experiments} 个参数组合")
    print(f"每个组合运行 {args.num_cycles} 个循环")
    print("-"*80)
    
    all_results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for idx, (small_temp, small_alpha, large_temp, large_alpha) in enumerate(param_combinations, 1):
        print("\n" + "="*80)
        print(f"实验 {idx}/{total_experiments}")
        print(f"参数: Small(T={small_temp}, α={small_alpha}), Large(T={large_temp}, α={large_alpha})")
        print("="*80)
        
        # 设置当前参数
        args.small_temperature = small_temp
        args.small_alpha = small_alpha
        args.large_temperature = large_temp
        args.large_alpha = large_alpha
        
        # 重新设置随机种子以确保可重复性
        set_seed(args.seed + idx)
        
        try:
            # 运行实验
            result = run_heterogeneous_kd_experiment(
                large_trainloaders, small_trainloader, testloader,
                args, device
            )
            
            all_results.append(result)
            
            print(f"\n实验 {idx} 完成!")
            print(f"最终平均准确率: {result['final_avg_test_acc']:.2f}%")
            print(f"小节点准确率: {result['final_small_test_acc']:.2f}%")
            print(f"大节点平均准确率: {np.mean(result['final_large_test_accs']):.2f}%")
            
            # 保存中间结果
            if idx % 3 == 0 or idx == total_experiments:
                intermediate_path = os.path.join(
                    args.output_dir, 
                    f'hetero_intermediate_{timestamp}.json'
                )
                with open(intermediate_path, 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"✓ 中间结果已保存: {intermediate_path}")
        
        except Exception as e:
            print(f"\n❌ 实验 {idx} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存最终结果
    print("\n" + "="*80)
    print("所有实验完成!")
    print("="*80)
    
    json_path, csv_path = save_results(all_results, args, timestamp)
    
    # 输出最佳结果
    if all_results:
        best_result = max(all_results, key=lambda x: x['final_avg_test_acc'])
        print("\n最佳参数组合:")
        print(f"  Small节点: T={best_result['config']['small_temperature']}, "
              f"α={best_result['config']['small_alpha']}")
        print(f"  Large节点: T={best_result['config']['large_temperature']}, "
              f"α={best_result['config']['large_alpha']}")
        print(f"  最终平均准确率: {best_result['final_avg_test_acc']:.2f}%")
        print(f"  小节点准确率: {best_result['final_small_test_acc']:.2f}%")
        print(f"  大节点平均准确率: {np.mean(best_result['final_large_test_accs']):.2f}%")
    
    return all_results


def main():
    """主函数"""
    args = parse_args()
    print_config(args)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 运行参数搜索
    results = run_param_search(args)
    
    print("\n实验全部完成!")


if __name__ == '__main__':
    main()
