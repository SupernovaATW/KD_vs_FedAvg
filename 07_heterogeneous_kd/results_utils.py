"""
结果处理工具函数
"""

import json
import os
import pandas as pd
import numpy as np


def save_intermediate_results(results, output_dir, timestamp):
    """保存中间结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    intermediate_path = os.path.join(output_dir, f'hetero_intermediate_{timestamp}.json')
    with open(intermediate_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return intermediate_path


def load_results_from_json(json_path):
    """从JSON文件加载结果"""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results


def results_to_dataframe(results):
    """将结果转换为DataFrame"""
    data = []
    for result in results:
        config = result['config']
        data.append({
            'small_temperature': config['small_temperature'],
            'small_alpha': config['small_alpha'],
            'large_temperature': config['large_temperature'],
            'large_alpha': config['large_alpha'],
            'num_cycles': config['num_cycles'],
            'final_avg_acc': result['final_avg_test_acc'],
            'final_small_acc': result['final_small_test_acc'],
            'final_large_avg_acc': np.mean(result['final_large_test_accs']),
            'final_large_node1_acc': result['final_large_test_accs'][0],
            'final_large_node2_acc': result['final_large_test_accs'][1],
            'final_large_node3_acc': result['final_large_test_accs'][2],
            'final_large_node4_acc': result['final_large_test_accs'][3],
        })
    
    return pd.DataFrame(data)


def find_best_params(results, metric='final_avg_test_acc'):
    """找到最佳参数组合"""
    best_result = max(results, key=lambda x: x[metric])
    return best_result


def compare_param_effects(results):
    """比较不同参数的影响"""
    df = results_to_dataframe(results)
    
    # 分析小节点参数影响
    small_temp_effect = df.groupby('small_temperature')['final_avg_acc'].agg(['mean', 'std', 'count'])
    small_alpha_effect = df.groupby('small_alpha')['final_avg_acc'].agg(['mean', 'std', 'count'])
    
    # 分析大节点参数影响
    large_temp_effect = df.groupby('large_temperature')['final_avg_acc'].agg(['mean', 'std', 'count'])
    large_alpha_effect = df.groupby('large_alpha')['final_avg_acc'].agg(['mean', 'std', 'count'])
    
    return {
        'small_temperature': small_temp_effect,
        'small_alpha': small_alpha_effect,
        'large_temperature': large_temp_effect,
        'large_alpha': large_alpha_effect
    }


def print_results_summary(results):
    """打印结果摘要"""
    print("\n" + "="*80)
    print("实验结果摘要")
    print("="*80)
    
    # 最佳结果
    best = find_best_params(results)
    print("\n最佳参数组合:")
    print(f"  小节点: T={best['config']['small_temperature']}, "
          f"α={best['config']['small_alpha']}")
    print(f"  大节点: T={best['config']['large_temperature']}, "
          f"α={best['config']['large_alpha']}")
    print(f"  最终平均准确率: {best['final_avg_test_acc']:.2f}%")
    print(f"  小节点准确率: {best['final_small_test_acc']:.2f}%")
    print(f"  大节点平均准确率: {np.mean(best['final_large_test_accs']):.2f}%")
    
    # 统计信息
    all_accs = [r['final_avg_test_acc'] for r in results]
    print(f"\n整体统计:")
    print(f"  实验总数: {len(results)}")
    print(f"  平均准确率: {np.mean(all_accs):.2f}% ± {np.std(all_accs):.2f}%")
    print(f"  最高准确率: {np.max(all_accs):.2f}%")
    print(f"  最低准确率: {np.min(all_accs):.2f}%")
    
    # 参数影响分析
    param_effects = compare_param_effects(results)
    
    print("\n参数影响分析:")
    print("\n  小节点Temperature:")
    print(param_effects['small_temperature'].to_string())
    
    print("\n  小节点Alpha:")
    print(param_effects['small_alpha'].to_string())
    
    print("\n  大节点Temperature:")
    print(param_effects['large_temperature'].to_string())
    
    print("\n  大节点Alpha:")
    print(param_effects['large_alpha'].to_string())
    
    print("\n" + "="*80)


def get_cycle_statistics(result):
    """获取单个实验的循环统计信息"""
    cycle_results = result['cycle_results']
    
    stats = {
        'initial_avg_acc': cycle_results[0]['avg_test_acc'],
        'final_avg_acc': cycle_results[-1]['avg_test_acc'],
        'improvement': cycle_results[-1]['avg_test_acc'] - cycle_results[0]['avg_test_acc'],
        'best_cycle': max(cycle_results, key=lambda x: x['avg_test_acc'])['cycle'],
        'best_cycle_acc': max(cycle_results, key=lambda x: x['avg_test_acc'])['avg_test_acc']
    }
    
    return stats
