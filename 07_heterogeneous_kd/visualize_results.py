"""
结果可视化脚本 - 异构模型知识蒸馏实验
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_results(json_path):
    """加载JSON结果文件"""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results


def plot_param_heatmap(results, output_dir):
    """绘制参数热力图"""
    # 准备数据
    data = []
    for result in results:
        config = result['config']
        data.append({
            'small_temp': config['small_temperature'],
            'small_alpha': config['small_alpha'],
            'large_temp': config['large_temperature'],
            'large_alpha': config['large_alpha'],
            'final_acc': result['final_avg_test_acc']
        })
    
    df = pd.DataFrame(data)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('异构模型知识蒸馏参数研究 - 热力图', fontsize=16, y=0.995)
    
    # 1. Small节点参数热力图
    pivot1 = df.pivot_table(
        values='final_acc',
        index='small_alpha',
        columns='small_temp',
        aggfunc='mean'
    )
    sns.heatmap(pivot1, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0, 0], cbar_kws={'label': '准确率 (%)'})
    axes[0, 0].set_title('小节点参数影响 (平均所有大节点参数)')
    axes[0, 0].set_xlabel('Temperature')
    axes[0, 0].set_ylabel('Alpha')
    
    # 2. Large节点参数热力图
    pivot2 = df.pivot_table(
        values='final_acc',
        index='large_alpha',
        columns='large_temp',
        aggfunc='mean'
    )
    sns.heatmap(pivot2, annot=True, fmt='.2f', cmap='YlGnBu', ax=axes[0, 1], cbar_kws={'label': '准确率 (%)'})
    axes[0, 1].set_title('大节点参数影响 (平均所有小节点参数)')
    axes[0, 1].set_xlabel('Temperature')
    axes[0, 1].set_ylabel('Alpha')
    
    # 3. Temperature对比
    temp_data = df.groupby(['small_temp', 'large_temp'])['final_acc'].mean().reset_index()
    pivot3 = temp_data.pivot(index='large_temp', columns='small_temp', values='final_acc')
    sns.heatmap(pivot3, annot=True, fmt='.2f', cmap='viridis', ax=axes[1, 0], cbar_kws={'label': '准确率 (%)'})
    axes[1, 0].set_title('Temperature交互影响')
    axes[1, 0].set_xlabel('小节点 Temperature')
    axes[1, 0].set_ylabel('大节点 Temperature')
    
    # 4. Alpha对比
    alpha_data = df.groupby(['small_alpha', 'large_alpha'])['final_acc'].mean().reset_index()
    pivot4 = alpha_data.pivot(index='large_alpha', columns='small_alpha', values='final_acc')
    sns.heatmap(pivot4, annot=True, fmt='.2f', cmap='plasma', ax=axes[1, 1], cbar_kws={'label': '准确率 (%)'})
    axes[1, 1].set_title('Alpha交互影响')
    axes[1, 1].set_xlabel('小节点 Alpha')
    axes[1, 1].set_ylabel('大节点 Alpha')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'param_heatmaps.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 热力图已保存: {save_path}")
    plt.close()


def plot_cycle_progress(results, output_dir):
    """绘制训练循环进度"""
    # 选择最佳结果
    best_result = max(results, key=lambda x: x['final_avg_test_acc'])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'最佳参数组合的训练进度\n'
        f'Small(T={best_result["config"]["small_temperature"]}, '
        f'α={best_result["config"]["small_alpha"]}), '
        f'Large(T={best_result["config"]["large_temperature"]}, '
        f'α={best_result["config"]["large_alpha"]})',
        fontsize=14
    )
    
    cycles = [r['cycle'] for r in best_result['cycle_results']]
    small_accs = [r['small_test_acc'] for r in best_result['cycle_results']]
    avg_accs = [r['avg_test_acc'] for r in best_result['cycle_results']]
    
    # 1. 整体准确率趋势
    axes[0, 0].plot(cycles, avg_accs, 'o-', linewidth=2, markersize=8, label='平均准确率')
    axes[0, 0].plot(cycles, small_accs, 's-', linewidth=2, markersize=8, label='小节点准确率')
    axes[0, 0].set_xlabel('循环次数')
    axes[0, 0].set_ylabel('准确率 (%)')
    axes[0, 0].set_title('整体准确率趋势')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 各大节点准确率
    for i in range(4):
        model_name = "ResNet8" if i < 2 else "ResNet18"
        large_accs = [r['large_test_accs'][i] for r in best_result['cycle_results']]
        axes[0, 1].plot(cycles, large_accs, 'o-', linewidth=2, markersize=6, 
                       label=f'大节点{i+1} ({model_name})')
    axes[0, 1].set_xlabel('循环次数')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].set_title('大节点准确率趋势')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 准确率提升分析
    if len(cycles) > 1:
        small_improvements = np.diff(small_accs)
        avg_improvements = np.diff(avg_accs)
        
        x = range(1, len(cycles))
        width = 0.35
        axes[1, 0].bar([i - width/2 for i in x], small_improvements, width, 
                      label='小节点提升', alpha=0.8)
        axes[1, 0].bar([i + width/2 for i in x], avg_improvements, width, 
                      label='平均提升', alpha=0.8)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1, 0].set_xlabel('循环间隔')
        axes[1, 0].set_ylabel('准确率提升 (%)')
        axes[1, 0].set_title('循环间准确率提升')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([f'{i}->{i+1}' for i in range(1, len(cycles))])
    
    # 4. 最终模型对比
    final_large = best_result['final_large_test_accs']
    final_small = best_result['final_small_test_acc']
    
    models = ['大1\n(R8)', '大2\n(R8)', '大3\n(R18)', '大4\n(R18)', '小\n(R34)']
    accs = final_large + [final_small]
    colors = ['#ff7f0e', '#ff7f0e', '#2ca02c', '#2ca02c', '#d62728']
    
    bars = axes[1, 1].bar(models, accs, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('准确率 (%)')
    axes[1, 1].set_title('最终模型准确率对比')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_progress.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练进度图已保存: {save_path}")
    plt.close()


def plot_param_comparison(results, output_dir):
    """绘制参数对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('参数影响对比分析', fontsize=14)
    
    # 准备数据
    data = []
    for result in results:
        config = result['config']
        data.append({
            'small_temp': config['small_temperature'],
            'small_alpha': config['small_alpha'],
            'large_temp': config['large_temperature'],
            'large_alpha': config['large_alpha'],
            'final_acc': result['final_avg_test_acc'],
            'small_acc': result['final_small_test_acc'],
            'large_avg_acc': np.mean(result['final_large_test_accs'])
        })
    
    df = pd.DataFrame(data)
    
    # 1. Temperature影响（固定最佳alpha）
    best_small_alpha = df.groupby('small_alpha')['final_acc'].mean().idxmax()
    best_large_alpha = df.groupby('large_alpha')['final_acc'].mean().idxmax()
    
    df_temp = df[(df['small_alpha'] == best_small_alpha) & (df['large_alpha'] == best_large_alpha)]
    temp_grouped = df_temp.groupby(['small_temp', 'large_temp'])['final_acc'].mean().reset_index()
    
    for large_t in sorted(df_temp['large_temp'].unique()):
        subset = temp_grouped[temp_grouped['large_temp'] == large_t]
        axes[0].plot(subset['small_temp'], subset['final_acc'], 'o-', 
                    linewidth=2, markersize=8, label=f'大节点T={large_t}')
    
    axes[0].set_xlabel('小节点 Temperature')
    axes[0].set_ylabel('平均准确率 (%)')
    axes[0].set_title(f'Temperature影响 (α固定为{best_small_alpha}/{best_large_alpha})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Alpha影响（固定最佳temperature）
    best_small_temp = df.groupby('small_temp')['final_acc'].mean().idxmax()
    best_large_temp = df.groupby('large_temp')['final_acc'].mean().idxmax()
    
    df_alpha = df[(df['small_temp'] == best_small_temp) & (df['large_temp'] == best_large_temp)]
    alpha_grouped = df_alpha.groupby(['small_alpha', 'large_alpha'])['final_acc'].mean().reset_index()
    
    for large_a in sorted(df_alpha['large_alpha'].unique()):
        subset = alpha_grouped[alpha_grouped['large_alpha'] == large_a]
        axes[1].plot(subset['small_alpha'], subset['final_acc'], 's-', 
                    linewidth=2, markersize=8, label=f'大节点α={large_a}')
    
    axes[1].set_xlabel('小节点 Alpha')
    axes[1].set_ylabel('平均准确率 (%)')
    axes[1].set_title(f'Alpha影响 (T固定为{best_small_temp}/{best_large_temp})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'param_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 参数对比图已保存: {save_path}")
    plt.close()


def generate_summary_report(results, output_dir):
    """生成文本摘要报告"""
    report_path = os.path.join(output_dir, 'summary_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("异构模型知识蒸馏实验 - 摘要报告\n")
        f.write("="*80 + "\n\n")
        
        # 最佳结果
        best_result = max(results, key=lambda x: x['final_avg_test_acc'])
        f.write("最佳参数组合:\n")
        f.write(f"  小节点: Temperature={best_result['config']['small_temperature']}, "
                f"Alpha={best_result['config']['small_alpha']}\n")
        f.write(f"  大节点: Temperature={best_result['config']['large_temperature']}, "
                f"Alpha={best_result['config']['large_alpha']}\n")
        f.write(f"  最终平均准确率: {best_result['final_avg_test_acc']:.2f}%\n")
        f.write(f"  小节点准确率: {best_result['final_small_test_acc']:.2f}%\n")
        f.write(f"  大节点平均准确率: {np.mean(best_result['final_large_test_accs']):.2f}%\n")
        f.write("\n")
        
        # 统计信息
        all_accs = [r['final_avg_test_acc'] for r in results]
        f.write("整体统计:\n")
        f.write(f"  实验总数: {len(results)}\n")
        f.write(f"  平均准确率: {np.mean(all_accs):.2f}% ± {np.std(all_accs):.2f}%\n")
        f.write(f"  最高准确率: {np.max(all_accs):.2f}%\n")
        f.write(f"  最低准确率: {np.min(all_accs):.2f}%\n")
        f.write("\n")
        
        # Top 5 结果
        top5 = sorted(results, key=lambda x: x['final_avg_test_acc'], reverse=True)[:5]
        f.write("Top 5 参数组合:\n")
        for i, result in enumerate(top5, 1):
            f.write(f"\n  {i}. 准确率: {result['final_avg_test_acc']:.2f}%\n")
            f.write(f"     Small(T={result['config']['small_temperature']}, "
                   f"α={result['config']['small_alpha']}), "
                   f"Large(T={result['config']['large_temperature']}, "
                   f"α={result['config']['large_alpha']})\n")
            f.write(f"     小节点: {result['final_small_test_acc']:.2f}%, "
                   f"大节点平均: {np.mean(result['final_large_test_accs']):.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ 摘要报告已保存: {report_path}")


def visualize_results(json_path):
    """可视化实验结果"""
    print("\n开始生成可视化结果...")
    
    # 加载结果
    results = load_results(json_path)
    
    if not results:
        print("❌ 没有找到实验结果")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(json_path)
    
    # 生成各种图表
    plot_param_heatmap(results, output_dir)
    plot_cycle_progress(results, output_dir)
    plot_param_comparison(results, output_dir)
    generate_summary_report(results, output_dir)
    
    print("\n✓ 所有可视化结果已生成!")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化异构模型KD实验结果')
    parser.add_argument('json_path', type=str, help='实验结果JSON文件路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_path):
        print(f"❌ 文件不存在: {args.json_path}")
        return 1
    
    visualize_results(args.json_path)
    return 0


if __name__ == '__main__':
    exit(main())
