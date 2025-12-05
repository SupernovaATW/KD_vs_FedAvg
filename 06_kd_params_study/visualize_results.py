"""
可视化KD参数研究结果（循环迭代版本）
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-darkgrid')


def visualize_results(results_df, output_dir='param_study_results'):
    """
    创建可视化图表
    
    Args:
        results_df: 实验结果DataFrame
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n生成可视化图表到: {output_dir}")
    
    # 1. 热力图：最终FedAvg准确率
    print("\n生成热力图: 最终FedAvg准确率")
    pivot_final = results_df.pivot(index='alpha', columns='temperature', values='final_fedavg_acc')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot_final.values, cmap='YlOrRd', aspect='auto')
    
    # 设置刻度
    ax.set_xticks(np.arange(len(pivot_final.columns)))
    ax.set_yticks(np.arange(len(pivot_final.index)))
    ax.set_xticklabels(pivot_final.columns)
    ax.set_yticklabels(pivot_final.index)
    
    # 添加数值标注
    for i in range(len(pivot_final.index)):
        for j in range(len(pivot_final.columns)):
            text = ax.text(j, i, f'{pivot_final.values[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('最终FedAvg准确率 (%)', fontsize=16, pad=20)
    ax.set_xlabel('Temperature (T)', fontsize=14)
    ax.set_ylabel('Alpha (α)', fontsize=14)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('准确率 (%)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_final_fedavg_acc.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 已保存: heatmap_final_fedavg_acc.png")
    
    # 2. 热力图：总提升（vs基线）
    print("\n生成热力图: 总提升（vs基线）")
    pivot_improvement = results_df.pivot(index='alpha', columns='temperature', 
                                         values='total_improvement_over_baseline')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot_improvement.values, cmap='RdYlGn', aspect='auto',
                  vmin=pivot_improvement.values.min(), vmax=pivot_improvement.values.max())
    
    ax.set_xticks(np.arange(len(pivot_improvement.columns)))
    ax.set_yticks(np.arange(len(pivot_improvement.index)))
    ax.set_xticklabels(pivot_improvement.columns)
    ax.set_yticklabels(pivot_improvement.index)
    
    for i in range(len(pivot_improvement.index)):
        for j in range(len(pivot_improvement.columns)):
            text = ax.text(j, i, f'{pivot_improvement.values[i, j]:+.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_title('总提升 vs 基线 (%)', fontsize=16, pad=20)
    ax.set_xlabel('Temperature (T)', fontsize=14)
    ax.set_ylabel('Alpha (α)', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('提升 (%)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_improvement_over_baseline.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 已保存: heatmap_improvement_over_baseline.png")
    
    # 3. 折线图：不同Alpha下Temperature的影响
    print("\n生成折线图: Temperature影响")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for alpha in sorted(results_df['alpha'].unique()):
        subset = results_df[results_df['alpha'] == alpha].sort_values('temperature')
        ax.plot(subset['temperature'], subset['final_fedavg_acc'],
                marker='o', linewidth=2, markersize=8, label=f'α={alpha}')
    
    ax.set_xlabel('Temperature (T)', fontsize=14)
    ax.set_ylabel('最终FedAvg准确率 (%)', fontsize=14)
    ax.set_title('不同Alpha下Temperature对最终性能的影响', fontsize=16, pad=20)
    ax.legend(title='Alpha', fontsize=10, title_fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_effect.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 已保存: temperature_effect.png")
    
    # 4. 折线图：不同Temperature下Alpha的影响
    print("\n生成折线图: Alpha影响")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for temp in sorted(results_df['temperature'].unique()):
        subset = results_df[results_df['temperature'] == temp].sort_values('alpha')
        ax.plot(subset['alpha'], subset['final_fedavg_acc'],
                marker='s', linewidth=2, markersize=8, label=f'T={temp}')
    
    ax.set_xlabel('Alpha (α)', fontsize=14)
    ax.set_ylabel('最终FedAvg准确率 (%)', fontsize=14)
    ax.set_title('不同Temperature下Alpha对最终性能的影响', fontsize=16, pad=20)
    ax.legend(title='Temperature', fontsize=10, title_fontsize=12, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alpha_effect.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 已保存: alpha_effect.png")
    
    # 5. 对比图：初始vs最终
    print("\n生成对比图: 初始vs最终")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    avg_initial = results_df['initial_fedavg_acc'].mean()
    avg_baseline = results_df['baseline_acc'].mean()
    avg_final_kd = results_df['final_kd_acc'].mean()
    avg_final_fedavg = results_df['final_fedavg_acc'].mean()
    max_final_fedavg = results_df['final_fedavg_acc'].max()
    
    categories = ['初始FedAvg', '基线\n(独立训练)', '最终KD\n(平均)', '最终FedAvg\n(平均)', '最终FedAvg\n(最佳)']
    values = [avg_initial, avg_baseline, avg_final_kd, avg_final_fedavg, max_final_fedavg]
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('准确率 (%)', fontsize=14)
    ax.set_title('循环知识迁移性能对比', fontsize=16, pad=20)
    ax.set_ylim([min(values) - 5, max(values) + 3])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 已保存: performance_comparison.png")
    
    print("\n✓ 所有可视化图表已生成完成!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化KD参数研究结果')
    parser.add_argument('--csv', type=str, required=True, help='结果CSV文件路径')
    parser.add_argument('--output-dir', type=str, default='visualizations', help='输出目录')
    
    args = parser.parse_args()
    
    # 加载结果
    results_df = pd.read_csv(args.csv)
    print(f"加载实验结果: {args.csv}")
    print(f"数据形状: {results_df.shape}")
    
    # 生成可视化
    visualize_results(results_df, args.output_dir)
