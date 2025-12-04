"""
可视化KD参数研究结果
可以独立运行，用于分析已有的实验结果
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-darkgrid')
from pathlib import Path
import argparse


def load_results(csv_path):
    """加载实验结果CSV文件"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到结果文件: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"加载实验结果: {csv_path}")
    print(f"数据形状: {df.shape}")
    print(f"\n列名: {list(df.columns)}")
    print(f"\n数据预览:")
    print(df.head())
    
    return df


def create_visualizations(results_df, output_dir='visualizations'):
    """
    创建所有可视化图表
    
    Args:
        results_df: 实验结果DataFrame
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n生成可视化图表到: {output_dir}")
    
    # 1. 热力图：学生模型准确率
    print("\n生成热力图: 学生模型准确率")
    pivot_student = results_df.pivot(index='alpha', columns='temperature', values='student_acc')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot_student.values, cmap='YlOrRd', aspect='auto')
    
    # 设置坐标轴
    ax.set_xticks(range(len(pivot_student.columns)))
    ax.set_xticklabels(pivot_student.columns)
    ax.set_yticks(range(len(pivot_student.index)))
    ax.set_yticklabels(pivot_student.index)
    
    ax.set_xlabel('Temperature (T)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Alpha (α)', fontsize=13, fontweight='bold')
    ax.set_title('Student Model Accuracy with Different KD Parameters\n(Higher is Better)', 
                fontsize=15, fontweight='bold', pad=20)
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, label='Student Accuracy (%)')
    cbar.ax.tick_params(labelsize=11)
    
    # 在每个格子中标注数值
    for i in range(len(pivot_student.index)):
        for j in range(len(pivot_student.columns)):
            value = pivot_student.values[i, j]
            text_color = 'white' if value < pivot_student.values.mean() else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                   color=text_color, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_student_acc.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 已保存: heatmap_student_acc.png")
    
    # 2. 热力图：相比基线的提升
    print("\n生成热力图: 相比基线的提升")
    pivot_baseline = results_df.pivot(index='alpha', columns='temperature', 
                                      values='improvement_over_baseline')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 使用diverging colormap
    vmax = max(abs(pivot_baseline.values.min()), abs(pivot_baseline.values.max()))
    im = ax.imshow(pivot_baseline.values, cmap='RdYlGn', aspect='auto',
                  vmin=-vmax, vmax=vmax)
    
    ax.set_xticks(range(len(pivot_baseline.columns)))
    ax.set_xticklabels(pivot_baseline.columns)
    ax.set_yticks(range(len(pivot_baseline.index)))
    ax.set_yticklabels(pivot_baseline.index)
    
    ax.set_xlabel('Temperature (T)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Alpha (α)', fontsize=13, fontweight='bold')
    ax.set_title('Improvement over Baseline (Small Node Alone)\n(Positive is Better)',
                fontsize=15, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax, label='Improvement (%)')
    cbar.ax.tick_params(labelsize=11)
    
    # 标注数值
    for i in range(len(pivot_baseline.index)):
        for j in range(len(pivot_baseline.columns)):
            value = pivot_baseline.values[i, j]
            text_color = 'white' if abs(value) > abs(pivot_baseline.values).mean() else 'black'
            ax.text(j, i, f'{value:+.2f}', ha='center', va='center',
                   color=text_color, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_improvement_baseline.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 已保存: heatmap_improvement_baseline.png")
    
    # 3. 热力图：相比教师的提升
    print("\n生成热力图: 相比教师模型的提升")
    pivot_teacher = results_df.pivot(index='alpha', columns='temperature',
                                     values='improvement_over_teacher')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    vmax = max(abs(pivot_teacher.values.min()), abs(pivot_teacher.values.max()))
    im = ax.imshow(pivot_teacher.values, cmap='RdYlGn', aspect='auto',
                  vmin=-vmax, vmax=vmax)
    
    ax.set_xticks(range(len(pivot_teacher.columns)))
    ax.set_xticklabels(pivot_teacher.columns)
    ax.set_yticks(range(len(pivot_teacher.index)))
    ax.set_yticklabels(pivot_teacher.index)
    
    ax.set_xlabel('Temperature (T)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Alpha (α)', fontsize=13, fontweight='bold')
    ax.set_title('Improvement over Teacher Model (FedAvg)\n(Positive is Better)',
                fontsize=15, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax, label='Improvement (%)')
    cbar.ax.tick_params(labelsize=11)
    
    # 标注数值
    for i in range(len(pivot_teacher.index)):
        for j in range(len(pivot_teacher.columns)):
            value = pivot_teacher.values[i, j]
            text_color = 'white' if abs(value) > abs(pivot_teacher.values).mean() else 'black'
            ax.text(j, i, f'{value:+.2f}', ha='center', va='center',
                   color=text_color, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_improvement_teacher.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 已保存: heatmap_improvement_teacher.png")
    
    # 4. 折线图：Temperature的影响（固定alpha）
    print("\n生成折线图: Temperature的影响")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    alphas_sorted = sorted(results_df['alpha'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas_sorted)))
    
    for idx, alpha_val in enumerate(alphas_sorted):
        subset = results_df[results_df['alpha'] == alpha_val].sort_values('temperature')
        ax.plot(subset['temperature'], subset['student_acc'],
               marker='o', label=f'α={alpha_val}', linewidth=2.5, 
               markersize=9, color=colors[idx])
    
    ax.set_xlabel('Temperature (T)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Student Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Effect of Temperature on Student Performance\n(Different Alpha Values)',
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_effect.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 已保存: temperature_effect.png")
    
    # 5. 折线图：Alpha的影响（固定temperature）
    print("\n生成折线图: Alpha的影响")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    temps_sorted = sorted(results_df['temperature'].unique())
    colors = plt.cm.plasma(np.linspace(0, 1, len(temps_sorted)))
    
    for idx, temp_val in enumerate(temps_sorted):
        subset = results_df[results_df['temperature'] == temp_val].sort_values('alpha')
        ax.plot(subset['alpha'], subset['student_acc'],
               marker='s', label=f'T={temp_val}', linewidth=2.5,
               markersize=9, color=colors[idx])
    
    ax.set_xlabel('Alpha (α)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Student Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Effect of Alpha on Student Performance\n(Different Temperature Values)',
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='best', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alpha_effect.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 已保存: alpha_effect.png")
    
    # 6. 对比柱状图
    print("\n生成对比柱状图")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 计算平均值
    avg_baseline = results_df['baseline_acc'].mean()
    avg_teacher = results_df['teacher_acc'].mean()
    avg_student = results_df['student_acc'].mean()
    max_student = results_df['student_acc'].max()
    
    categories = ['Baseline\n(Small Node)', 'Teacher\n(FedAvg)', 'Student\n(Avg KD)', 'Student\n(Best KD)']
    values = [avg_baseline, avg_teacher, avg_student, max_student]
    colors_bar = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728']
    
    bars = ax.bar(categories, values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 在柱子上标注数值
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.2f}%', ha='center', va='bottom',
               fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Performance Comparison: Baseline vs Teacher vs Student (KD)',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim([min(values) - 5, max(values) + 3])
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_bar.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 已保存: comparison_bar.png")
    
    # 7. 3D曲面图
    print("\n生成3D曲面图")
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建网格
        temperatures = sorted(results_df['temperature'].unique())
        alphas = sorted(results_df['alpha'].unique())
        T_grid, A_grid = np.meshgrid(temperatures, alphas)
        
        # 准备Z值
        Z = pivot_student.values
        
        # 绘制曲面
        surf = ax.plot_surface(T_grid, A_grid, Z, cmap='viridis', 
                              alpha=0.9, edgecolor='none')
        
        # 添加轮廓线
        ax.contour(T_grid, A_grid, Z, zdir='z', offset=Z.min()-1, 
                  cmap='viridis', alpha=0.5, linewidths=2)
        
        ax.set_xlabel('Temperature (T)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel('Alpha (α)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_zlabel('Student Accuracy (%)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_title('Student Performance vs KD Parameters (3D Surface)',
                    fontsize=15, fontweight='bold', pad=20)
        
        fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'surface_3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 已保存: surface_3d.png")
    except Exception as e:
        print(f"  ⚠ 无法生成3D图表: {e}")
    
    print(f"\n✓ 所有可视化图表已保存到: {output_dir}")


def print_summary_statistics(results_df):
    """打印结果统计摘要"""
    print("\n" + "="*80)
    print("实验结果统计摘要")
    print("="*80)
    
    # 最佳参数组合
    best_idx = results_df['student_acc'].idxmax()
    best_result = results_df.iloc[best_idx]
    
    print(f"\n最佳参数组合:")
    print(f"  Temperature: {best_result['temperature']}")
    print(f"  Alpha: {best_result['alpha']}")
    print(f"  学生模型准确率: {best_result['student_acc']:.2f}%")
    print(f"  基线模型准确率: {best_result['baseline_acc']:.2f}%")
    print(f"  教师模型准确率: {best_result['teacher_acc']:.2f}%")
    print(f"  相比基线提升: {best_result['improvement_over_baseline']:+.2f}%")
    print(f"  相比教师提升: {best_result['improvement_over_teacher']:+.2f}%")
    
    # 统计信息
    print(f"\n整体统计:")
    print(f"  实验总数: {len(results_df)}")
    print(f"  学生准确率范围: {results_df['student_acc'].min():.2f}% - {results_df['student_acc'].max():.2f}%")
    print(f"  学生准确率均值: {results_df['student_acc'].mean():.2f}%")
    print(f"  学生准确率标准差: {results_df['student_acc'].std():.2f}%")
    print(f"  平均相比基线提升: {results_df['improvement_over_baseline'].mean():+.2f}%")
    print(f"  平均相比教师提升: {results_df['improvement_over_teacher'].mean():+.2f}%")
    
    # Temperature分析
    print(f"\n按Temperature分析 (平均学生准确率):")
    temp_analysis = results_df.groupby('temperature')['student_acc'].agg(['mean', 'std'])
    for temp, row in temp_analysis.iterrows():
        print(f"  T={temp}: {row['mean']:.2f}% ± {row['std']:.2f}%")
    
    # Alpha分析
    print(f"\n按Alpha分析 (平均学生准确率):")
    alpha_analysis = results_df.groupby('alpha')['student_acc'].agg(['mean', 'std'])
    for alpha, row in alpha_analysis.iterrows():
        print(f"  α={alpha}: {row['mean']:.2f}% ± {row['std']:.2f}%")
    
    print("="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='可视化KD参数研究结果')
    parser.add_argument('--csv', type=str, required=True, 
                       help='实验结果CSV文件路径')
    parser.add_argument('--output', type=str, default='visualizations',
                       help='输出目录 (默认: visualizations)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("KD参数研究结果可视化工具")
    print("="*80)
    
    # 加载结果
    results_df = load_results(args.csv)
    
    # 打印统计摘要
    print_summary_statistics(results_df)
    
    # 生成可视化
    create_visualizations(results_df, args.output)
    
    print("\n✓ 完成!")


if __name__ == '__main__':
    # 如果没有提供参数，尝试自动找到最新的结果文件
    if len(sys.argv) == 1:
        results_dir = 'param_study_results'
        if os.path.exists(results_dir):
            csv_files = list(Path(results_dir).glob('kd_params_study_*.csv'))
            if csv_files:
                latest_csv = max(csv_files, key=os.path.getmtime)
                print(f"自动检测到最新结果文件: {latest_csv}")
                
                results_df = load_results(str(latest_csv))
                print_summary_statistics(results_df)
                create_visualizations(results_df, 'param_study_results/visualizations')
                
                print("\n✓ 完成!")
            else:
                print("错误: 在 param_study_results/ 目录下找不到结果文件")
                print("使用方法: python visualize_results.py --csv <path_to_csv>")
        else:
            print("错误: 找不到 param_study_results/ 目录")
            print("使用方法: python visualize_results.py --csv <path_to_csv>")
    else:
        main()
