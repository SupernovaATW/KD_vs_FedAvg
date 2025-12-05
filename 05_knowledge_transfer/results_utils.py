"""
结果可视化和保存
"""

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-darkgrid')
from datetime import datetime


def plot_results(results, output_dir='transfer_results', timestamp=None):
    """
    绘制实验结果可视化图表
    
    Args:
        results: 实验结果字典
        output_dir: 输出目录
        timestamp: 时间戳
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    exp_dir = os.path.join(output_dir, f'exp_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 提取数据
    initial_fedavg_acc = results['stages'][0]['best_acc']
    baseline_acc = results['baseline']['best_acc']
    
    kd_accs = []
    fedavg_accs = []
    
    for stage in results['stages'][1:]:
        kd_accs.append(stage['kd_stage']['best_acc'])
        fedavg_accs.append(stage['fedavg_stage']['best_acc'])
    
    num_cycles = len(kd_accs)
    
    # 图1: 准确率变化曲线
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # X轴：循环次数
    x = np.arange(num_cycles + 1)
    
    # 准确率序列：初始FedAvg -> KD1 -> FedAvg1 -> KD2 -> FedAvg2 -> ...
    accs_sequence = [initial_fedavg_acc]
    for i in range(num_cycles):
        accs_sequence.append(kd_accs[i])
        accs_sequence.append(fedavg_accs[i])
    
    x_sequence = np.arange(len(accs_sequence))
    
    # 绘制主曲线
    ax.plot(x_sequence, accs_sequence, 'b-o', linewidth=2.5, markersize=8, 
           label='Knowledge Transfer Cycle', alpha=0.9)
    
    # 绘制基线
    ax.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=2.5,
              label='Baseline (Small Node Alone)', alpha=0.7)
    
    # 标注KD和FedAvg阶段
    for i in range(num_cycles):
        kd_x = 1 + i * 2
        fedavg_x = 2 + i * 2
        ax.annotate(f'KD{i+1}', xy=(kd_x, kd_accs[i]), xytext=(5, 5),
                   textcoords='offset points', fontsize=9, alpha=0.7)
        ax.annotate(f'FA{i+1}', xy=(fedavg_x, fedavg_accs[i]), xytext=(5, -15),
                   textcoords='offset points', fontsize=9, alpha=0.7)
    
    ax.set_xlabel('Stage', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Knowledge Transfer Cycle: Accuracy Evolution', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # 添加结果文本框
    final_acc = fedavg_accs[-1]
    textstr = f'Final Results:\n'
    textstr += f'Final FedAvg: {final_acc:.2f}%\n'
    textstr += f'Initial FedAvg: {initial_fedavg_acc:.2f}%\n'
    textstr += f'Baseline: {baseline_acc:.2f}%\n'
    textstr += f'Improvement over baseline: +{final_acc - baseline_acc:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, 'accuracy_evolution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存图表: {plot_path}")
    
    # 图2: 最终结果对比
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = ['Initial\nFedAvg', 'Baseline\n(Small Node)', f'Final\nFedAvg\n(Cycle {num_cycles})']
    accuracies = [initial_fedavg_acc, baseline_acc, fedavg_accs[-1]]
    colors_bar = ['#3498db', '#95a5a6', '#e74c3c']
    
    bars = ax.bar(models, accuracies, color=colors_bar, alpha=0.8, 
                 edgecolor='black', linewidth=2)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Performance Comparison: FedAvg vs Knowledge Transfer', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, min(100, max(accuracies) * 1.15))
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加配置信息
    config = results['config']
    config_text = f"Config: {config['num_cycles']} cycles, "
    config_text += f"T={config['temperature']}, α={config['alpha']}"
    ax.text(0.5, 0.02, config_text, transform=ax.transAxes,
           fontsize=10, ha='center', style='italic',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plot_path = os.path.join(exp_dir, 'final_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存图表: {plot_path}")
    
    # 图3: 每个循环的提升效果
    if num_cycles > 1:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        cycles = np.arange(1, num_cycles + 1)
        improvements_kd = [kd - baseline_acc for kd in kd_accs]
        improvements_fedavg = [fa - baseline_acc for fa in fedavg_accs]
        
        width = 0.35
        x = np.arange(num_cycles)
        
        bars1 = ax.bar(x - width/2, improvements_kd, width, label='After KD',
                      color='#3498db', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, improvements_fedavg, width, label='After FedAvg',
                      color='#e74c3c', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Cycle', fontsize=14, fontweight='bold')
        ax.set_ylabel('Improvement over Baseline (%)', fontsize=14, fontweight='bold')
        ax.set_title('Improvement over Baseline per Cycle', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(cycles)
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        plot_path = os.path.join(exp_dir, 'improvement_per_cycle.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  保存图表: {plot_path}")


def save_results(results, output_dir='transfer_results'):
    """
    保存实验结果
    
    Args:
        results: 实验结果字典
        output_dir: 输出目录
    
    Returns:
        timestamp: 时间戳
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建实验目录
    exp_dir = os.path.join(output_dir, f'exp_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存JSON格式的详细结果
    json_path = os.path.join(exp_dir, 'transfer_results.json')
    
    # 移除不可序列化的对象（如history中的numpy数组）
    results_to_save = {}
    for key, value in results.items():
        if key == 'stages':
            results_to_save[key] = []
            for stage in value:
                stage_copy = {}
                for sk, sv in stage.items():
                    if isinstance(sv, dict) and 'history' in sv:
                        # 简化history，只保留关键指标
                        stage_copy[sk] = {
                            'best_acc': sv.get('best_acc', 0),
                            'improvement_over_baseline': sv.get('improvement_over_baseline', 0),
                            'improvement_over_teacher': sv.get('improvement_over_teacher', 0),
                            'improvement_over_student': sv.get('improvement_over_student', 0)
                        }
                    else:
                        stage_copy[sk] = sv
                results_to_save[key].append(stage_copy)
        elif key == 'baseline':
            results_to_save[key] = {'best_acc': value.get('best_acc', 0)}
        else:
            results_to_save[key] = value
    
    with open(json_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"详细结果已保存到: {json_path}")
    
    # 保存CSV格式结果
    csv_path = os.path.join(exp_dir, 'transfer_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Stage', 'Best Accuracy (%)', 'Notes'])
        
        writer.writerow(['Initial FedAvg', f"{results['stages'][0]['best_acc']:.2f}", 
                        '4 large nodes training'])
        writer.writerow(['Baseline (Small Node)', f"{results['baseline']['best_acc']:.2f}",
                        'Small node alone, no KD'])
        
        for i, stage in enumerate(results['stages'][1:], 1):
            writer.writerow([f'Cycle {i} - KD', f"{stage['kd_stage']['best_acc']:.2f}",
                           f"Small node KD from cycle {i-1} teacher"])
            writer.writerow([f'Cycle {i} - FedAvg', f"{stage['fedavg_stage']['best_acc']:.2f}",
                           'FedAvg with student model'])
        
        writer.writerow([''])
        writer.writerow(['Configuration', '', ''])
        for key, value in results['config'].items():
            writer.writerow([key, value, ''])
    
    print(f"结果摘要已保存到: {csv_path}")
    
    return timestamp
