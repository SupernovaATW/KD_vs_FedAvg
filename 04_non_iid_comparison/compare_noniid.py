import torch
import argparse
import json
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import os
import sys
import pandas as pd
import numpy as np
from typing import Any, Optional, Tuple, cast

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.models import ResNet18
from common.data_loader import get_noniid_cifar10_dataloaders
from common.fedavg_training import train_fedavg
from common.kd_training import train_kd_pipeline


def setup_logging(log_dir='comparison_outputs'):
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'compare_noniid_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"日志文件已创建: {log_file}")
    return log_file


def plot_comparison(results, save_dir='comparison_outputs', dirichlet_alpha=None):
    """绘制对比图"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 提取数据
    dirichlet_alphas = [r['dirichlet_alpha'] for r in results]
    fedavg_accs = [r['fedavg_best_acc'] for r in results]
    kd_accs = [r['kd_best_acc'] for r in results]
    improvements = [r['improvement'] for r in results]
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 准确率对比
    x = range(len(dirichlet_alphas))
    width = 0.35
    
    axes[0].bar([i - width/2 for i in x], fedavg_accs, width, label='FedAvg', alpha=0.8)
    axes[0].bar([i + width/2 for i in x], kd_accs, width, label='KD (Best Params)', alpha=0.8)
    axes[0].set_xlabel('Dirichlet Alpha (Non-IID Level)', fontsize=12)
    axes[0].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[0].set_title('FedAvg vs KD Performance Across Non-IID Levels', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'α={a}' for a in dirichlet_alphas])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 性能提升
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    axes[1].bar(x, improvements, color=colors, alpha=0.8)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Dirichlet Alpha (Non-IID Level)', fontsize=12)
    axes[1].set_ylabel('Improvement (%)', fontsize=12)
    axes[1].set_title('KD Improvement over FedAvg', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'α={a}' for a in dirichlet_alphas])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(save_dir, f'comparison_noniid_best_params_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存到: {plot_file}")
    plt.close()
    
    return plot_file


def plot_training_curves(results, save_dir='comparison_outputs', dirichlet_alpha=None):
    """绘制训练曲线"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    n_experiments = len(results)
    fig, axes = plt.subplots(n_experiments, 2, figsize=(15, 5*n_experiments))
    
    if n_experiments == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(results):
        dir_alpha = result['dirichlet_alpha']
        fedavg_history = result['fedavg_history']
        kd_history = result['kd_history']
        
        # 训练准确率
        axes[idx, 0].plot(fedavg_history['train_acc'], label='FedAvg', marker='o', markersize=3)
        axes[idx, 0].plot(kd_history['student']['train_acc'], label='KD', marker='s', markersize=3)
        axes[idx, 0].set_xlabel('Epoch')
        axes[idx, 0].set_ylabel('Accuracy (%)')
        axes[idx, 0].set_title(f'Training Accuracy (Dirichlet α={dir_alpha})')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # 测试准确率
        axes[idx, 1].plot(fedavg_history['test_acc'], label='FedAvg', marker='o', markersize=3)
        axes[idx, 1].plot(kd_history['student']['test_acc'], label='KD', marker='s', markersize=3)
        axes[idx, 1].set_xlabel('Epoch')
        axes[idx, 1].set_ylabel('Accuracy (%)')
        axes[idx, 1].set_title(f'Test Accuracy (Dirichlet α={dir_alpha})')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(save_dir, f'training_curves_noniid_best_params_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {plot_file}")
    plt.close()
    
    return plot_file


def run_single_comparison(alpha, temperature, epochs, batch_size, lr, device, dirichlet_alpha, num_workers=2, seed=7000, visualize_data=False):
    """运行单次对比实验"""
    print(f"\n{'='*80}")
    print(f"对比实验: Dirichlet Alpha={dirichlet_alpha}, KD Alpha={alpha}, Temperature={temperature}")
    print(f"{'='*80}")
    
    logging.info(f"\n{'='*80}")
    logging.info(f"对比实验: Dirichlet Alpha={dirichlet_alpha}, KD Alpha={alpha}, Temperature={temperature}")
    logging.info(f"{'='*80}")
    
    # 加载Non-IID数据
    print("\n加载Non-IID数据...")
    trainloader1, trainloader2, testloader, distribution_report = cast(
        Tuple[Any, Any, Any, Optional[str]],
        get_noniid_cifar10_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            alpha=dirichlet_alpha,
            num_clients=2,
            seed=seed,
            visualize=visualize_data,
            save_dir='comparison_outputs',
            return_distribution=True
        )
    )
    if isinstance(distribution_report, str) and distribution_report:
        logging.info("\n" + distribution_report)
    
    # 训练FedAvg
    print("\n" + "="*80)
    print("开始训练 FedAvg")
    print("="*80)
    logging.info("\n开始训练 FedAvg")
    
    fedavg_model, fedavg_history, fedavg_best_acc = train_fedavg(
        ResNet18,
        trainloader1,
        trainloader2,
        testloader,
        num_epochs=epochs,
        lr=lr,
        device=device
    )
    
    fedavg_final_acc = fedavg_history['test_acc'][-1]
    print(f"FedAvg - 最佳准确率: {fedavg_best_acc:.2f}%, 最终准确率: {fedavg_final_acc:.2f}%")
    logging.info(f"FedAvg - 最佳准确率: {fedavg_best_acc:.2f}%, 最终准确率: {fedavg_final_acc:.2f}%")
    
    # 训练KD
    print("\n" + "="*80)
    print("开始训练 Knowledge Distillation")
    print("="*80)
    logging.info("\n开始训练 Knowledge Distillation")
    
    teacher_model, student_model, kd_history, kd_best_acc = train_kd_pipeline(
        ResNet18,
        trainloader1,
        trainloader2,
        testloader,
        teacher_epochs=epochs,
        student_epochs=epochs,
        lr=lr,
        device=device,
        temperature=temperature,
        alpha=alpha
    )
    
    kd_final_acc = kd_history['student']['test_acc'][-1]
    teacher_final_acc = kd_history['teacher']['test_acc'][-1]
    print(f"KD - Teacher最终准确率: {teacher_final_acc:.2f}%")
    print(f"KD - Student最佳准确率: {kd_best_acc:.2f}%, 最终准确率: {kd_final_acc:.2f}%")
    logging.info(f"KD - Teacher最终准确率: {teacher_final_acc:.2f}%")
    logging.info(f"KD - Student最佳准确率: {kd_best_acc:.2f}%, 最终准确率: {kd_final_acc:.2f}%")
    
    # 计算提升
    improvement = kd_best_acc - fedavg_best_acc
    
    result = {
        'alpha': alpha,
        'temperature': temperature,
        'dirichlet_alpha': dirichlet_alpha,
        'fedavg_best_acc': fedavg_best_acc,
        'fedavg_final_acc': fedavg_final_acc,
        'kd_best_acc': kd_best_acc,
        'kd_final_acc': kd_final_acc,
        'teacher_final_acc': teacher_final_acc,
        'improvement': improvement,
        'fedavg_history': fedavg_history,
        'kd_history': kd_history
    }
    
    print(f"\n{'='*80}")
    print(f"对比结果 (Alpha={alpha}):")
    print(f"  FedAvg最佳准确率: {fedavg_best_acc:.2f}%")
    print(f"  KD最佳准确率: {kd_best_acc:.2f}%")
    print(f"  提升: {improvement:+.2f}%")
    print(f"{'='*80}")
    
    logging.info(f"\n对比结果 (Alpha={alpha}):")
    logging.info(f"  FedAvg最佳准确率: {fedavg_best_acc:.2f}%")
    logging.info(f"  KD最佳准确率: {kd_best_acc:.2f}%")
    logging.info(f"  提升: {improvement:+.2f}%")
    
    return result


def main(args):
    """主函数"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    log_file = setup_logging(args.log_dir)
    
    # 根据03实验结果，定义每个dirichlet_alpha下的最佳KD参数
    best_kd_params = {
        1.0: {'temperature': 2.0, 'alpha': 0.3},  # improvement=11.30%, student_best_acc=92.1%
        0.5: {'temperature': 6.0, 'alpha': 0.3},  # improvement=3.46%, student_best_acc=90.82%
        0.1: {'temperature': 5.0, 'alpha': 0.3}   # improvement=24.35%, student_best_acc=85.09%
    }
    
    print("\n" + "="*80)
    print("FedAvg vs KD 对比实验 (Non-IID) - 使用最佳参数")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {args.device}")
    print(f"Epoch数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"Dirichlet Alpha列表: {args.dirichlet_alphas}")
    print(f"随机种子: {args.seed}")
    print("\n每个Dirichlet Alpha的最佳KD参数 (来自03实验):")
    for dir_alpha, params in best_kd_params.items():
        if dir_alpha in args.dirichlet_alphas:
            print(f"  Dirichlet α={dir_alpha}: Temperature={params['temperature']}, KD α={params['alpha']}")
    print(f"日志文件: {log_file}")
    print("="*80)
    
    logging.info("="*80)
    logging.info("FedAvg vs KD 对比实验 (Non-IID) - 使用最佳参数")
    logging.info("="*80)
    logging.info(f"基础配置:")
    logging.info(f"  设备: {args.device}")
    logging.info(f"  Epoch数: {args.epochs}")
    logging.info(f"  批次大小: {args.batch_size}")
    logging.info(f"  学习率: {args.lr}")
    logging.info(f"  Dirichlet Alpha列表: {args.dirichlet_alphas}")
    logging.info(f"  随机种子: {args.seed}")
    logging.info("\n每个Dirichlet Alpha的最佳KD参数:")
    for dir_alpha, params in best_kd_params.items():
        if dir_alpha in args.dirichlet_alphas:
            logging.info(f"  Dirichlet α={dir_alpha}: Temperature={params['temperature']}, KD α={params['alpha']}")
    
    # 运行所有对比实验
    all_results = []
    plot_files = []
    curves_files = []
    total_runs = len(args.dirichlet_alphas)
    current_run = 0
    visualized = False
    
    for dir_alpha in args.dirichlet_alphas:
        current_run += 1
        
        # 获取该dirichlet_alpha下的最佳KD参数
        if dir_alpha not in best_kd_params:
            print(f"\n警告: Dirichlet Alpha={dir_alpha} 没有预设的最佳参数，跳过")
            logging.warning(f"Dirichlet Alpha={dir_alpha} 没有预设的最佳参数，跳过")
            continue
        
        kd_params = best_kd_params[dir_alpha]
        temperature = kd_params['temperature']
        alpha = kd_params['alpha']
        
        print(f"\n{'-'*80}")
        print(f"Dirichlet Alpha: {dir_alpha} | 使用最佳参数: Temperature={temperature}, KD α={alpha} ({current_run}/{total_runs})")
        print(f"{'-'*80}")
        
        visualize_data = not visualized
        result = run_single_comparison(
            alpha=alpha,
            temperature=temperature,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            dirichlet_alpha=dir_alpha,
            num_workers=args.num_workers,
            seed=args.seed,
            visualize_data=visualize_data
        )
        
        if visualize_data:
            visualized = True
        
        all_results.append(result)
    
    if not all_results:
        print("没有生成任何结果")
        logging.error("没有生成任何结果")
        return
    
    # 针对所有结果绘制对比图
    plot_files.append(plot_comparison(all_results, save_dir=args.log_dir, dirichlet_alpha=None))
    curves_files.append(plot_training_curves(all_results, save_dir=args.log_dir, dirichlet_alpha=None))
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存详细结果（包含训练历史）
    json_file = os.path.join(args.log_dir, f'comparison_noniid_detailed_{timestamp}.json')
    
    # 准备可序列化的结果
    serializable_results = []
    for r in all_results:
        serializable_results.append({
            'alpha': r['alpha'],
            'temperature': r['temperature'],
            'dirichlet_alpha': r['dirichlet_alpha'],
            'fedavg_best_acc': r['fedavg_best_acc'],
            'fedavg_final_acc': r['fedavg_final_acc'],
            'kd_best_acc': r['kd_best_acc'],
            'kd_final_acc': r['kd_final_acc'],
            'teacher_final_acc': r['teacher_final_acc'],
            'improvement': r['improvement']
        })
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)
    
    # 保存简要结果到CSV
    df = pd.DataFrame(serializable_results)
    csv_file = os.path.join(args.log_dir, f'comparison_noniid_summary_{timestamp}.csv')
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    # 输出总结
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    print(f"结果已保存到:")
    print(f"  - {json_file}")
    print(f"  - {csv_file}")
    for f in plot_files:
        print(f"  - {f}")
    for f in curves_files:
        print(f"  - {f}")
    print(f"  - {log_file}")
    
    print("\n" + "="*80)
    print("结果总结")
    print("="*80)
    print(df.to_string(index=False))
    
    # 找出最佳配置
    if not df.empty:
        best_idx = int(df['kd_best_acc'].idxmax())
        best_result = df.iloc[best_idx]
        
        print("\n" + "="*80)
        print("最佳配置:")
        print(f"  Dirichlet Alpha: {best_result['dirichlet_alpha']}")
        print(f"  KD Alpha: {best_result['alpha']}")
        print(f"  Temperature: {best_result['temperature']}")
        print(f"  FedAvg最佳准确率: {best_result['fedavg_best_acc']:.2f}%")
        print(f"  KD最佳准确率: {best_result['kd_best_acc']:.2f}%")
        print(f"  提升: {best_result['improvement']:+.2f}%")
        print("="*80)
        
        logging.info("\n实验完成")
        logging.info(f"最佳Dirichlet Alpha: {best_result['dirichlet_alpha']}")
        logging.info(f"最佳KD Alpha: {best_result['alpha']}")
        logging.info(f"最佳Temperature: {best_result['temperature']}")
        logging.info(f"KD最佳准确率: {best_result['kd_best_acc']:.2f}%")
        logging.info(f"提升: {best_result['improvement']:+.2f}%")
    else:
        print("未生成任何结果，无法分析最佳配置。")
        logging.warning("未生成任何结果，无法分析最佳配置。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FedAvg vs KD 对比实验 (Non-IID)')
    
    # 基础参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.1, help='学习率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载器工作进程数')
    parser.add_argument('--log_dir', type=str, default='comparison_outputs', help='日志目录')
    parser.add_argument('--seed', type=int, default=7000, help='随机种子')
    
    # Non-IID参数
    parser.add_argument('--dirichlet_alphas', type=float, nargs='+', default=[1.0, 0.5, 0.1],
                        help='Dirichlet分布的alpha参数列表，越小越Non-IID (默认使用03实验的最佳参数配置)')
    
    args = parser.parse_args()
    
    print(f"使用设备: {args.device}")
    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    main(args)
