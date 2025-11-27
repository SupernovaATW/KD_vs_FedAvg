import torch
import argparse
import json
import pandas as pd
from datetime import datetime
import logging
import os
from itertools import product

from models import ResNet18
from data_loader import get_split_cifar10_dataloaders
from kd_training import train_kd_pipeline


def setup_logging(log_dir='logs'):
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'kd_tuning_{timestamp}.log')
    
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


def run_kd_experiment(temperature, alpha, epochs, batch_size, lr, device, num_workers=2):
    """运行单次KD实验"""
    print(f"\n{'='*80}")
    print(f"实验配置: Temperature={temperature}, Alpha={alpha}")
    print(f"{'='*80}")
    
    logging.info(f"\n{'='*80}")
    logging.info(f"实验配置: Temperature={temperature}, Alpha={alpha}, Epochs={epochs}")
    logging.info(f"{'='*80}")
    
    # 加载数据
    trainloader_teacher, trainloader_student, testloader = get_split_cifar10_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        split_ratio=0.5
    )
    
    # 训练
    teacher_model, student_model, kd_history, student_best_acc = train_kd_pipeline(
        ResNet18,
        trainloader_teacher,
        trainloader_student,
        testloader,
        teacher_epochs=epochs,
        student_epochs=epochs,
        lr=lr,
        device=device,
        temperature=temperature,
        alpha=alpha
    )
    
    teacher_final_acc = kd_history['teacher']['test_acc'][-1] if kd_history['teacher']['test_acc'] else 0
    student_final_acc = kd_history['student']['test_acc'][-1]
    
    result = {
        'temperature': temperature,
        'alpha': alpha,
        'teacher_final_acc': teacher_final_acc,
        'student_best_acc': student_best_acc,
        'student_final_acc': student_final_acc,
        'improvement': student_best_acc - teacher_final_acc
    }
    
    logging.info(f"结果: Teacher={teacher_final_acc:.2f}%, Student Best={student_best_acc:.2f}%, Improvement={result['improvement']:.2f}%")
    
    return result


def main(args):
    """主函数：遍历参数组合"""
    log_file = setup_logging(args.log_dir)
    
    print("\n" + "="*80)
    print("知识蒸馏参数调优实验")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {args.device}")
    print(f"Epoch数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"日志文件: {log_file}")
    print("="*80)
    
    logging.info("="*80)
    logging.info("知识蒸馏参数调优实验")
    logging.info("="*80)
    logging.info(f"基础配置:")
    logging.info(f"  - Epoch数: {args.epochs}")
    logging.info(f"  - 批次大小: {args.batch_size}")
    logging.info(f"  - 学习率: {args.lr}")
    logging.info(f"  - 设备: {args.device}")
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and args.device == 'cuda':
        logging.warning("CUDA不可用，将使用CPU")
    
    # 定义参数网格
    if args.custom_params:
        # 使用命令行指定的参数
        temperatures = args.temperatures
        alphas = args.alphas
    else:
        # 默认参数网格
        temperatures = [2.0, 3.0, 4.0, 5.0, 6.0]
        alphas = [0.3, 0.5, 0.7, 0.9]
    
    print(f"\n测试参数组合:")
    print(f"  Temperature: {temperatures}")
    print(f"  Alpha: {alphas}")
    print(f"  总共 {len(temperatures) * len(alphas)} 个实验\n")
    
    logging.info(f"\n测试参数组合:")
    logging.info(f"  Temperature: {temperatures}")
    logging.info(f"  Alpha: {alphas}")
    logging.info(f"  总共 {len(temperatures) * len(alphas)} 个实验")
    
    # 运行所有实验
    results = []
    total_experiments = len(temperatures) * len(alphas)
    current_exp = 0
    
    for temperature in temperatures:
        for alpha in alphas:
            current_exp += 1
            print(f"\n[{current_exp}/{total_experiments}] 运行实验...")
            logging.info(f"\n[{current_exp}/{total_experiments}] 运行实验...")
            
            try:
                result = run_kd_experiment(
                    temperature=temperature,
                    alpha=alpha,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=device,
                    num_workers=args.num_workers
                )
                results.append(result)
            except Exception as e:
                error_msg = f"实验失败 (T={temperature}, α={alpha}): {str(e)}"
                print(f"❌ {error_msg}")
                logging.error(error_msg)
                continue
    
    # 创建结果DataFrame
    df = pd.DataFrame(results)
    
    # 按student_best_acc降序排列
    df = df.sort_values('student_best_acc', ascending=False)
    
    # 打印结果表格
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)
    print(df.to_string(index=False))
    
    logging.info("\n" + "="*80)
    logging.info("实验结果汇总")
    logging.info("="*80)
    logging.info("\n" + df.to_string(index=False))
    
    # 找出最佳参数
    best_result = df.iloc[0]
    print("\n" + "="*80)
    print("最佳参数配置")
    print("="*80)
    print(f"Temperature: {best_result['temperature']}")
    print(f"Alpha: {best_result['alpha']}")
    print(f"教师模型最终准确率: {best_result['teacher_final_acc']:.2f}%")
    print(f"学生模型最佳准确率: {best_result['student_best_acc']:.2f}%")
    print(f"学生模型最终准确率: {best_result['student_final_acc']:.2f}%")
    print(f"提升幅度: {best_result['improvement']:.2f}%")
    print("="*80)
    
    logging.info("\n" + "="*80)
    logging.info("最佳参数配置")
    logging.info("="*80)
    logging.info(f"Temperature: {best_result['temperature']}")
    logging.info(f"Alpha: {best_result['alpha']}")
    logging.info(f"教师模型最终准确率: {best_result['teacher_final_acc']:.2f}%")
    logging.info(f"学生模型最佳准确率: {best_result['student_best_acc']:.2f}%")
    logging.info(f"提升幅度: {best_result['improvement']:.2f}%")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存CSV
    csv_path = f'kd_tuning_results_{timestamp}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {csv_path}")
    logging.info(f"结果已保存到: {csv_path}")
    
    # 保存JSON
    json_path = f'kd_tuning_results_{timestamp}.json'
    results_dict = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'device': device,
            'temperatures': temperatures,
            'alphas': alphas
        },
        'best_config': {
            'temperature': float(best_result['temperature']),
            'alpha': float(best_result['alpha']),
            'student_best_acc': float(best_result['student_best_acc'])
        },
        'all_results': results
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)
    print(f"详细结果已保存到: {json_path}")
    logging.info(f"详细结果已保存到: {json_path}")
    
    # 创建可视化热力图（如果有多个参数）
    if len(temperatures) > 1 and len(alphas) > 1:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 创建透视表
            pivot_table = df.pivot_table(
                values='student_best_acc',
                index='alpha',
                columns='temperature',
                aggfunc='mean'
            )
            
            # 绘制热力图
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto')
            
            # 设置刻度
            ax.set_xticks(np.arange(len(pivot_table.columns)))
            ax.set_yticks(np.arange(len(pivot_table.index)))
            ax.set_xticklabels(pivot_table.columns)
            ax.set_yticklabels(pivot_table.index)
            
            # 标签
            ax.set_xlabel('Temperature', fontsize=12)
            ax.set_ylabel('Alpha', fontsize=12)
            ax.set_title('Student Best Accuracy (%)', fontsize=14)
            
            # 添加数值标注
            for i in range(len(pivot_table.index)):
                for j in range(len(pivot_table.columns)):
                    text = ax.text(j, i, f'{pivot_table.values[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=10)
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Accuracy (%)', rotation=270, labelpad=20)
            
            plt.tight_layout()
            heatmap_path = f'kd_tuning_heatmap_{timestamp}.png'
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            print(f"热力图已保存到: {heatmap_path}")
            logging.info(f"热力图已保存到: {heatmap_path}")
            plt.close()
        except Exception as e:
            logging.warning(f"无法生成热力图: {str(e)}")
    
    print("\n" + "="*80)
    print(f"所有实验完成! 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    logging.info("\n" + "="*80)
    logging.info(f"所有实验完成! 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='知识蒸馏参数调优')
    
    # 基础配置
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.1, help='初始学习率')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='训练设备')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载线程数')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志目录')
    
    # 参数网格
    parser.add_argument('--custom_params', action='store_true', help='使用自定义参数组合')
    parser.add_argument('--temperatures', type=float, nargs='+', 
                       default=[2.0, 3.0, 4.0, 5.0, 6.0],
                       help='要测试的温度参数列表')
    parser.add_argument('--alphas', type=float, nargs='+',
                       default=[0.3, 0.5, 0.7, 0.9],
                       help='要测试的alpha参数列表')
    
    args = parser.parse_args()
    
    main(args)
