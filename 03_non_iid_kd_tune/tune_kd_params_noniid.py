import os
import sys
import torch
import argparse
import json
import pandas as pd
from datetime import datetime
import logging
import numpy as np
from typing import Any, Optional, Sequence, Tuple, cast

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'kd_non_iid_tuning_results')
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.models import ResNet18
from common.data_loader import get_noniid_cifar10_dataloaders
from common.kd_training import train_kd_pipeline


def setup_logging(log_dir=DEFAULT_OUTPUT_DIR):
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'kd_tuning_noniid_{timestamp}.log')
    
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


def run_kd_experiment(temperature, alpha, epochs, batch_size, lr, device, dirichlet_alpha,
                      num_workers=2, seed=7000, visualize_data=False, output_dir=DEFAULT_OUTPUT_DIR):
    """运行单次KD实验"""
    print(f"\n{'='*80}")
    print(f"实验配置: Temperature={temperature}, Alpha={alpha}, Dirichlet={dirichlet_alpha}")
    print(f"{'='*80}")
    
    logging.info(f"\n{'='*80}")
    logging.info(f"实验配置: Temperature={temperature}, Alpha={alpha}, Dirichlet={dirichlet_alpha}, Epochs={epochs}")
    logging.info(f"{'='*80}")
    
    # 加载Non-IID数据
    distribution_report: Optional[str] = None
    loaders = get_noniid_cifar10_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        alpha=dirichlet_alpha,
        num_clients=2,
        seed=seed,
        visualize=visualize_data,
        save_dir=output_dir,
        return_distribution=True
    )

    trainloader_teacher: Any = None
    trainloader_student: Any = None
    testloader: Any = None

    if isinstance(loaders, tuple):
        if len(loaders) == 4:
            lt_teacher, lt_student, lt_test, lt_report = cast(Tuple[Any, Any, Any, Optional[str]], loaders)
            trainloader_teacher, trainloader_student, testloader = lt_teacher, lt_student, lt_test
            distribution_report = lt_report
        elif len(loaders) == 3 and isinstance(loaders[0], Sequence):
            client_loaders = list(loaders[0])
            if len(client_loaders) < 2:
                raise ValueError("需要至少两个客户端以运行KD实验")
            trainloader_teacher, trainloader_student = client_loaders[:2]
            testloader = loaders[1]
            maybe_report = loaders[2]
            distribution_report = maybe_report if isinstance(maybe_report, str) else None
        elif len(loaders) == 2 and isinstance(loaders[0], Sequence):
            client_loaders = list(loaders[0])
            if len(client_loaders) < 2:
                raise ValueError("需要至少两个客户端以运行KD实验")
            trainloader_teacher, trainloader_student = client_loaders[:2]
            testloader = loaders[1]
        else:
            raise TypeError("无法解析Non-IID数据加载器的返回结构")
    else:
        raise TypeError("get_noniid_cifar10_dataloaders 应返回tuple结构")

    if distribution_report:
        logging.info("\n" + distribution_report)
    
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
        'dirichlet_alpha': dirichlet_alpha,
        'teacher_final_acc': teacher_final_acc,
        'student_best_acc': student_best_acc,
        'student_final_acc': student_final_acc,
        'improvement': student_best_acc - teacher_final_acc
    }
    
    logging.info(f"结果: Teacher={teacher_final_acc:.2f}%, Student Best={student_best_acc:.2f}%, Improvement={result['improvement']:.2f}%")
    
    return result


def main(args):
    """主函数：遍历参数组合"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = args.log_dir or args.output_dir
    log_file = setup_logging(log_dir)
    
    print("\n" + "="*80)
    print("知识蒸馏参数调优实验 (Non-IID)")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {args.device}")
    print(f"Epoch数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"Dirichlet Alpha列表: {args.dirichlet_alphas}")
    print(f"随机种子: {args.seed}")
    print(f"日志文件: {log_file}")
    print(f"结果输出目录: {args.output_dir}")
    print("="*80)
    
    logging.info("="*80)
    logging.info("知识蒸馏参数调优实验 (Non-IID)")
    logging.info("="*80)
    logging.info(f"基础配置:")
    logging.info(f"  设备: {args.device}")
    logging.info(f"  Epoch数: {args.epochs}")
    logging.info(f"  批次大小: {args.batch_size}")
    logging.info(f"  学习率: {args.lr}")
    logging.info(f"  Dirichlet Alpha列表: {args.dirichlet_alphas}")
    logging.info(f"  随机种子: {args.seed}")
    logging.info(f"  日志文件: {log_file}")
    logging.info(f"  结果输出目录: {args.output_dir}")
    
    # 参数网格
    temperatures = args.temperatures
    alphas = args.alphas
    
    logging.info(f"\n参数网格:")
    logging.info(f"  Temperature: {temperatures}")
    logging.info(f"  Alpha: {alphas}")
    logging.info(f"  Dirichlet Alpha: {args.dirichlet_alphas}")
    total_experiments = len(temperatures) * len(alphas) * len(args.dirichlet_alphas)
    logging.info(f"  总实验数: {total_experiments}")
    logging.info("="*80)
    
    # 存储所有结果
    all_results = []
    
    # 遍历所有参数组合
    current_experiment = 0
    visualized_alphas = set()
    
    for dirichlet_alpha in args.dirichlet_alphas:
        for temperature in temperatures:
            for alpha in alphas:
                current_experiment += 1
                print(f"\n{'='*80}")
                print(f"进度: {current_experiment}/{total_experiments}")
                print(f"当前Dirichlet Alpha: {dirichlet_alpha}")
                print(f"{'='*80}")
                
                # 每个Dirichlet alpha至少记录一次分布
                visualize_data = dirichlet_alpha not in visualized_alphas
                
                result = run_kd_experiment(
                    temperature=temperature,
                    alpha=alpha,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=args.device,
                    dirichlet_alpha=dirichlet_alpha,
                    num_workers=args.num_workers,
                    seed=args.seed,
                    visualize_data=visualize_data,
                    output_dir=args.output_dir
                )
                
                if visualize_data:
                    visualized_alphas.add(dirichlet_alpha)
                
                all_results.append(result)
            
            # 保存中间结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            intermediate_json = os.path.join(args.output_dir, f'kd_tuning_noniid_intermediate_{timestamp}.json')
            with open(intermediate_json, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
            logging.info(f"中间结果已保存: {intermediate_json}")
    
    # 转换为Daframe并分析
    df = pd.DataFrame(all_results)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = os.path.join(args.output_dir, f'kd_tuning_noniid_results_{timestamp}.csv')
    json_file = os.path.join(args.output_dir, f'kd_tuning_noniid_results_{timestamp}.json')
    
    df.to_csv(csv_file, index=False, encoding='utf-8')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    print(f"结果已保存到:")
    print(f"  - {csv_file}")
    print(f"  - {json_file}")
    print(f"  - {log_file}")
    
    # 分析最佳结果
    print("\n" + "="*80)
    print("最佳结果分析")
    print("="*80)
    
    if df.empty:
        print("没有收集到任何实验结果，请检查训练是否提前终止。")
        logging.warning("没有可用于分析的实验结果")
        return
    
    # 按student_best_acc排序
    df_sorted = df.sort_values('student_best_acc', ascending=False)
    
    print("\nTop 5 配置 (按Student最佳准确率):")
    print(df_sorted.head(min(5, len(df_sorted))).to_string(index=False))
    
    # 最佳配置
    best_config = df_sorted.iloc[0]
    print(f"\n最佳配置:")
    print(f"  Temperature: {best_config['temperature']}")
    print(f"  Alpha: {best_config['alpha']}")
    print(f"  Dirichlet Alpha: {best_config['dirichlet_alpha']}")
    print(f"  Teacher准确率: {best_config['teacher_final_acc']:.2f}%")
    print(f"  Student最佳准确率: {best_config['student_best_acc']:.2f}%")
    print(f"  提升: {best_config['improvement']:.2f}%")
    
    logging.info("\n" + "="*80)
    logging.info("实验完成")
    logging.info("="*80)
    logging.info(
        "最佳配置: Temperature=%s, Alpha=%s, Dirichlet Alpha=%s",
        best_config['temperature'],
        best_config['alpha'],
        best_config['dirichlet_alpha']
    )
    logging.info("Student最佳准确率: %.2f%%", best_config['student_best_acc'])
    
    # 生成Dirichlet Alpha特定的热力图
    if len(args.temperatures) > 1 and len(args.alphas) > 1:
        try:
            import matplotlib.pyplot as plt
            heatmap_paths = []
            alpha_order = list(dict.fromkeys(args.alphas))
            temp_order = list(dict.fromkeys(args.temperatures))
            cmap = plt.cm.get_cmap('YlOrRd').copy()
            cmap.set_bad(color='lightgray')
            for dir_alpha in args.dirichlet_alphas:
                dir_df = df[df['dirichlet_alpha'] == dir_alpha]
                if dir_df.empty:
                    continue
                pivot_table = dir_df.pivot_table(
                    values='student_best_acc',
                    index='alpha',
                    columns='temperature',
                    aggfunc='mean'
                )
                pivot_table = pivot_table.reindex(index=alpha_order, columns=temp_order)
                if pivot_table.dropna(how='all').empty:
                    continue
                data = pivot_table.to_numpy(dtype=float)
                masked_data = np.ma.masked_invalid(data)
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(masked_data, cmap=cmap, aspect='auto')
                ax.set_xticks(np.arange(len(temp_order)))
                ax.set_yticks(np.arange(len(alpha_order)))
                ax.set_xticklabels([f"{t}" for t in temp_order])
                ax.set_yticklabels([f"{a}" for a in alpha_order])
                ax.set_xlabel('Temperature', fontsize=12)
                ax.set_ylabel('Alpha', fontsize=12)
                ax.set_title(f'Student Best Accuracy (%)\nDirichlet α={dir_alpha}', fontsize=14)
                for i, alpha_val in enumerate(alpha_order):
                    for j, temp_val in enumerate(temp_order):
                        if not masked_data.mask[i, j]:
                            ax.text(j, i, f"{masked_data[i, j]:.2f}", ha='center', va='center', color='black', fontsize=9)
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Accuracy (%)', rotation=270, labelpad=20)
                plt.tight_layout()
                dir_tag = str(dir_alpha).replace('.', 'p')
                heatmap_path = os.path.join(args.output_dir, f'kd_tuning_noniid_heatmap_dir{dir_tag}_{timestamp}.png')
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                heatmap_paths.append(heatmap_path)
                plt.close()
            if heatmap_paths:
                print("\n热力图已生成:")
                for path in heatmap_paths:
                    print(f"  - {path}")
                logging.info("生成的热力图: %s", heatmap_paths)
        except Exception as exc:
            logging.warning("无法生成热力图: %s", exc)
    
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KD参数调优 (Non-IID)')
    
    # 基础参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.1, help='学习率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载器工作进程数')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='日志目录（默认写入输出目录）')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='结果输出目录')
    parser.add_argument('--seed', type=int, default=7000, help='随机种子')
    
    # Non-IID参数
    parser.add_argument('--dirichlet_alphas', type=float, nargs='+', default=[1.0, 0.5, 0.1],
                        help='Dirichlet分布的alpha参数列表，越小越Non-IID')
    
    # 调优参数范围
    parser.add_argument('--temperatures', type=float, nargs='+', 
                        default=[2.0, 3.0, 4.0, 5.0, 6.0],
                        help='温度参数列表')
    parser.add_argument('--alphas', type=float, nargs='+',
                        default=[0.3, 0.5, 0.7, 0.9],
                        help='Alpha参数列表')
    
    args = parser.parse_args()
    
    print(f"使用设备: {args.device}")
    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    main(args)
