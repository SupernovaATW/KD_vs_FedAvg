import torch
import argparse
import json
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import os

from models import ResNet18
from data_loader import get_cifar10_dataloaders, get_split_cifar10_dataloaders
from fedavg_training import train_fedavg
from kd_training import train_kd_pipeline


def plot_comparison(fedavg_history, kd_history, save_path='comparison_plot.png'):
    """
    绘制两种方法的训练曲线对比图
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 训练准确率
    axes[0, 0].plot(fedavg_history['train_acc'], label='FedAvg', marker='o', markersize=3)
    axes[0, 0].plot(kd_history['student']['train_acc'], label='KD (Student)', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Training Accuracy Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 测试准确率
    axes[0, 1].plot(fedavg_history['test_acc'], label='FedAvg', marker='o', markersize=3)
    axes[0, 1].plot(kd_history['student']['test_acc'], label='KD (Student)', marker='s', markersize=3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Test Accuracy Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 训练损失
    axes[1, 0].plot(fedavg_history['train_loss'], label='FedAvg', marker='o', markersize=3)
    axes[1, 0].plot(kd_history['student']['train_loss'], label='KD (Student)', marker='s', markersize=3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 测试损失
    axes[1, 1].plot(fedavg_history['test_loss'], label='FedAvg', marker='o', markersize=3)
    axes[1, 1].plot(kd_history['student']['test_loss'], label='KD (Student)', marker='s', markersize=3)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Test Loss Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存到: {save_path}")
    plt.close()


def save_results(results, filename='experiment_results.json'):
    """保存实验结果到JSON文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"实验结果已保存到: {filename}")
    logging.info(f"实验结果已保存到: {filename}")


def setup_logging(log_dir='logs'):
    """设置日志系统"""
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'experiment_{timestamp}.log')
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logging.info(f"日志文件已创建: {log_file}")
    return log_file


def main(args):
    """主实验函数"""
    # 设置日志
    log_file = setup_logging(args.log_dir)
    
    print("\n" + "="*80)
    print("CIFAR-10 + ResNet18: 联邦平均 vs 知识蒸馏 对比实验")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {args.device}")
    print(f"Epoch数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"知识蒸馏温度: {args.temperature}")
    print(f"知识蒸馏alpha: {args.alpha}")
    print(f"日志文件: {log_file}")
    print("="*80)
    
    logging.info("="*80)
    logging.info("CIFAR-10 + ResNet18: 联邦平均 vs 知识蒸馏 对比实验")
    logging.info("="*80)
    logging.info(f"实验配置:")
    logging.info(f"  - Epoch数: {args.epochs}")
    logging.info(f"  - 批次大小: {args.batch_size}")
    logging.info(f"  - 学习率: {args.lr}")
    logging.info(f"  - 设备: {args.device}")
    logging.info(f"  - 知识蒸馏温度: {args.temperature}")
    logging.info(f"  - 知识蒸馏alpha: {args.alpha}")
    logging.info(f"  - 运行FedAvg: {args.run_fedavg}")
    logging.info(f"  - 运行KD: {args.run_kd}")
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and args.device == 'cuda':
        warning_msg = "⚠ 警告: CUDA不可用，将使用CPU"
        print(warning_msg)
        logging.warning("CUDA不可用，将使用CPU")
    
    logging.info(f"实际使用设备: {device}")
    
    results = {
        'config': vars(args),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': device,
        'log_file': log_file
    }
    
    # ========== 方法1: 联邦平均 ==========
    if args.run_fedavg:
        print("\n" + "="*80)
        print("实验1: 联邦平均 (同时训练两个模型并平均)")
        print("="*80)
        
        logging.info("\n" + "="*80)
        logging.info("实验1: 联邦平均 (FedAvg)")
        logging.info("="*80)
        
        # 加载分割的数据（模拟两个客户端）
        trainloader1, trainloader2, testloader = get_split_cifar10_dataloaders(
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            split_ratio=0.5
        )
        
        logging.info(f"数据加载完成 - 训练集1: {len(trainloader1)}批次, 训练集2: {len(trainloader2)}批次, 测试集: {len(testloader)}批次")
        
        fedavg_model, fedavg_history, fedavg_best_acc = train_fedavg(
            ResNet18,
            trainloader1,
            trainloader2,
            testloader,
            num_epochs=args.epochs,
            lr=args.lr,
            device=device
        )
        
        logging.info(f"FedAvg训练完成 - 最佳测试准确率: {fedavg_best_acc:.2f}%")
        
        results['fedavg'] = {
            'best_accuracy': fedavg_best_acc,
            'final_train_acc': fedavg_history['train_acc'][-1],
            'final_test_acc': fedavg_history['test_acc'][-1],
            'history': fedavg_history
        }
    
    # ========== 方法2: 知识蒸馏 ==========
    if args.run_kd:
        print("\n" + "="*80)
        print("实验2: 知识蒸馏 (先训练教师，再蒸馏到学生)")
        print("="*80)
        
        logging.info("\n" + "="*80)
        logging.info("实验2: 知识蒸馏 (Knowledge Distillation)")
        logging.info("="*80)
        
        # 加载分割的数据集（公平对比：教师和学生各用一半数据）
        trainloader_teacher, trainloader_student, testloader = get_split_cifar10_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split_ratio=0.5
        )
        
        logging.info(f"数据加载完成 - 教师训练集: {len(trainloader_teacher)}批次, 学生训练集: {len(trainloader_student)}批次, 测试集: {len(testloader)}批次")
        
        teacher_model, student_model, kd_history, kd_best_acc = train_kd_pipeline(
            ResNet18,
            trainloader_teacher,
            trainloader_student,
            testloader,
            teacher_epochs=args.epochs,
            student_epochs=args.epochs,
            lr=args.lr,
            device=device,
            temperature=args.temperature,
            alpha=args.alpha
        )
        
        logging.info(f"KD训练完成 - 学生模型最佳准确率: {kd_best_acc:.2f}%")
        
        results['kd'] = {
            'teacher_best_accuracy': kd_history['teacher']['test_acc'][-1] if kd_history['teacher']['test_acc'] else 0,
            'student_best_accuracy': kd_best_acc,
            'final_train_acc': kd_history['student']['train_acc'][-1],
            'final_test_acc': kd_history['student']['test_acc'][-1],
            'history': kd_history
        }
    
    # ========== 结果对比 ==========
    print("\n" + "="*80)
    print("实验结果对比")
    print("="*80)
    
    logging.info("\n" + "="*80)
    logging.info("实验结果对比")
    logging.info("="*80)
    
    if args.run_fedavg:
        print(f"\n联邦平均 (FedAvg):")
        print(f"  最佳测试准确率: {results['fedavg']['best_accuracy']:.2f}%")
        print(f"  最终训练准确率: {results['fedavg']['final_train_acc']:.2f}%")
        print(f"  最终测试准确率: {results['fedavg']['final_test_acc']:.2f}%")
        
        logging.info(f"联邦平均 (FedAvg):")
        logging.info(f"  最佳测试准确率: {results['fedavg']['best_accuracy']:.2f}%")
        logging.info(f"  最终训练准确率: {results['fedavg']['final_train_acc']:.2f}%")
        logging.info(f"  最终测试准确率: {results['fedavg']['final_test_acc']:.2f}%")
    
    if args.run_kd:
        print(f"\n知识蒸馏 (KD):")
        print(f"  教师模型最终准确率: {results['kd']['teacher_best_accuracy']:.2f}%")
        print(f"  学生模型最佳准确率: {results['kd']['student_best_accuracy']:.2f}%")
        print(f"  学生模型最终训练准确率: {results['kd']['final_train_acc']:.2f}%")
        print(f"  学生模型最终测试准确率: {results['kd']['final_test_acc']:.2f}%")
        
        logging.info(f"知识蒸馏 (KD):")
        logging.info(f"  教师模型最终准确率: {results['kd']['teacher_best_accuracy']:.2f}%")
        logging.info(f"  学生模型最佳准确率: {results['kd']['student_best_accuracy']:.2f}%")
        logging.info(f"  学生模型最终训练准确率: {results['kd']['final_train_acc']:.2f}%")
        logging.info(f"  学生模型最终测试准确率: {results['kd']['final_test_acc']:.2f}%")
    
    if args.run_fedavg and args.run_kd:
        diff = results['kd']['student_best_accuracy'] - results['fedavg']['best_accuracy']
        print(f"\n差异分析:")
        print(f"  知识蒸馏相比联邦平均: {diff:+.2f}%")
        
        logging.info(f"\n差异分析:")
        logging.info(f"  知识蒸馏相比联邦平均: {diff:+.2f}%")
        
        if diff > 0:
            result_msg = "✓ 知识蒸馏效果更好!"
            print(f"  {result_msg}")
            logging.info(f"  {result_msg}")
        elif diff < 0:
            result_msg = "✓ 联邦平均效果更好!"
            print(f"  {result_msg}")
            logging.info(f"  {result_msg}")
        else:
            result_msg = "≈ 两种方法效果相当"
            print(f"  {result_msg}")
            logging.info(f"  {result_msg}")
        
        # 绘制对比图
        if not args.no_plot:
            plot_comparison(
                results['fedavg']['history'],
                results['kd']['history'],
                save_path=args.plot_path
            )
            logging.info(f"对比图已保存: {args.plot_path}")
    
    # 保存结果
    if args.save_results:
        save_results(results, filename=args.results_path)
    
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "="*80)
    print(f"实验完成! 结束时间: {end_time}")
    print("="*80)
    
    logging.info("\n" + "="*80)
    logging.info(f"实验完成! 结束时间: {end_time}")
    logging.info("="*80)
    logging.info(f"日志已保存到: {log_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-10 + ResNet18: FedAvg vs KD 对比实验')
    
    # 实验配置
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.1, help='初始学习率')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='训练设备')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载线程数')
    
    # 知识蒸馏参数
    parser.add_argument('--temperature', type=float, default=4.0, help='知识蒸馏温度参数')
    parser.add_argument('--alpha', type=float, default=0.7, help='知识蒸馏损失权重')
    
    # 运行选项
    parser.add_argument('--run_fedavg', action='store_true', default=True, help='运行联邦平均实验')
    parser.add_argument('--run_kd', action='store_true', default=True, help='运行知识蒸馏实验')
    parser.add_argument('--no_plot', action='store_true', help='不生成对比图')
    parser.add_argument('--save_results', action='store_true', default=True, help='保存实验结果')
    
    # 输出路径
    parser.add_argument('--plot_path', type=str, default='comparison_plot.png', help='对比图保存路径')
    parser.add_argument('--results_path', type=str, default='experiment_results.json', help='结果JSON保存路径')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志文件保存目录')
    
    args = parser.parse_args()
    
    main(args)
