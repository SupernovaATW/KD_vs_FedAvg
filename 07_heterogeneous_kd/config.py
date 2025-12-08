"""
配置文件 - 异构模型知识蒸馏实验
"""

import argparse


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='异构模型知识蒸馏实验')
    
    # 数据相关参数
    parser.add_argument('--num-large-nodes', type=int, default=4,
                       help='大数据节点数量 (默认: 4, 2个ResNet8 + 2个ResNet18)')
    parser.add_argument('--large-to-small-ratio', type=int, default=10,
                       help='大小节点数据量比例 (默认: 10)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='批次大小 (默认: 128)')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='数据加载线程数 (默认: 2)')
    
    # 训练参数
    parser.add_argument('--local-epochs', type=int, default=20,
                       help='大节点每个循环的本地训练epoch数 (默认: 20)')
    parser.add_argument('--small-node-epochs', type=int, default=20,
                       help='小节点(ResNet34)训练epoch数 (默认: 20)')
    parser.add_argument('--large-node-kd-epochs', type=int, default=20,
                       help='大节点从小节点学习的KD epoch数 (默认: 20)')
    
    # 循环迭代参数
    parser.add_argument('--num-cycles', type=int, default=100,
                       help='知识迁移循环次数 (默认: 10)')
    
    # 学习率
    parser.add_argument('--lr', type=float, default=0.1,
                       help='学习率 (默认: 0.1)')
    
    # KD参数网格 - 小节点从大节点学习
    parser.add_argument('--small-temperatures', type=float, nargs='+',
                       default=[2.0, 4.0, 6.0],
                       help='小节点学习的Temperature参数列表 (默认: [2.0, 4.0, 6.0])')
    parser.add_argument('--small-alphas', type=float, nargs='+',
                       default=[0.5, 0.7, 0.9],
                       help='小节点学习的Alpha参数列表 (默认: [0.5, 0.7, 0.9])')
    
    # KD参数网格 - 大节点从小节点学习
    parser.add_argument('--large-temperatures', type=float, nargs='+',
                       default=[2.0, 4.0, 6.0],
                       help='大节点学习的Temperature参数列表 (默认: [2.0, 4.0, 6.0])')
    parser.add_argument('--large-alphas', type=float, nargs='+',
                       default=[0.5, 0.7, 0.9],
                       help='大节点学习的Alpha参数列表 (默认: [0.5, 0.7, 0.9])')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=7000,
                       help='随机种子 (默认: 7000)')
    parser.add_argument('--output-dir', type=str, default='hetero_results',
                       help='输出目录 (默认: hetero_results)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='跳过可视化生成')
    
    return parser.parse_args()


def print_config(args):
    """打印实验配置"""
    print("="*80)
    print("异构模型知识蒸馏实验")
    print("="*80)
    print(f"大数据节点数量: {args.num_large_nodes} (2个ResNet8 + 2个ResNet18)")
    print(f"小数据节点模型: ResNet34")
    print(f"大小节点数据量比例: {args.large_to_small_ratio}:1")
    print(f"批次大小: {args.batch_size}")
    print(f"数据加载线程数: {args.num_workers}")
    print("-"*80)
    print(f"大节点本地训练epochs: {args.local_epochs}")
    print(f"小节点KD训练epochs: {args.small_node_epochs}")
    print(f"大节点KD训练epochs: {args.large_node_kd_epochs}")
    print(f"循环次数: {args.num_cycles}")
    print(f"学习率: {args.lr}")
    print("-"*80)
    print(f"小节点学习参数:")
    print(f"  - Temperatures: {args.small_temperatures}")
    print(f"  - Alphas: {args.small_alphas}")
    print(f"大节点学习参数:")
    print(f"  - Temperatures: {args.large_temperatures}")
    print(f"  - Alphas: {args.large_alphas}")
    print("-"*80)
    print(f"随机种子: {args.seed}")
    print(f"输出目录: {args.output_dir}")
    print("="*80)
    print()
