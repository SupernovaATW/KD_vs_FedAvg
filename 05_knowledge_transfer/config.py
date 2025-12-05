"""
配置文件 - 命令行参数解析
"""

import argparse


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='知识迁移循环实验')
    
    # 数据相关参数
    parser.add_argument('--num-large-nodes', type=int, default=4,
                       help='大数据节点数量 (默认: 4)')
    parser.add_argument('--large-to-small-ratio', type=int, default=10,
                       help='大小节点数据量比例 (默认: 10)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='批次大小 (默认: 128)')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='数据加载线程数 (默认: 2)')
    
    # 每个循环的训练参数
    parser.add_argument('--local-epochs', type=int, default=20,
                       help='大节点每个循环的本地训练epoch数 (默认: 20)')
    parser.add_argument('--kd-epochs', type=int, default=20,
                       help='小数据节点KD训练epoch数 (默认: 20)')
    
    # 循环迭代参数
    parser.add_argument('--num-cycles', type=int, default=100,
                       help='知识迁移循环次数 (默认: 100)')
    
    # 学习率
    parser.add_argument('--lr', type=float, default=0.1,
                       help='学习率 (默认: 0.1)')
    
    # KD参数
    parser.add_argument('--temperature', type=float, default=4.0,
                       help='蒸馏温度 (默认: 4.0)')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='蒸馏损失权重 (默认: 0.7)')
    
    # 实验参数
    parser.add_argument('--num-cycles', type=int, default=5,
                       help='知识迁移循环次数 (默认: 5)')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--output-dir', type=str, default='transfer_results',
                       help='输出目录 (默认: transfer_results)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='跳过可视化生成')
    
    return parser.parse_args()


def print_config(args):
    """打印实验配置"""
    print("="*80)
    print("知识迁移循环实验")
    print("="*80)
    print("\n实验配置:")
    print(f"  大数据节点数量: {args.num_large_nodes}")
    print(f"  大小节点数据比例: {args.large_to_small_ratio}:1")
    print(f"  批次大小: {args.batch_size}")
    print(f"  大节点训练: {args.local_epochs}epochs (然后聚合)")
    print(f"  小节点KD训练: {args.kd_epochs}epochs")
    print(f"  循环次数: {args.num_cycles}")
    print(f"  KD参数 - Temperature: {args.temperature}, Alpha: {args.alpha}")
    print(f"  学习率: {args.lr}")
    print(f"  随机种子: {args.seed}")
    print(f"  输出目录: {args.output_dir}")
    print("="*80)
