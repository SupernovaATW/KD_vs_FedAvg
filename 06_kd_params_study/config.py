"""
配置文件 - 命令行参数解析
"""

import argparse


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='知识蒸馏参数研究实验')
    
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

    
    # KD参数网格
    parser.add_argument('--temperatures', type=float, nargs='+',
                       default=[1, 2, 3, 4, 5, 6, 8, 10],
                       help='Temperature参数列表 (默认: [1, 2, 3, 4, 5, 6, 8, 10])')
    parser.add_argument('--alphas', type=float, nargs='+',
                       default=[0.1, 0.3, 0.5, 0.7, 0.9],
                       help='Alpha参数列表 (默认: [0.1, 0.3, 0.5, 0.7, 0.9])')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=7000,
                       help='随机种子 (默认: 7000)')
    parser.add_argument('--output-dir', type=str, default='param_study_results',
                       help='输出目录 (默认: param_study_results)')
    parser.add_argument('--no-visualize', action='store_true',
                       help='跳过可视化生成')
    
    return parser.parse_args()


def print_config(args):
    """打印实验配置"""
    print("="*80)
    print("知识蒸馏参数研究实验")
    print("="*80)
    print("\n实验配置:")
    print(f"  大数据节点数量: {args.num_large_nodes}")
    print(f"  大小节点数据比例: {args.large_to_small_ratio}:1")
    print(f"  批次大小: {args.batch_size}")
    print(f"  每个循环:")
    print(f"    - 大节点训练: {args.local_epochs}epochs (然后聚合)")
    print(f"    - 小节点KD: {args.kd_epochs}epochs")
    print(f"  总循环次数: {args.num_cycles}")
    print(f"  学习率: {args.lr}")
    print(f"  Temperature范围: {args.temperatures}")
    print(f"  Alpha范围: {args.alphas}")
    print(f"  随机种子: {args.seed}")
    print(f"  输出目录: {args.output_dir}")
    print("="*80)
