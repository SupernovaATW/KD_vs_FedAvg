"""
可视化现有结果文件
如果你已经运行过实验并保存了JSON结果，可以用这个脚本重新生成图表
"""

import os
import sys
import json
import argparse

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from knowledge_transfer_experiment import plot_results


def visualize_existing_results(json_path):
    """
    从已有的JSON结果文件生成可视化图表
    
    Args:
        json_path: JSON结果文件的路径
    """
    print(f"加载结果文件: {json_path}")
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # 提取时间戳（如果有）
    filename = os.path.basename(json_path)
    if 'transfer_results_' in filename:
        timestamp = filename.replace('transfer_results_', '').replace('.json', '')
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 获取输出目录
    output_dir = os.path.dirname(json_path)
    if not output_dir:
        output_dir = 'transfer_results'
    
    print(f"生成图表到: {output_dir}")
    plot_results(results, output_dir, timestamp)
    print("\n可视化完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化知识迁移实验结果')
    parser.add_argument('json_file', type=str, nargs='?', 
                       help='JSON结果文件路径')
    parser.add_argument('--dir', type=str, default='transfer_results',
                       help='结果目录（如果不指定json_file，将使用目录中最新的文件）')
    
    args = parser.parse_args()
    
    if args.json_file:
        # 使用指定的JSON文件
        if not os.path.exists(args.json_file):
            print(f"错误: 文件不存在: {args.json_file}")
            sys.exit(1)
        visualize_existing_results(args.json_file)
    else:
        # 在目录中查找最新的JSON文件
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, args.dir)
        
        if not os.path.exists(results_dir):
            print(f"错误: 目录不存在: {results_dir}")
            print("请先运行实验或指定正确的结果目录")
            sys.exit(1)
        
        # 查找所有JSON结果文件
        json_files = [f for f in os.listdir(results_dir) 
                     if f.startswith('transfer_results_') and f.endswith('.json')]
        
        if not json_files:
            print(f"错误: 在 {results_dir} 中没有找到结果文件")
            print("请先运行实验")
            sys.exit(1)
        
        # 使用最新的文件
        json_files.sort()
        latest_file = json_files[-1]
        json_path = os.path.join(results_dir, latest_file)
        
        print(f"找到 {len(json_files)} 个结果文件，使用最新的: {latest_file}")
        visualize_existing_results(json_path)
