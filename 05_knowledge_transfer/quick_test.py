"""
知识迁移实验 - 快速测试版本
使用较少的训练轮数进行快速验证
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from knowledge_transfer_experiment import run_knowledge_transfer_experiment, save_results
import torch


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 运行快速测试实验
    results = run_knowledge_transfer_experiment(
        num_large_nodes=4,
        large_to_small_ratio=10,
        num_rounds=5,                   # 5个循环迭代
        local_epochs=2,                 # 每个round的FedAvg本地训练2个epoch
        kd_epochs=2,                    # 每个round中KD训练2个epoch
        lr=0.1,
        temperature=4.0,
        alpha=0.7,
        batch_size=128,
        device=device,
        seed=42
    )
    
    # 保存结果
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'transfer_results')
    save_results(results, output_dir)
    
    print("\n快速测试完成！如需完整实验，请运行 knowledge_transfer_experiment.py")
