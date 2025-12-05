"""
实验运行器 - 知识迁移循环实验
"""

from common.experiment_utils import run_knowledge_transfer_cycle


def run_knowledge_transfer_experiment(large_trainloaders, small_trainloader, testloader,
                                      args, device='cuda'):
    """
    运行知识迁移循环实验（调用common中的通用函数）
    
    Args:
        large_trainloaders: 大数据节点的训练数据加载器列表
        small_trainloader: 小数据节点的训练数据加载器
        testloader: 测试数据加载器
        args: 命令行参数
        device: 设备
    
    Returns:
        实验结果字典
    """
    config_dict = {
        'num_large_nodes': args.num_large_nodes,
        'large_to_small_ratio': args.large_to_small_ratio,
        'seed': args.seed
    }
    
    return run_knowledge_transfer_cycle(
        large_trainloaders=large_trainloaders,
        small_trainloader=small_trainloader,
        testloader=testloader,
        fedavg_rounds=1,  # 每个循环聚合1次
        local_epochs=args.local_epochs,
        kd_epochs=args.kd_epochs,
        num_cycles=args.num_cycles,
        temperature=args.temperature,
        alpha=args.alpha,
        lr=args.lr,
        device=device,
        verbose=True,
        config_dict=config_dict
    )
