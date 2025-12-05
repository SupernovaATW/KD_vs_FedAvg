"""
实验运行器 - KD参数研究（循环迭代版本）
"""

from common.experiment_utils import run_knowledge_transfer_cycle


def run_single_experiment(temperature, alpha, large_trainloaders, small_trainloader, 
                         testloader, args, device='cuda'):
    """
    运行单次实验（固定的T和α，循环迭代版本）
    调用common中的通用函数
    
    Args:
        temperature: 蒸馏温度
        alpha: 蒸馏损失权重
        large_trainloaders: 4个大数据节点的训练数据
        small_trainloader: 小数据节点的训练数据
        testloader: 测试数据
        args: 命令行参数
        device: 设备
    
    Returns:
        实验结果字典
    """
    print(f"\n{'='*80}")
    print(f"实验: Temperature={temperature}, Alpha={alpha}")
    print(f"{'='*80}")
    
    results = run_knowledge_transfer_cycle(
        large_trainloaders=large_trainloaders,
        small_trainloader=small_trainloader,
        testloader=testloader,
        fedavg_rounds=1,  # 每个循环聚合1次
        local_epochs=args.local_epochs,
        kd_epochs=args.kd_epochs,
        num_cycles=args.num_cycles,
        temperature=temperature,
        alpha=alpha,
        lr=args.lr,
        device=device,
        verbose=False,  # 参数扫描时不打印详细信息
        config_dict=None
    )
    
    # 添加温度和alpha到结果
    results['temperature'] = temperature
    results['alpha'] = alpha
    
    # 打印简化摘要
    summary = results['summary']
    print(f"\n结果汇总:")
    print(f"  基线: {summary['baseline_acc']:.2f}%")
    print(f"  最终教师: {summary['final_teacher_acc']:.2f}%")
    print(f"  最终学生: {summary['final_student_acc']:.2f}%")
    print(f"  最佳学生: {summary['best_student_acc']:.2f}%")
    print(f"  总提升(vs基线): {summary['total_improvement_over_baseline']:+.2f}%")
    
    return results
