"""
通用实验运行工具
"""

import torch
import torch.nn as nn
from .models import ResNet18
from .training_utils import (
    train_fedavg_large_nodes,
    train_fedavg_with_initial_model,
    train_small_node_with_kd,
    train_small_node_baseline
)


def run_knowledge_transfer_cycle(large_trainloaders, small_trainloader, testloader,
                                  fedavg_rounds, local_epochs, kd_epochs,
                                  num_cycles, temperature, alpha, lr,
                                  device='cuda', verbose=True, config_dict=None):
    """
    运行循环知识迁移实验（简化版本，无预训练）
    
    实验流程（每个循环）：
    1. 大数据节点进行FedAvg训练 → 得到教师模型
    2. 小数据节点从教师模型进行KD学习 → 得到学生模型
    3. 使用学生模型作为下一轮的初始模型
    4. 重复num_cycles次
    
    Args:
        large_trainloaders: 大数据节点的训练数据加载器列表
        small_trainloader: 小数据节点的训练数据加载器
        testloader: 测试数据加载器
        fedavg_rounds: 每个循环中FedAvg通信轮次
        local_epochs: FedAvg每轮本地训练epoch数
        kd_epochs: 小数据节点KD训练epoch数
        num_cycles: 知识迁移循环次数
        temperature: 蒸馏温度
        alpha: 蒸馏损失权重
        lr: 学习率
        device: 设备
        verbose: 是否打印详细信息
        config_dict: 额外的配置信息（可选，用于记录）
    
    Returns:
        实验结果字典
    """
    results = {
        'config': config_dict or {},
        'cycles': []
    }
    
    # 更新基本配置
    results['config'].update({
        'fedavg_rounds': fedavg_rounds,
        'local_epochs': local_epochs,
        'kd_epochs': kd_epochs,
        'num_cycles': num_cycles,
        'temperature': temperature,
        'alpha': alpha,
        'lr': lr
    })
    
    # ========== 基线: 小数据节点独立训练 ==========
    if verbose:
        print("\n" + "="*80)
        print("基线: 小数据节点独立训练（不使用KD）")
        print("="*80)
    
    baseline_model, baseline_history, baseline_best_acc = train_small_node_baseline(
        ResNet18, small_trainloader, testloader,
        num_epochs=kd_epochs,
        lr=lr,
        device=device,
        verbose=verbose
    )
    
    results['baseline'] = {
        'best_acc': baseline_best_acc,
        'history': baseline_history
    }
    
    if verbose:
        print(f"\n基线完成! 小数据节点独立训练最佳准确率: {baseline_best_acc:.2f}%")
    
    # ========== 循环知识迁移 ==========
    current_model = None  # 第一轮从零开始，后续使用学生模型
    
    for cycle in range(num_cycles):
        if verbose:
            print("\n" + "="*80)
            print(f"循环 {cycle+1}/{num_cycles}")
            print("="*80)
        
        # 步骤1: 大数据节点FedAvg训练
        if verbose:
            print(f"\n[循环{cycle+1}] 步骤1: 大数据节点FedAvg训练")
        
        if current_model is None:
            # 第一轮：从零开始训练
            teacher_model, fedavg_history, fedavg_best_acc = train_fedavg_large_nodes(
                ResNet18, large_trainloaders, testloader,
                num_rounds=fedavg_rounds,
                local_epochs=local_epochs,
                lr=lr,
                device=device,
                verbose=verbose
            )
        else:
            # 后续循环：使用上一轮的学生模型继续训练
            teacher_model, fedavg_history, fedavg_best_acc = train_fedavg_with_initial_model(
                ResNet18, large_trainloaders, testloader, current_model,
                num_rounds=fedavg_rounds,
                local_epochs=local_epochs,
                lr=lr,
                device=device,
                verbose=verbose
            )
        
        if verbose:
            print(f"\n[循环{cycle+1}] 步骤1完成! 教师模型准确率: {fedavg_best_acc:.2f}%")
        
        # 步骤2: 小数据节点KD训练
        if verbose:
            print(f"\n[循环{cycle+1}] 步骤2: 小数据节点KD学习")
        
        student_model = ResNet18().to(device)
        student_model, kd_history, kd_best_acc = train_small_node_with_kd(
            student_model, teacher_model, small_trainloader, testloader,
            num_epochs=kd_epochs,
            lr=lr,
            device=device,
            temperature=temperature,
            alpha=alpha,
            verbose=verbose
        )
        
        if verbose:
            print(f"\n[循环{cycle+1}] 步骤2完成! 学生模型准确率: {kd_best_acc:.2f}%")
            print(f"  相比基线提升: {kd_best_acc - baseline_best_acc:+.2f}%")
            print(f"  相比教师提升: {kd_best_acc - fedavg_best_acc:+.2f}%")
        
        # 记录本轮结果
        results['cycles'].append({
            'cycle': cycle + 1,
            'teacher_acc': fedavg_best_acc,
            'student_acc': kd_best_acc,
            'improvement_over_baseline': kd_best_acc - baseline_best_acc,
            'improvement_over_teacher': kd_best_acc - fedavg_best_acc,
            'teacher_history': fedavg_history,
            'student_history': kd_history
        })
        
        # 更新当前模型为学生模型，用于下一轮
        current_model = student_model
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"循环 {cycle+1} 完成! 教师({fedavg_best_acc:.2f}%) → 学生({kd_best_acc:.2f}%)")
            print(f"{'='*80}")
    
    # 汇总最终结果
    final_teacher_acc = results['cycles'][-1]['teacher_acc']
    final_student_acc = results['cycles'][-1]['student_acc']
    
    results['summary'] = {
        'baseline_acc': baseline_best_acc,
        'final_teacher_acc': final_teacher_acc,
        'final_student_acc': final_student_acc,
        'best_student_acc': max(c['student_acc'] for c in results['cycles']),
        'best_teacher_acc': max(c['teacher_acc'] for c in results['cycles']),
        'total_improvement_over_baseline': final_student_acc - baseline_best_acc
    }
    
    return results
