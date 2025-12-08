"""
异构模型知识蒸馏实验运行器

架构说明:
- 4个大数据节点: 2个ResNet8 + 2个ResNet18
- 1个小数据节点: ResNet34
- 大节点不做FedAvg，直接全部发给小节点
- 小节点使用4个大节点模型的平均logits做知识蒸馏
- 训练完成后，小节点的模型发回大节点
- 大节点使用小节点模型做知识蒸馏
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from tqdm import tqdm

from common.models import ResNet8, ResNet18, ResNet34
from common.kd_training import train_epoch_multi_teacher_kd, train_epoch_kd, evaluate
from common.training_utils import train_epoch_standard


def train_large_nodes_locally(models, trainloaders, lr, local_epochs, device):
    """
    在大节点上本地训练模型（不做联邦平均）
    
    Args:
        models: 模型列表 [ResNet8, ResNet8, ResNet18, ResNet18]
        trainloaders: 训练数据加载器列表
        lr: 学习率
        local_epochs: 本地训练epoch数
        device: 设备
    
    Returns:
        训练后的模型列表和统计信息
    """
    print("\n" + "="*80)
    print("阶段1: 大节点本地训练 (不做FedAvg)")
    print("="*80)
    
    criterion = nn.CrossEntropyLoss()
    stats = []
    
    for i, (model, trainloader) in enumerate(zip(models, trainloaders)):
        model_name = "ResNet8" if i < 2 else "ResNet18"
        print(f"\n训练大节点{i+1} ({model_name})...")
        
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=local_epochs)
        
        node_stats = {'train_loss': [], 'train_acc': []}
        
        for epoch in range(local_epochs):
            train_loss, train_acc = train_epoch_standard(
                model, trainloader, criterion, optimizer, device
            )
            scheduler.step()
            
            node_stats['train_loss'].append(train_loss)
            node_stats['train_acc'].append(train_acc)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{local_epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        
        stats.append(node_stats)
        print(f"  大节点{i+1}训练完成! 最终准确率: {train_acc:.2f}%")
    
    return models, stats


def train_small_node_with_multi_teacher(small_model, teacher_models, trainloader, testloader,
                                        lr, epochs, temperature, alpha, device):
    """
    小节点使用多个大节点模型的平均logits进行知识蒸馏
    
    Args:
        small_model: 小节点模型 (ResNet34)
        teacher_models: 教师模型列表 (4个大节点模型)
        trainloader: 小节点训练数据
        testloader: 测试数据
        lr: 学习率
        epochs: 训练epoch数
        temperature: 蒸馏温度
        alpha: 蒸馏损失权重
        device: 设备
    
    Returns:
        训练后的小节点模型和统计信息
    """
    print("\n" + "="*80)
    print("阶段2: 小节点从大节点学习 (多教师知识蒸馏)")
    print(f"Temperature: {temperature}, Alpha: {alpha}")
    print("="*80)
    
    optimizer = optim.SGD(small_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    stats = {
        'train_loss': [], 'train_kd_loss': [], 'train_ce_loss': [],
        'train_acc': [], 'test_loss': [], 'test_acc': []
    }
    
    best_test_acc = 0.0
    
    for epoch in range(epochs):
        # 训练
        train_loss, kd_loss, ce_loss, train_acc = train_epoch_multi_teacher_kd(
            small_model, teacher_models, trainloader, optimizer, device,
            temperature, alpha
        )
        
        # 测试
        test_loss, test_acc = evaluate(small_model, testloader, criterion, device)
        
        scheduler.step()
        
        # 记录统计
        stats['train_loss'].append(train_loss)
        stats['train_kd_loss'].append(kd_loss)
        stats['train_ce_loss'].append(ce_loss)
        stats['train_acc'].append(train_acc)
        stats['test_loss'].append(test_loss)
        stats['test_acc'].append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train - Loss: {train_loss:.4f}, KD: {kd_loss:.4f}, CE: {ce_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}% (Best: {best_test_acc:.2f}%)")
    
    print(f"\n小节点训练完成! 最佳测试准确率: {best_test_acc:.2f}%")
    
    return small_model, stats


def train_large_nodes_with_small_teacher(large_models, small_model, trainloaders, testloader,
                                        lr, epochs, temperature, alpha, device):
    """
    大节点从小节点学习 (单教师知识蒸馏)
    
    Args:
        large_models: 大节点模型列表
        small_model: 小节点模型 (作为教师)
        trainloaders: 大节点训练数据加载器列表
        testloader: 测试数据
        lr: 学习率
        epochs: 训练epoch数
        temperature: 蒸馏温度
        alpha: 蒸馏损失权重
        device: 设备
    
    Returns:
        训练后的大节点模型列表和统计信息
    """
    print("\n" + "="*80)
    print("阶段3: 大节点从小节点学习 (知识蒸馏)")
    print(f"Temperature: {temperature}, Alpha: {alpha}")
    print("="*80)
    
    criterion = nn.CrossEntropyLoss()
    all_stats = []
    
    for i, (large_model, trainloader) in enumerate(zip(large_models, trainloaders)):
        model_name = "ResNet8" if i < 2 else "ResNet18"
        print(f"\n训练大节点{i+1} ({model_name})从小节点(ResNet34)学习...")
        
        optimizer = optim.SGD(large_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        stats = {
            'train_loss': [], 'train_kd_loss': [], 'train_ce_loss': [],
            'train_acc': [], 'test_loss': [], 'test_acc': []
        }
        
        best_test_acc = 0.0
        
        for epoch in range(epochs):
            # 训练
            train_loss, kd_loss, ce_loss, train_acc = train_epoch_kd(
                large_model, small_model, trainloader, optimizer, device,
                temperature, alpha
            )
            
            # 测试
            test_loss, test_acc = evaluate(large_model, testloader, criterion, device)
            
            scheduler.step()
            
            # 记录统计
            stats['train_loss'].append(train_loss)
            stats['train_kd_loss'].append(kd_loss)
            stats['train_ce_loss'].append(ce_loss)
            stats['train_acc'].append(train_acc)
            stats['test_loss'].append(test_loss)
            stats['test_acc'].append(test_acc)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs}:")
                print(f"    Train - Loss: {train_loss:.4f}, KD: {kd_loss:.4f}, CE: {ce_loss:.4f}, Acc: {train_acc:.2f}%")
                print(f"    Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}% (Best: {best_test_acc:.2f}%)")
        
        all_stats.append(stats)
        print(f"  大节点{i+1}训练完成! 最佳测试准确率: {best_test_acc:.2f}%")
    
    return large_models, all_stats


def run_one_cycle(large_models, small_model, large_trainloaders, small_trainloader, testloader,
                 args, device):
    """
    运行一个完整的知识迁移循环
    
    流程:
    1. 大节点本地训练
    2. 小节点从大节点学习 (多教师KD)
    3. 大节点从小节点学习 (单教师KD)
    
    Args:
        large_models: 大节点模型列表
        small_model: 小节点模型
        large_trainloaders: 大节点训练数据加载器列表
        small_trainloader: 小节点训练数据加载器
        testloader: 测试数据加载器
        args: 命令行参数
        device: 设备
    
    Returns:
        更新后的模型和统计信息
    """
    # 阶段1: 大节点本地训练
    large_models, large_stats = train_large_nodes_locally(
        large_models, large_trainloaders, args.lr, args.local_epochs, device
    )
    
    # 阶段2: 小节点从大节点学习
    small_model, small_stats = train_small_node_with_multi_teacher(
        small_model, large_models, small_trainloader, testloader,
        args.lr, args.small_node_epochs, 
        args.small_temperature, args.small_alpha, device
    )
    
    # 阶段3: 大节点从小节点学习
    large_models, large_kd_stats = train_large_nodes_with_small_teacher(
        large_models, small_model, large_trainloaders, testloader,
        args.lr, args.large_node_kd_epochs,
        args.large_temperature, args.large_alpha, device
    )
    
    return large_models, small_model, {
        'large_local': large_stats,
        'small_from_large': small_stats,
        'large_from_small': large_kd_stats
    }


def run_heterogeneous_kd_experiment(large_trainloaders, small_trainloader, testloader,
                                   args, device='cuda'):
    """
    运行完整的异构模型知识蒸馏实验
    
    Args:
        large_trainloaders: 大节点训练数据加载器列表 (4个)
        small_trainloader: 小节点训练数据加载器
        testloader: 测试数据加载器
        args: 命令行参数
        device: 设备
    
    Returns:
        实验结果字典
    """
    print("\n" + "="*80)
    print("异构模型知识蒸馏实验")
    print("="*80)
    print(f"大节点: 2个ResNet8 + 2个ResNet18")
    print(f"小节点: 1个ResNet34")
    print(f"循环次数: {args.num_cycles}")
    print(f"小节点学习参数: T={args.small_temperature}, α={args.small_alpha}")
    print(f"大节点学习参数: T={args.large_temperature}, α={args.large_alpha}")
    print("="*80)
    
    # 初始化模型
    large_models = [
        ResNet8().to(device),   # 大节点1
        ResNet8().to(device),   # 大节点2
        ResNet18().to(device),  # 大节点3
        ResNet18().to(device),  # 大节点4
    ]
    small_model = ResNet34().to(device)
    
    # 记录每个循环的结果
    all_cycle_results = []
    criterion = nn.CrossEntropyLoss()
    
    for cycle in range(args.num_cycles):
        print("\n" + "#"*80)
        print(f"循环 {cycle+1}/{args.num_cycles}")
        print("#"*80)
        
        # 运行一个循环
        large_models, small_model, cycle_stats = run_one_cycle(
            large_models, small_model, large_trainloaders, small_trainloader,
            testloader, args, device
        )
        
        # 评估所有模型
        print("\n" + "-"*80)
        print(f"循环 {cycle+1} 最终评估:")
        print("-"*80)
        
        large_test_accs = []
        for i, model in enumerate(large_models):
            model_name = "ResNet8" if i < 2 else "ResNet18"
            _, test_acc = evaluate(model, testloader, criterion, device)
            large_test_accs.append(test_acc)
            print(f"  大节点{i+1} ({model_name}): {test_acc:.2f}%")
        
        _, small_test_acc = evaluate(small_model, testloader, criterion, device)
        print(f"  小节点 (ResNet34): {small_test_acc:.2f}%")
        print(f"  平均准确率: {np.mean(large_test_accs + [small_test_acc]):.2f}%")
        print("-"*80)
        
        # 记录结果
        cycle_result = {
            'cycle': cycle + 1,
            'large_test_accs': large_test_accs,
            'small_test_acc': small_test_acc,
            'avg_test_acc': np.mean(large_test_accs + [small_test_acc]),
            'stats': cycle_stats
        }
        all_cycle_results.append(cycle_result)
    
    # 汇总结果
    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)
    
    final_result = {
        'config': {
            'num_cycles': args.num_cycles,
            'local_epochs': args.local_epochs,
            'small_node_epochs': args.small_node_epochs,
            'large_node_kd_epochs': args.large_node_kd_epochs,
            'small_temperature': args.small_temperature,
            'small_alpha': args.small_alpha,
            'large_temperature': args.large_temperature,
            'large_alpha': args.large_alpha,
            'lr': args.lr,
            'seed': args.seed
        },
        'cycle_results': all_cycle_results,
        'final_large_test_accs': all_cycle_results[-1]['large_test_accs'],
        'final_small_test_acc': all_cycle_results[-1]['small_test_acc'],
        'final_avg_test_acc': all_cycle_results[-1]['avg_test_acc']
    }
    
    return final_result
