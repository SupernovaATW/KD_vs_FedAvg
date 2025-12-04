"""
知识蒸馏参数研究实验
场景：5个节点的联邦学习
- 4个大数据节点(性能有限): 每个节点有大量数据
- 1个小数据节点(性能很好): 数据量只有大数据节点的1/10

实验目的：
研究不同的KD参数(temperature T 和 alpha α)对知识迁移效果的影响

实验设置：
- Temperature (T): [1, 2, 3, 4, 5, 6, 8, 10]
- Alpha (α): [0.1, 0.3, 0.5, 0.7, 0.9]

实验流程（针对每组参数）：
1. 4个大数据节点进行联邦平均(FedAvg)训练，得到聚合模型
2. 使用聚合模型作为教师模型
3. 对小数据节点(性能好)使用知识蒸馏，测试不同的T和α
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
plt.style.use('seaborn-v0_8-darkgrid')
import pandas as pd
from itertools import product
from tqdm import tqdm
import copy
import torchvision
import torchvision.transforms as transforms

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from common.models import ResNet18
from common.kd_training import train_epoch_kd, evaluate


def split_dataset_for_nodes(trainset, num_large_nodes=4, large_to_small_ratio=10, seed=42):
    """
    将训练集分配给5个节点：4个大数据节点 + 1个小数据节点
    
    Args:
        trainset: 训练数据集
        num_large_nodes: 大数据节点数量
        large_to_small_ratio: 大数据节点与小数据节点的数据量比例
        seed: 随机种子
    
    Returns:
        large_nodes_indices: 4个大数据节点的数据索引列表
        small_node_indices: 1个小数据节点的数据索引
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    total_samples = len(trainset)
    
    # 计算每个节点的样本数
    small_node_size = total_samples // (num_large_nodes * large_to_small_ratio + 1)
    large_node_size = small_node_size * large_to_small_ratio
    
    print(f"\n数据分配:")
    print(f"总样本数: {total_samples}")
    print(f"每个大数据节点: {large_node_size} 样本")
    print(f"小数据节点: {small_node_size} 样本")
    print(f"大小比例: {large_to_small_ratio}:1")
    
    # 随机打乱索引
    all_indices = np.arange(total_samples)
    np.random.shuffle(all_indices)
    
    # 分配给各个节点
    large_nodes_indices = []
    start_idx = 0
    
    for i in range(num_large_nodes):
        end_idx = start_idx + large_node_size
        large_nodes_indices.append(all_indices[start_idx:end_idx].tolist())
        start_idx = end_idx
    
    # 剩余的给小节点
    small_node_indices = all_indices[start_idx:start_idx + small_node_size].tolist()
    
    return large_nodes_indices, small_node_indices


def train_epoch_standard(model, trainloader, criterion, optimizer, device):
    """标准训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(trainloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def federated_averaging_multi(models):
    """
    多模型联邦平均
    
    Args:
        models: 模型列表
    
    Returns:
        平均后的模型
    """
    averaged_model = copy.deepcopy(models[0])
    
    with torch.no_grad():
        for param_name, param in averaged_model.named_parameters():
            # 计算所有模型对应参数的平均值
            param.data.zero_()
            for model in models:
                param.data += dict(model.named_parameters())[param_name].data
            param.data /= len(models)
    
    return averaged_model


def train_fedavg_large_nodes(model_class, trainloaders, testloader, num_rounds=50, 
                             local_epochs=5, lr=0.1, device='cuda', verbose=True):
    """
    训练4个大数据节点的联邦平均模型
    
    Args:
        model_class: 模型类
        trainloaders: 4个训练数据加载器的列表
        testloader: 测试数据加载器
        num_rounds: 通信轮次数（每轮进行一次聚合）
        local_epochs: 每个节点在每轮中的本地训练epoch数
        lr: 学习率
        device: 设备
        verbose: 是否打印详细信息
    
    Returns:
        聚合后的模型、历史记录、最佳准确率
    """
    if verbose:
        print("\n" + "="*70)
        print("阶段1: 训练4个大数据节点的联邦平均模型")
        print(f"通信轮次: {num_rounds}, 每轮本地训练: {local_epochs} epochs")
        print(f"总训练量: {num_rounds * local_epochs} epochs")
        print("="*70)
    
    num_nodes = len(trainloaders)
    total_epochs = num_rounds * local_epochs
    
    # 初始化模型
    models = [model_class().to(device) for _ in range(num_nodes)]
    
    # 确保所有模型初始参数相同
    for i in range(1, num_nodes):
        models[i].load_state_dict(models[0].state_dict())
    
    criterion = nn.CrossEntropyLoss()
    optimizers = [optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) 
                  for model in models]
    schedulers = [optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs) 
                  for opt in optimizers]
    
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'node_losses': [[] for _ in range(num_nodes)],
        'node_accs': [[] for _ in range(num_nodes)],
        'rounds': []
    }
    
    global_epoch = 0
    averaged_model = None
    
    for round_idx in range(num_rounds):
        if verbose:
            print(f"\n{'='*70}")
            print(f"通信轮次 {round_idx+1}/{num_rounds}")
            print(f"{'='*70}")
        
        round_losses = []
        round_accs = []
        
        # 每个节点进行本地训练
        for local_epoch in range(local_epochs):
            global_epoch += 1
            
            epoch_losses = []
            epoch_accs = []
            
            for i in range(num_nodes):
                loss, acc = train_epoch_standard(
                    models[i], trainloaders[i], criterion, optimizers[i], device
                )
                epoch_losses.append(loss)
                epoch_accs.append(acc)
                history['node_losses'][i].append(loss)
                history['node_accs'][i].append(acc)
            
            round_losses.append(np.mean(epoch_losses))
            round_accs.append(np.mean(epoch_accs))
            
            # 更新学习率
            for scheduler in schedulers:
                scheduler.step()
        
        # 联邦平均（每个round结束后）
        averaged_model = federated_averaging_multi(models)
        
        # 用平均后的参数更新所有模型
        for model in models:
            model.load_state_dict(averaged_model.state_dict())
        
        # 评估聚合后的模型
        test_loss, test_acc = evaluate(averaged_model, testloader, criterion, device)
        
        if verbose:
            print(f"  >>> Round {round_idx+1} 聚合模型 - Test Acc: {test_acc:.2f}%")
        
        # 记录本轮的历史
        history['train_loss'].extend(round_losses)
        history['train_acc'].extend(round_accs)
        history['test_loss'].extend([test_loss] * local_epochs)
        history['test_acc'].extend([test_acc] * local_epochs)
        history['rounds'].append({
            'round': round_idx + 1,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'avg_train_loss': np.mean(round_losses),
            'avg_train_acc': np.mean(round_accs)
        })
        
        # 更新最佳准确率
        if test_acc > best_acc:
            best_acc = test_acc
    
    if verbose:
        print(f"\n4个大数据节点的联邦平均训练完成!")
        print(f"完成 {num_rounds} 个通信轮次，共 {total_epochs} 个训练epoch")
        print(f"最佳测试准确率: {best_acc:.2f}%")
    
    return averaged_model, history, best_acc


def train_small_node_with_kd(student_model, teacher_model, trainloader, testloader,
                             num_epochs=100, lr=0.1, device='cuda',
                             temperature=4.0, alpha=0.7, verbose=True):
    """
    使用知识蒸馏训练小数据节点
    
    Args:
        student_model: 学生模型(小数据节点的模型)
        teacher_model: 教师模型(联邦平均后的聚合模型)
        trainloader: 小数据节点的训练数据
        testloader: 测试数据
        num_epochs: 训练轮数
        lr: 学习率
        device: 设备
        temperature: 蒸馏温度
        alpha: 蒸馏损失权重
        verbose: 是否打印详细信息
    
    Returns:
        训练后的模型、历史记录、最佳准确率
    """
    if verbose:
        print("\n" + "="*70)
        print(f"阶段2: 使用知识蒸馏训练小数据节点(T={temperature}, α={alpha})")
        print("="*70)
    
    optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_kd_loss': [],
        'train_ce_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, kd_loss, ce_loss, train_acc = train_epoch_kd(
            student_model, teacher_model, trainloader, optimizer, device,
            temperature, alpha
        )
        
        # 测试
        test_loss, test_acc = evaluate(student_model, testloader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_kd_loss'].append(kd_loss)
        history['train_ce_loss'].append(ce_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 更新最佳准确率
        if test_acc > best_acc:
            best_acc = test_acc
        
        # 只在关键epoch打印信息
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} - Test Acc: {test_acc:.2f}%, Best: {best_acc:.2f}%")
    
    if verbose:
        print(f"\n小数据节点训练完成! 最佳测试准确率: {best_acc:.2f}%")
    
    return student_model, history, best_acc


def run_single_experiment(temperature, alpha, large_trainloaders, small_trainloader, 
                         testloader, device='cuda', teacher_model=None):
    """
    运行单次实验（固定的T和α）
    
    Args:
        temperature: 蒸馏温度
        alpha: 蒸馏损失权重
        large_trainloaders: 4个大数据节点的训练数据
        small_trainloader: 小数据节点的训练数据
        testloader: 测试数据
        device: 设备
        teacher_model: 预训练的教师模型（如果提供，则跳过FedAvg训练）
    
    Returns:
        实验结果字典
    """
    print(f"\n{'='*80}")
    print(f"实验: Temperature={temperature}, Alpha={alpha}")
    print(f"{'='*80}")
    
    # 如果没有提供教师模型，则训练FedAvg得到教师模型
    if teacher_model is None:
        teacher_model, fedavg_history, fedavg_best_acc = train_fedavg_large_nodes(
            ResNet18, large_trainloaders, testloader,
            num_rounds=10, local_epochs=5, lr=0.1, device=device, verbose=False
        )
        print(f"FedAvg教师模型训练完成，测试准确率: {fedavg_best_acc:.2f}%")
    else:
        # 评估提供的教师模型
        criterion = nn.CrossEntropyLoss()
        _, fedavg_best_acc = evaluate(teacher_model, testloader, criterion, device)
        print(f"使用预训练教师模型，测试准确率: {fedavg_best_acc:.2f}%")
    
    # 使用知识蒸馏训练学生模型
    student_model = ResNet18().to(device)
    student_model, kd_history, kd_best_acc = train_small_node_with_kd(
        student_model, teacher_model, small_trainloader, testloader,
        num_epochs=100, lr=0.1, device=device,
        temperature=temperature, alpha=alpha, verbose=False
    )
    
    # 训练基线模型（小数据节点独立训练，不使用KD）
    baseline_model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(baseline_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    baseline_best_acc = 0.0
    for epoch in range(100):
        train_loss, train_acc = train_epoch_standard(
            baseline_model, small_trainloader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(baseline_model, testloader, criterion, device)
        scheduler.step()
        if test_acc > baseline_best_acc:
            baseline_best_acc = test_acc
    
    # 计算提升
    improvement_over_baseline = kd_best_acc - baseline_best_acc
    improvement_over_teacher = kd_best_acc - fedavg_best_acc
    
    print(f"结果: 基线={baseline_best_acc:.2f}%, 教师={fedavg_best_acc:.2f}%, "
          f"学生(KD)={kd_best_acc:.2f}%")
    print(f"提升: 相比基线={improvement_over_baseline:+.2f}%, "
          f"相比教师={improvement_over_teacher:+.2f}%")
    
    return {
        'temperature': temperature,
        'alpha': alpha,
        'baseline_acc': baseline_best_acc,
        'teacher_acc': fedavg_best_acc,
        'student_acc': kd_best_acc,
        'improvement_over_baseline': improvement_over_baseline,
        'improvement_over_teacher': improvement_over_teacher,
        'kd_history': kd_history
    }


def visualize_results(results_df, output_dir='param_study_results'):
    """
    可视化参数研究结果
    
    Args:
        results_df: 包含实验结果的DataFrame
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 热力图：不同T和α组合的学生模型准确率
    pivot_student = results_df.pivot(index='alpha', columns='temperature', values='student_acc')
    
    plt.figure(figsize=(12, 8))
    im = plt.imshow(pivot_student.values, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Student Accuracy (%)')
    plt.xticks(range(len(pivot_student.columns)), pivot_student.columns)
    plt.yticks(range(len(pivot_student.index)), pivot_student.index)
    plt.xlabel('Temperature (T)', fontsize=12)
    plt.ylabel('Alpha (α)', fontsize=12)
    plt.title('Student Model Accuracy with Different KD Parameters', fontsize=14, fontweight='bold')
    
    # 在每个格子中标注数值
    for i in range(len(pivot_student.index)):
        for j in range(len(pivot_student.columns)):
            value = pivot_student.values[i, j]
            plt.text(j, i, f'{value:.2f}', ha='center', va='center', 
                    color='white' if value < pivot_student.values.mean() else 'black',
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_student_acc.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 热力图：相比基线的提升
    pivot_improvement_baseline = results_df.pivot(index='alpha', columns='temperature', 
                                                   values='improvement_over_baseline')
    
    plt.figure(figsize=(12, 8))
    im = plt.imshow(pivot_improvement_baseline.values, cmap='RdYlGn', aspect='auto',
                   vmin=pivot_improvement_baseline.values.min(), 
                   vmax=pivot_improvement_baseline.values.max())
    plt.colorbar(im, label='Improvement over Baseline (%)')
    plt.xticks(range(len(pivot_improvement_baseline.columns)), pivot_improvement_baseline.columns)
    plt.yticks(range(len(pivot_improvement_baseline.index)), pivot_improvement_baseline.index)
    plt.xlabel('Temperature (T)', fontsize=12)
    plt.ylabel('Alpha (α)', fontsize=12)
    plt.title('Improvement over Baseline with Different KD Parameters', 
             fontsize=14, fontweight='bold')
    
    # 标注数值
    for i in range(len(pivot_improvement_baseline.index)):
        for j in range(len(pivot_improvement_baseline.columns)):
            value = pivot_improvement_baseline.values[i, j]
            plt.text(j, i, f'{value:+.2f}', ha='center', va='center',
                    color='white' if abs(value) > abs(pivot_improvement_baseline.values).mean() else 'black',
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_improvement_baseline.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 热力图：相比教师模型的提升
    pivot_improvement_teacher = results_df.pivot(index='alpha', columns='temperature',
                                                 values='improvement_over_teacher')
    
    plt.figure(figsize=(12, 8))
    im = plt.imshow(pivot_improvement_teacher.values, cmap='RdYlGn', aspect='auto',
                   vmin=pivot_improvement_teacher.values.min(),
                   vmax=pivot_improvement_teacher.values.max())
    plt.colorbar(im, label='Improvement over Teacher (%)')
    plt.xticks(range(len(pivot_improvement_teacher.columns)), pivot_improvement_teacher.columns)
    plt.yticks(range(len(pivot_improvement_teacher.index)), pivot_improvement_teacher.index)
    plt.xlabel('Temperature (T)', fontsize=12)
    plt.ylabel('Alpha (α)', fontsize=12)
    plt.title('Improvement over Teacher Model with Different KD Parameters',
             fontsize=14, fontweight='bold')
    
    # 标注数值
    for i in range(len(pivot_improvement_teacher.index)):
        for j in range(len(pivot_improvement_teacher.columns)):
            value = pivot_improvement_teacher.values[i, j]
            plt.text(j, i, f'{value:+.2f}', ha='center', va='center',
                    color='white' if abs(value) > abs(pivot_improvement_teacher.values).mean() else 'black',
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_improvement_teacher.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 折线图：固定alpha，改变temperature的影响
    plt.figure(figsize=(14, 8))
    for alpha_val in sorted(results_df['alpha'].unique()):
        subset = results_df[results_df['alpha'] == alpha_val].sort_values('temperature')
        plt.plot(subset['temperature'], subset['student_acc'], 
                marker='o', label=f'α={alpha_val}', linewidth=2, markersize=8)
    
    plt.xlabel('Temperature (T)', fontsize=12)
    plt.ylabel('Student Accuracy (%)', fontsize=12)
    plt.title('Effect of Temperature on Student Performance (Different α)', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_effect.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 折线图：固定temperature，改变alpha的影响
    plt.figure(figsize=(14, 8))
    for temp_val in sorted(results_df['temperature'].unique()):
        subset = results_df[results_df['temperature'] == temp_val].sort_values('alpha')
        plt.plot(subset['alpha'], subset['student_acc'],
                marker='s', label=f'T={temp_val}', linewidth=2, markersize=8)
    
    plt.xlabel('Alpha (α)', fontsize=12)
    plt.ylabel('Student Accuracy (%)', fontsize=12)
    plt.title('Effect of Alpha on Student Performance (Different T)',
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alpha_effect.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. 3D曲面图（可选）
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建网格
        temperatures = sorted(results_df['temperature'].unique())
        alphas = sorted(results_df['alpha'].unique())
        T_grid, A_grid = np.meshgrid(temperatures, alphas)
        
        # 准备Z值（student_acc）
        Z = pivot_student.values
        
        # 绘制曲面
        surf = ax.plot_surface(T_grid, A_grid, Z, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('Temperature (T)', fontsize=11)
        ax.set_ylabel('Alpha (α)', fontsize=11)
        ax.set_zlabel('Student Accuracy (%)', fontsize=11)
        ax.set_title('Student Performance vs KD Parameters (3D View)',
                    fontsize=14, fontweight='bold')
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'surface_3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create 3D plot: {e}")
    
    print(f"\n所有可视化图表已保存到: {output_dir}")


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n加载CIFAR-10数据集...")
    
    # 数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 数据目录
    data_root = os.path.join(project_root, 'data')
    
    # 下载和加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test)
    
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    # 分配数据给各个节点
    large_nodes_indices, small_node_indices = split_dataset_for_nodes(
        trainset, num_large_nodes=4, large_to_small_ratio=10, seed=42
    )
    
    # 创建数据加载器
    large_trainloaders = [
        DataLoader(Subset(trainset, indices), batch_size=128, shuffle=True, num_workers=2)
        for indices in large_nodes_indices
    ]
    small_trainloader = DataLoader(
        Subset(trainset, small_node_indices), batch_size=128, shuffle=True, num_workers=2
    )
    
    # 定义参数网格
    temperatures = [1, 2, 3, 4, 5, 6, 8, 10]
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print(f"\n{'='*80}")
    print("参数研究实验设置")
    print(f"{'='*80}")
    print(f"Temperature (T) 范围: {temperatures}")
    print(f"Alpha (α) 范围: {alphas}")
    print(f"总实验次数: {len(temperatures) * len(alphas)}")
    print(f"{'='*80}")
    
    # 首先训练一个共用的教师模型
    print("\n预训练教师模型（FedAvg）...")
    teacher_model, _, teacher_acc = train_fedavg_large_nodes(
        ResNet18, large_trainloaders, testloader,
        num_rounds=10, local_epochs=5, lr=0.1, device=device, verbose=True
    )
    print(f"教师模型准确率: {teacher_acc:.2f}%")
    
    # 运行所有实验
    all_results = []
    param_combinations = list(product(temperatures, alphas))
    
    print(f"\n{'='*80}")
    print("开始参数扫描实验...")
    print(f"{'='*80}")
    
    for idx, (temp, alpha) in enumerate(param_combinations, 1):
        print(f"\n[{idx}/{len(param_combinations)}] 正在测试 T={temp}, α={alpha}")
        
        result = run_single_experiment(
            temperature=temp,
            alpha=alpha,
            large_trainloaders=large_trainloaders,
            small_trainloader=small_trainloader,
            testloader=testloader,
            device=device,
            teacher_model=teacher_model  # 使用预训练的教师模型
        )
        all_results.append(result)
        
        # 保存中间结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'kd_history'} 
                                   for r in all_results])
        results_df.to_csv(f'param_study_results/intermediate_results_{timestamp}.csv', index=False)
    
    # 转换为DataFrame
    results_summary = []
    for r in all_results:
        results_summary.append({
            'temperature': r['temperature'],
            'alpha': r['alpha'],
            'baseline_acc': r['baseline_acc'],
            'teacher_acc': r['teacher_acc'],
            'student_acc': r['student_acc'],
            'improvement_over_baseline': r['improvement_over_baseline'],
            'improvement_over_teacher': r['improvement_over_teacher']
        })
    
    results_df = pd.DataFrame(results_summary)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = 'param_study_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存CSV
    csv_path = os.path.join(output_dir, f'kd_params_study_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n结果已保存到: {csv_path}")
    
    # 保存详细JSON
    json_path = os.path.join(output_dir, f'kd_params_study_detailed_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"详细结果已保存到: {json_path}")
    
    # 生成可视化
    print("\n生成可视化图表...")
    visualize_results(results_df, output_dir)
    
    # 打印结果摘要
    print("\n" + "="*80)
    print("实验结果摘要")
    print("="*80)
    
    # 找出最佳参数组合
    best_idx = results_df['student_acc'].idxmax()
    best_result = results_df.iloc[best_idx]
    
    print(f"\n最佳参数组合:")
    print(f"  Temperature: {best_result['temperature']}")
    print(f"  Alpha: {best_result['alpha']}")
    print(f"  学生模型准确率: {best_result['student_acc']:.2f}%")
    print(f"  相比基线提升: {best_result['improvement_over_baseline']:+.2f}%")
    print(f"  相比教师提升: {best_result['improvement_over_teacher']:+.2f}%")
    
    # 统计分析
    print(f"\n统计信息:")
    print(f"  学生模型准确率范围: {results_df['student_acc'].min():.2f}% - {results_df['student_acc'].max():.2f}%")
    print(f"  平均学生模型准确率: {results_df['student_acc'].mean():.2f}%")
    print(f"  相比基线平均提升: {results_df['improvement_over_baseline'].mean():+.2f}%")
    print(f"  相比教师平均提升: {results_df['improvement_over_teacher'].mean():+.2f}%")
    
    print("\n" + "="*80)
    print("参数研究实验完成!")
    print("="*80)


if __name__ == '__main__':
    main()
