"""
知识迁移实验
场景：5个节点的联邦学习
- 4个大数据节点(性能有限): 每个节点有大量数据
- 1个小数据节点(性能很好): 数据量只有大数据节点的1/10

实验流程：
1. 4个大数据节点进行联邦平均(FedAvg)训练，得到聚合模型
2. 使用聚合模型作为教师模型
3. 对小数据节点(性能好)使用知识蒸馏，提升其性能
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
plt.style.use('seaborn-v0_8-darkgrid')  # 使用更好看的样式

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from common.models import ResNet18
from common.data_loader import get_cifar10_dataloaders
from common.kd_training import train_epoch_kd, evaluate
from tqdm import tqdm
import copy


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
    # 设 小节点样本数为 x，则大节点样本数为 10x
    # 4 * 10x + x = total_samples
    # x = total_samples / 41
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
    
    for inputs, targets in tqdm(trainloader, desc="Training", leave=False):
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
                             local_epochs=5, lr=0.1, device='cuda'):
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
    
    Returns:
        聚合后的模型、历史记录、最佳准确率
    """
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
        'rounds': []  # 记录每个round的结果
    }
    
    global_epoch = 0
    averaged_model = None  # 初始化为None，避免lint警告
    
    for round_idx in range(num_rounds):
        print(f"\n{'='*70}")
        print(f"通信轮次 {round_idx+1}/{num_rounds}")
        print(f"{'='*70}")
        
        round_losses = []
        round_accs = []
        
        # 每个节点进行本地训练
        for local_epoch in range(local_epochs):
            global_epoch += 1
            print(f"\n  本地Epoch {local_epoch+1}/{local_epochs} (全局Epoch {global_epoch}/{total_epochs})")
            
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
                print(f"    节点 {i+1} - Loss: {loss:.4f}, Acc: {acc:.2f}%")
            
            round_losses.append(np.mean(epoch_losses))
            round_accs.append(np.mean(epoch_accs))
            
            # 更新学习率
            for scheduler in schedulers:
                scheduler.step()
        
        # 联邦平均（每个round结束后）
        print(f"\n  >>> 执行联邦平均聚合...")
        averaged_model = federated_averaging_multi(models)
        
        # 用平均后的参数更新所有模型
        for model in models:
            model.load_state_dict(averaged_model.state_dict())
        
        # 评估聚合后的模型
        test_loss, test_acc = evaluate(averaged_model, testloader, criterion, device)
        print(f"  >>> 聚合模型 - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # 记录本轮的历史
        history['train_loss'].extend(round_losses)
        history['train_acc'].extend(round_accs)
        history['test_loss'].extend([test_loss] * local_epochs)  # 只在round结束时测试，填充到每个epoch
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
            print(f"  ✓ 新的最佳准确率: {best_acc:.2f}%")
    
    print(f"\n4个大数据节点的联邦平均训练完成!")
    print(f"完成 {num_rounds} 个通信轮次，共 {total_epochs} 个训练epoch")
    print(f"最佳测试准确率: {best_acc:.2f}%")
    
    return averaged_model, history, best_acc


def train_fedavg_with_initial_model(model_class, trainloaders, testloader, initial_model,
                                    num_rounds=25, local_epochs=5, lr=0.1, device='cuda'):
    """
    使用初始模型（学生模型）继续进行联邦平均训练
    
    Args:
        model_class: 模型类
        trainloaders: 4个训练数据加载器的列表
        testloader: 测试数据加载器
        initial_model: 初始模型（从小数据节点训练得到的学生模型）
        num_rounds: 通信轮次数
        local_epochs: 每个节点在每轮中的本地训练epoch数
        lr: 学习率
        device: 设备
    
    Returns:
        聚合后的模型、历史记录、最佳准确率
    """
    print("\n使用知识蒸馏训练的学生模型作为初始模型")
    print(f"通信轮次: {num_rounds}, 每轮本地训练: {local_epochs} epochs")
    print(f"总训练量: {num_rounds * local_epochs} epochs")
    
    num_nodes = len(trainloaders)
    total_epochs = num_rounds * local_epochs
    
    # 初始化模型，所有节点都使用学生模型的参数
    models = [model_class().to(device) for _ in range(num_nodes)]
    for model in models:
        model.load_state_dict(initial_model.state_dict())
    
    criterion = nn.CrossEntropyLoss()
    optimizers = [optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) 
                  for model in models]
    schedulers = [optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs) 
                  for opt in optimizers]
    
    # 先评估初始模型（学生模型）的性能
    initial_loss, initial_acc = evaluate(initial_model, testloader, criterion, device)
    print(f"初始模型性能 - Loss: {initial_loss:.4f}, Acc: {initial_acc:.2f}%")
    
    best_acc = initial_acc
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'node_losses': [[] for _ in range(num_nodes)],
        'node_accs': [[] for _ in range(num_nodes)],
        'rounds': [],
        'initial_acc': initial_acc  # 记录初始准确率
    }
    
    global_epoch = 0
    averaged_model = None
    
    for round_idx in range(num_rounds):
        print(f"\n{'='*70}")
        print(f"通信轮次 {round_idx+1}/{num_rounds}")
        print(f"{'='*70}")
        
        round_losses = []
        round_accs = []
        
        # 每个节点进行本地训练
        for local_epoch in range(local_epochs):
            global_epoch += 1
            print(f"\n  本地Epoch {local_epoch+1}/{local_epochs} (全局Epoch {global_epoch}/{total_epochs})")
            
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
                print(f"    节点 {i+1} - Loss: {loss:.4f}, Acc: {acc:.2f}%")
            
            round_losses.append(np.mean(epoch_losses))
            round_accs.append(np.mean(epoch_accs))
            
            # 更新学习率
            for scheduler in schedulers:
                scheduler.step()
        
        # 联邦平均
        print(f"\n  >>> 执行联邦平均聚合...")
        averaged_model = federated_averaging_multi(models)
        
        # 用平均后的参数更新所有模型
        for model in models:
            model.load_state_dict(averaged_model.state_dict())
        
        # 评估聚合后的模型
        test_loss, test_acc = evaluate(averaged_model, testloader, criterion, device)
        print(f"  >>> 聚合模型 - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
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
            print(f"  ✓ 新的最佳准确率: {best_acc:.2f}%")
    
    improvement = best_acc - initial_acc
    print(f"\n使用学生模型继续联邦训练完成!")
    print(f"完成 {num_rounds} 个通信轮次，共 {total_epochs} 个训练epoch")
    print(f"初始准确率(学生模型): {initial_acc:.2f}%")
    print(f"最终最佳准确率: {best_acc:.2f}%")
    print(f"进一步提升: {improvement:.2f}%")
    
    return averaged_model, history, best_acc


def train_small_node_with_kd(student_model, teacher_model, trainloader, testloader,
                             num_epochs=100, lr=0.1, device='cuda',
                             temperature=4.0, alpha=0.7):
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
    
    Returns:
        训练后的模型、历史记录、最佳准确率
    """
    print("\n" + "="*70)
    print("阶段2: 使用知识蒸馏训练小数据节点(性能好)")
    print(f"Temperature: {temperature}, Alpha: {alpha}")
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
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练
        train_loss, kd_loss, ce_loss, train_acc = train_epoch_kd(
            student_model, teacher_model, trainloader, optimizer, device,
            temperature, alpha
        )
        print(f"  Train - Loss: {train_loss:.4f}, KD: {kd_loss:.4f}, "
              f"CE: {ce_loss:.4f}, Acc: {train_acc:.2f}%")
        
        # 测试
        test_loss, test_acc = evaluate(student_model, testloader, criterion, device)
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
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
            print(f"  ✓ 新的最佳准确率: {best_acc:.2f}%")
    
    print(f"\n小数据节点训练完成!")
    print(f"最佳测试准确率: {best_acc:.2f}%")
    
    return student_model, history, best_acc


def train_small_node_baseline(model_class, trainloader, testloader,
                              num_epochs=100, lr=0.1, device='cuda'):
    """
    训练小数据节点的基线模型(不使用知识蒸馏)
    
    Args:
        model_class: 模型类
        trainloader: 小数据节点的训练数据
        testloader: 测试数据
        num_epochs: 训练轮数
        lr: 学习率
        device: 设备
    
    Returns:
        训练后的模型、历史记录、最佳准确率
    """
    print("\n" + "="*70)
    print("基线: 小数据节点独立训练(不使用知识蒸馏)")
    print("="*70)
    
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练
        train_loss, train_acc = train_epoch_standard(model, trainloader, criterion, optimizer, device)
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        
        # 测试
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 更新最佳准确率
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"  ✓ 新的最佳准确率: {best_acc:.2f}%")
    
    print(f"\n基线训练完成!")
    print(f"最佳测试准确率: {best_acc:.2f}%")
    
    return model, history, best_acc


def run_knowledge_transfer_experiment(num_large_nodes=4, large_to_small_ratio=10,
                                      num_rounds=100, local_epochs=5, kd_epochs=5,
                                      lr=0.1, temperature=4.0, alpha=0.7,
                                      batch_size=128, device='cuda', seed=42):
    """
    运行循环知识迁移实验
    
    实验流程（每个round）：
    1. 4个大数据节点进行FedAvg训练（1轮通信）
    2. 小数据节点从聚合模型进行KD学习（若干epochs）
    3. 学生模型返回给4个大数据节点
    4. 重复上述过程 num_rounds 次
    
    Args:
        num_large_nodes: 大数据节点数量
        large_to_small_ratio: 大小节点数据量比例
        num_rounds: 循环迭代次数（每个round包含1次FedAvg+KD）
        local_epochs: FedAvg每轮的本地训练epoch数
        kd_epochs: 每个round中小数据节点的KD训练epoch数
        lr: 学习率
        temperature: 蒸馏温度
        alpha: 蒸馏损失权重
        batch_size: 批次大小
        device: 设备
        seed: 随机种子
    
    Returns:
        实验结果字典
    """
    total_fedavg_rounds = num_rounds
    total_fedavg_epochs = total_fedavg_rounds * local_epochs
    total_kd_epochs = num_rounds * kd_epochs
    
    print("\n" + "="*70)
    print("循环知识迁移实验")
    print(f"配置: {num_large_nodes}个大数据节点 + 1个小数据节点")
    print(f"数据比例: {large_to_small_ratio}:1")
    print(f"循环次数: {num_rounds} rounds")
    print(f"每个round: FedAvg 1轮×{local_epochs}epochs + KD {kd_epochs}epochs")
    print(f"总训练量: FedAvg {total_fedavg_epochs} epochs, KD {total_kd_epochs} epochs")
    print(f"学习率: {lr}, 蒸馏参数: T={temperature}, α={alpha}")
    print("="*70)
    
    # 加载数据
    from torchvision import datasets, transforms
    
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
    
    data_root = os.path.join(project_root, 'data')
    trainset = datasets.CIFAR10(root=data_root, train=True, download=True, 
                                transform=transform_train)
    testset = datasets.CIFAR10(root=data_root, train=False, download=True,
                               transform=transform_test)
    
    # 分配数据给各个节点
    large_nodes_indices, small_node_indices = split_dataset_for_nodes(
        trainset, num_large_nodes, large_to_small_ratio, seed
    )
    
    # 创建数据加载器
    large_trainloaders = [
        DataLoader(Subset(trainset, indices), batch_size=batch_size, 
                  shuffle=True, num_workers=2)
        for indices in large_nodes_indices
    ]
    
    small_trainloader = DataLoader(
        Subset(trainset, small_node_indices), batch_size=batch_size,
        shuffle=True, num_workers=2
    )
    
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # ============ 循环迭代训练 ============
    # 初始化模型（所有节点使用相同的初始模型）
    num_nodes = len(large_trainloaders)
    models = [ResNet18().to(device) for _ in range(num_nodes)]
    for i in range(1, num_nodes):
        models[i].load_state_dict(models[0].state_dict())
    
    student_model = ResNet18().to(device)
    student_model.load_state_dict(models[0].state_dict())
    
    criterion = nn.CrossEntropyLoss()
    
    # 记录历史
    history = {
        'test_acc_per_round': [],
        'fedavg_acc_per_round': [],
        'kd_acc_per_round': [],
        'round_details': []
    }
    
    best_acc = 0.0
    global_fedavg_epoch = 0
    global_kd_epoch = 0
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(project_root) / '05_knowledge_transfer' / 'transfer_results' / f'exp_{timestamp}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"开始循环训练: 共 {num_rounds} 轮")
    print(f"每轮包含: FedAvg训练(1轮×{local_epochs}epochs) + KD训练({kd_epochs}epochs)")
    print(f"结果将保存到: {results_dir}")
    print(f"{'='*80}\n")
    
    # 开始循环迭代
    for round in range(num_rounds):
        print(f"\n{'='*80}")
        print(f"第 {round+1}/{num_rounds} 轮")
        print(f"{'='*80}")
        
        # ========== 步骤1: FedAvg训练 ==========
        print(f"\n[Round {round+1}] 步骤1: 4个大数据节点FedAvg训练")
        
        # 为每个模型创建优化器
        optimizers = [optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) 
                      for model in models]
        
        # 本地训练
        for local_epoch in range(local_epochs):
            global_fedavg_epoch += 1
            
            for i in range(num_nodes):
                loss, acc = train_epoch_standard(
                    models[i], large_trainloaders[i], criterion, optimizers[i], device
                )
            
            if local_epoch == local_epochs - 1:  # 最后一个local epoch才打印
                print(f"  Local Epoch {local_epoch+1}/{local_epochs}, "
                      f"Global FedAvg Epoch {global_fedavg_epoch}")
        
        # 联邦平均
        averaged_model = federated_averaging_multi(models)
        for model in models:
            model.load_state_dict(averaged_model.state_dict())
        
        # 评估FedAvg后的模型
        fedavg_test_loss, fedavg_test_acc = evaluate(averaged_model, testloader, criterion, device)
        print(f"  FedAvg后测试准确率: {fedavg_test_acc:.2f}%")
        
        # ========== 步骤2: 小数据节点KD学习 ==========
        print(f"\n[Round {round+1}] 步骤2: 小数据节点从FedAvg模型进行KD学习")
        
        # 使用FedAvg模型作为教师
        teacher_model = averaged_model
        
        # 学生模型继续训练（或者可以重新初始化）
        student_optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        
        for kd_epoch in range(kd_epochs):
            global_kd_epoch += 1
            
            train_loss, kd_loss, ce_loss, train_acc = train_epoch_kd(
                student_model, teacher_model, small_trainloader, student_optimizer, device,
                temperature, alpha
            )
            
            if kd_epoch == kd_epochs - 1:  # 最后一个epoch才打印
                print(f"  KD Epoch {kd_epoch+1}/{kd_epochs}, Global KD Epoch {global_kd_epoch}, "
                      f"Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        
        # 评估学生模型
        kd_test_loss, kd_test_acc = evaluate(student_model, testloader, criterion, device)
        print(f"  KD后测试准确率: {kd_test_acc:.2f}%")
        
        # ========== 步骤3: 学生模型返回给大数据节点 ==========
        print(f"\n[Round {round+1}] 步骤3: 学生模型返回给大数据节点")
        
        # 所有大数据节点的模型更新为学生模型
        for model in models:
            model.load_state_dict(student_model.state_dict())
        
        # 评估返回后的模型
        final_test_loss, final_test_acc = evaluate(student_model, testloader, criterion, device)
        print(f"  返回后测试准确率: {final_test_acc:.2f}%")
        
        # 记录本轮历史
        history['test_acc_per_round'].append(final_test_acc)
        history['fedavg_acc_per_round'].append(fedavg_test_acc)
        history['kd_acc_per_round'].append(kd_test_acc)
        history['round_details'].append({
            'round': round + 1,
            'fedavg_acc': fedavg_test_acc,
            'kd_acc': kd_test_acc,
            'final_acc': final_test_acc,
            'fedavg_epochs': global_fedavg_epoch,
            'kd_epochs': global_kd_epoch
        })
        
        # 实时输出本轮对比结果
        print(f"\n{'─'*80}")
        print(f"第 {round+1} 轮结果对比:")
        print(f"  FedAvg准确率:  {fedavg_test_acc:6.2f}%")
        print(f"  KD准确率:      {kd_test_acc:6.2f}%")
        print(f"  最终准确率:    {final_test_acc:6.2f}%")
        print(f"  KD提升:        {kd_test_acc - fedavg_test_acc:+6.2f}%")
        
        # 更新最佳准确率
        if final_test_acc > best_acc:
            best_acc = final_test_acc
            print(f"  ✓ 新的最佳准确率: {best_acc:.2f}%")
        print(f"{'─'*80}")
        
        # 每5轮或最后一轮保存中间结果
        if (round + 1) % 5 == 0 or round == num_rounds - 1:
            intermediate_file = results_dir / f"intermediate_round_{round+1}.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'completed_rounds': round + 1,
                    'total_rounds': num_rounds,
                    'best_acc': best_acc,
                    'round_details': history['round_details'],
                    'last_round': {
                        'fedavg_acc': fedavg_test_acc,
                        'kd_acc': kd_test_acc,
                        'final_acc': final_test_acc
                    }
                }, f, indent=2, ensure_ascii=False)
            print(f"\n💾 已保存中间结果到: {intermediate_file.name}")
    
    final_model = student_model
    
    # ============ 基线: 纯FedAvg训练 ============
    print("\n" + "="*70)
    print("基线: 纯FedAvg训练（不使用知识蒸馏）")
    print("="*70)
    baseline_model, baseline_history, baseline_best_acc = train_fedavg_large_nodes(
        ResNet18, large_trainloaders, testloader,
        num_rounds=total_fedavg_rounds, local_epochs=local_epochs, lr=lr, device=device
    )
    
    # 汇总结果
    print("\n" + "="*80)
    print("循环训练完成!")
    print("="*80)
    
    print("\n开始评估基准模型...")
    
    # 基线模型评估后显示最终对比
    final_cyclic_acc = history['test_acc_per_round'][-1]
    final_kd_acc = history['kd_acc_per_round'][-1]
    
    print("\n" + "="*80)
    print("最终结果汇总")
    print("="*80)
    print(f"\n基准模型:")
    print(f"  纯FedAvg准确率:         {baseline_best_acc:.2f}%")
    print(f"\n循环训练 ({num_rounds}轮后):")
    print(f"  最终准确率:             {final_cyclic_acc:.2f}%")
    print(f"  最佳准确率:             {best_acc:.2f}%")
    print(f"\n提升效果:")
    print(f"  相比纯FedAvg:           {final_cyclic_acc - baseline_best_acc:+.2f}%")
    print(f"  最佳提升:               {best_acc - baseline_best_acc:+.2f}%")
    print("="*80)
    
    results = {
        'config': {
            'num_large_nodes': num_large_nodes,
            'large_to_small_ratio': large_to_small_ratio,
            'num_rounds': num_rounds,
            'local_epochs': local_epochs,
            'kd_epochs': kd_epochs,
            'total_fedavg_epochs': total_fedavg_epochs,
            'total_kd_epochs': total_kd_epochs,
            'lr': lr,
            'temperature': temperature,
            'alpha': alpha,
            'batch_size': batch_size,
            'seed': seed
        },
        'cyclic_training': {
            'best_acc': best_acc,
            'history': history
        },
        'baseline_fedavg': {
            'best_acc': baseline_best_acc,
            'history': baseline_history
        },
        'improvement': best_acc - baseline_best_acc
    }
    
    return results


def plot_results(results, output_dir='transfer_results', timestamp=None):
    """绘制循环迭代实验的可视化图表"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 图1: 每个round的准确率变化
    fig, ax = plt.subplots(figsize=(14, 7))
    
    rounds = range(1, len(results['cyclic_training']['history']['test_acc_per_round']) + 1)
    
    ax.plot(rounds, results['cyclic_training']['history']['test_acc_per_round'],
            'b-o', linewidth=2.5, markersize=6, label='Cyclic KD Training', alpha=0.9)
    
    # 如果基线有相同长度的数据，也绘制它
    baseline_rounds = len(results['baseline_fedavg']['history']['test_acc'])
    baseline_sample_points = np.linspace(0, baseline_rounds-1, len(rounds), dtype=int)
    baseline_sampled = [results['baseline_fedavg']['history']['test_acc'][i] for i in baseline_sample_points]
    ax.plot(rounds, baseline_sampled, 'gray', linestyle='--', linewidth=2.5, 
            marker='s', markersize=6, label='Baseline: FedAvg (No KD)', alpha=0.7)
    
    ax.set_xlabel('Round', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Cyclic Knowledge Transfer: Test Accuracy per Round', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # 添加结果文本框
    textstr = f'Final Results:\n'
    textstr += f'Cyclic KD: {results["cyclic_training"]["best_acc"]:.2f}%\n'
    textstr += f'Baseline FedAvg: {results["baseline_fedavg"]["best_acc"]:.2f}%\n'
    textstr += f'Improvement: +{results["improvement"]:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'cyclic_accuracy.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图2: 最终结果对比
    fig, ax = plt.subplots(figsize=(10, 7))
    
    models = ['Cyclic KD\nTraining', 'Baseline\nFedAvg']
    accuracies = [
        results['cyclic_training']['best_acc'],
        results['baseline_fedavg']['best_acc']
    ]
    colors_bar = ['#e74c3c', '#95a5a6']
    
    bars = ax.bar(models, accuracies, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Final Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, min(100, max(accuracies) * 1.15))
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加配置信息
    config = results['config']
    config_text = f"Config: {config['num_rounds']} rounds, "
    config_text += f"FedAvg {config['local_epochs']}e + "
    config_text += f"KD {config['kd_epochs']}e per round\n"
    config_text += f"Total: FedAvg {config['total_fedavg_epochs']}e, KD {config['total_kd_epochs']}e | "
    config_text += f"KD params: T={config['temperature']}, α={config['alpha']}"
    ax.text(0.5, 0.02, config_text, transform=ax.transAxes,
           fontsize=9, ha='center', style='italic',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'final_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results(results, output_dir='transfer_results'):
    """保存实验结果"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 为每次实验创建单独的文件夹
    exp_dir = os.path.join(output_dir, f'exp_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存JSON格式的详细结果
    json_path = os.path.join(exp_dir, 'transfer_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n详细结果已保存到: {json_path}")
    
    # 保存简要的CSV格式结果
    import csv
    csv_path = os.path.join(exp_dir, 'transfer_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Cyclic KD Training - Best Acc', f"{results['cyclic_training']['best_acc']:.2f}%"])
        writer.writerow(['Baseline FedAvg - Best Acc', f"{results['baseline_fedavg']['best_acc']:.2f}%"])
        writer.writerow(['Improvement', f"+{results['improvement']:.2f}%"])
        writer.writerow([''])
        writer.writerow(['Configuration', ''])
        for key, value in results['config'].items():
            writer.writerow([key, value])
    print(f"结果摘要已保存到: {csv_path}")
    
    # 保存详细的训练日志到txt文件
    log_path = os.path.join(output_dir, f'training_log_{timestamp}.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("循环知识迁移实验 - 详细训练日志\n")
        f.write(f"时间戳: {timestamp}\n")
        f.write("="*70 + "\n\n")
        
        # 配置信息
        f.write("配置信息:\n")
        f.write("-"*70 + "\n")
        for key, value in results['config'].items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # 循环训练详细记录
        f.write("="*70 + "\n")
        f.write("循环KD训练过程:\n")
        f.write("="*70 + "\n\n")
        for detail in results['cyclic_training']['history']['round_details']:
            f.write(f"Round {detail['round']}:\n")
            f.write(f"  FedAvg准确率: {detail['fedavg_acc']:.2f}%\n")
            f.write(f"  KD训练准确率: {detail['kd_acc']:.2f}%\n")
            f.write(f"  最终准确率: {detail['final_acc']:.2f}%\n")
            f.write(f"  累计FedAvg Epochs: {detail['fedavg_epochs']}\n")
            f.write(f"  累计KD Epochs: {detail['kd_epochs']}\n")
            f.write("\n")
        
        # 基线FedAvg训练记录
        f.write("="*70 + "\n")
        f.write("基线FedAvg训练过程:\n")
        f.write("="*70 + "\n\n")
        baseline_history = results['baseline_fedavg']['history']
        if 'rounds' in baseline_history:
            for round_info in baseline_history['rounds']:
                f.write(f"Round {round_info['round']}:\n")
                f.write(f"  测试准确率: {round_info['test_acc']:.2f}%\n")
                f.write(f"  测试损失: {round_info['test_loss']:.4f}\n")
                f.write(f"  平均训练损失: {round_info['avg_train_loss']:.4f}\n")
                f.write(f"  平均训练准确率: {round_info['avg_train_acc']:.2f}%\n")
                f.write("\n")
        
        # 最终结果汇总
        f.write("="*70 + "\n")
        f.write("最终结果汇总:\n")
        f.write("="*70 + "\n")
        f.write(f"循环KD训练最佳准确率: {results['cyclic_training']['best_acc']:.2f}%\n")
        f.write(f"基线FedAvg最佳准确率: {results['baseline_fedavg']['best_acc']:.2f}%\n")
        f.write(f"改进幅度: {results['improvement']:+.2f}%\n")
        f.write("\n")
        
        # 每轮准确率记录
        f.write("="*70 + "\n")
        f.write("每轮测试准确率:\n")
        f.write("="*70 + "\n")
        f.write("Round\tFedAvg Acc\tKD Acc\t\tFinal Acc\n")
        f.write("-"*70 + "\n")
        for i, (fedavg_acc, kd_acc, final_acc) in enumerate(zip(
            results['cyclic_training']['history']['fedavg_acc_per_round'],
            results['cyclic_training']['history']['kd_acc_per_round'],
            results['cyclic_training']['history']['test_acc_per_round']
        ), 1):
            f.write(f"{i}\t{fedavg_acc:.2f}%\t\t{kd_acc:.2f}%\t\t{final_acc:.2f}%\n")
    
    print(f"结果摘要已保存到: {csv_path}")
    
    # 保存详细的训练日志到txt文件
    log_path = os.path.join(exp_dir, 'training_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("循环知识迁移实验 - 详细训练日志\n")
        f.write(f"时间戳: {timestamp}\n")
        f.write("="*70 + "\n\n")
        
        # 配置信息
        f.write("配置信息:\n")
        f.write("-"*70 + "\n")
        for key, value in results['config'].items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # 循环训练详细记录
        f.write("="*70 + "\n")
        f.write("循环KD训练过程:\n")
        f.write("="*70 + "\n\n")
        for detail in results['cyclic_training']['history']['round_details']:
            f.write(f"Round {detail['round']}:\n")
            f.write(f"  FedAvg准确率: {detail['fedavg_acc']:.2f}%\n")
            f.write(f"  KD训练准确率: {detail['kd_acc']:.2f}%\n")
            f.write(f"  最终准确率: {detail['final_acc']:.2f}%\n")
            f.write(f"  累计FedAvg Epochs: {detail['fedavg_epochs']}\n")
            f.write(f"  累计KD Epochs: {detail['kd_epochs']}\n")
            f.write("\n")
        
        # 基线FedAvg训练记录
        f.write("="*70 + "\n")
        f.write("基线FedAvg训练过程:\n")
        f.write("="*70 + "\n\n")
        baseline_history = results['baseline_fedavg']['history']
        if 'rounds' in baseline_history:
            for round_info in baseline_history['rounds']:
                f.write(f"Round {round_info['round']}:\n")
                f.write(f"  测试准确率: {round_info['test_acc']:.2f}%\n")
                f.write(f"  测试损失: {round_info['test_loss']:.4f}\n")
                f.write(f"  平均训练损失: {round_info['avg_train_loss']:.4f}\n")
                f.write(f"  平均训练准确率: {round_info['avg_train_acc']:.2f}%\n")
                f.write("\n")
        
        # 最终结果汇总
        f.write("="*70 + "\n")
        f.write("最终结果汇总:\n")
        f.write("="*70 + "\n")
        f.write(f"循环KD训练最佳准确率: {results['cyclic_training']['best_acc']:.2f}%\n")
        f.write(f"基线FedAvg最佳准确率: {results['baseline_fedavg']['best_acc']:.2f}%\n")
        f.write(f"改进幅度: {results['improvement']:+.2f}%\n")
        f.write("\n")
        
        # 每轮准确率记录
        f.write("="*70 + "\n")
        f.write("每轮测试准确率:\n")
        f.write("="*70 + "\n")
        f.write("Round\tFedAvg Acc\tKD Acc\t\tFinal Acc\n")
        f.write("-"*70 + "\n")
        for i, (fedavg_acc, kd_acc, final_acc) in enumerate(zip(
            results['cyclic_training']['history']['fedavg_acc_per_round'],
            results['cyclic_training']['history']['kd_acc_per_round'],
            results['cyclic_training']['history']['test_acc_per_round']
        ), 1):
            f.write(f"{i}\t{fedavg_acc:.2f}%\t\t{kd_acc:.2f}%\t\t{final_acc:.2f}%\n")
    
    print(f"训练日志已保存到: {log_path}")
    
    # 生成可视化图表
    print("\n生成可视化图表...")
    plot_results(results, exp_dir, timestamp)
    
    print(f"\n所有结果已保存到文件夹: {exp_dir}")
    print(f"  - transfer_results.json (详细结果)")
    print(f"  - transfer_summary.csv (结果摘要)")
    print(f"  - training_log.txt (训练日志)")
    print(f"  - cyclic_accuracy.png (准确率曲线)")
    print(f"  - final_comparison.png (最终对比图)")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 运行实验
    results = run_knowledge_transfer_experiment(
        num_large_nodes=4,
        large_to_small_ratio=10,
        num_rounds=100,                 # 100个循环迭代
        local_epochs=20,                # 每个round的FedAvg本地训练20个epoch
        kd_epochs=20,                   # 每个round中KD训练20个epoch
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
