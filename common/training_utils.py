"""
通用训练工具函数
包含标准训练、联邦平均、KD训练、基线训练等
用于05和06实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from common.kd_training import train_epoch_kd, evaluate


def train_epoch_standard(model, trainloader, criterion, optimizer, device):
    """
    标准训练一个epoch
    
    Args:
        model: 模型
        trainloader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
    
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
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
        trainloaders: 训练数据加载器列表
        testloader: 测试数据加载器
        num_rounds: 通信轮次数
        local_epochs: 每轮本地训练epoch数
        lr: 学习率
        device: 设备
        verbose: 是否打印详细信息
    
    Returns:
        averaged_model: 聚合后的模型
        history: 历史记录
        best_acc: 最佳准确率
    """
    if verbose:
        print("\n" + "="*70)
        print("训练大数据节点联邦平均模型")
        print(f"通信轮次: {num_rounds}, 每轮本地训练: {local_epochs} epochs")
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
            print(f"\n通信轮次 {round_idx+1}/{num_rounds}")
        
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
        
        # 联邦平均
        averaged_model = federated_averaging_multi(models)
        
        # 用平均后的参数更新所有模型
        for model in models:
            model.load_state_dict(averaged_model.state_dict())
        
        # 评估聚合后的模型
        test_loss, test_acc = evaluate(averaged_model, testloader, criterion, device)
        
        if verbose:
            print(f"  Round {round_idx+1} - Test Acc: {test_acc:.2f}%")
        
        # 记录历史
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
        print(f"\n联邦平均训练完成! 最佳准确率: {best_acc:.2f}%")
    
    return averaged_model, history, best_acc


def train_fedavg_with_initial_model(model_class, trainloaders, testloader, initial_model,
                                     num_rounds=25, local_epochs=5, lr=0.1, 
                                     device='cuda', verbose=True):
    """
    使用初始模型继续进行联邦平均训练
    
    Args:
        model_class: 模型类
        trainloaders: 训练数据加载器列表
        testloader: 测试数据加载器
        initial_model: 初始模型
        num_rounds: 通信轮次数
        local_epochs: 每轮本地训练epoch数
        lr: 学习率
        device: 设备
        verbose: 是否打印详细信息
    
    Returns:
        averaged_model: 聚合后的模型
        history: 历史记录
        best_acc: 最佳准确率
    """
    if verbose:
        print("\n" + "="*70)
        print("使用初始模型继续FedAvg训练")
        print(f"通信轮次: {num_rounds}, 每轮本地训练: {local_epochs} epochs")
        print("="*70)
    
    num_nodes = len(trainloaders)
    total_epochs = num_rounds * local_epochs
    
    # 初始化所有节点模型为initial_model的副本
    models = [copy.deepcopy(initial_model) for _ in range(num_nodes)]
    
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
            print(f"\n通信轮次 {round_idx+1}/{num_rounds}")
        
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
        
        # 联邦平均
        averaged_model = federated_averaging_multi(models)
        
        # 用平均后的参数更新所有模型
        for model in models:
            model.load_state_dict(averaged_model.state_dict())
        
        # 评估聚合后的模型
        test_loss, test_acc = evaluate(averaged_model, testloader, criterion, device)
        
        if verbose:
            print(f"  Round {round_idx+1} - Test Acc: {test_acc:.2f}%")
        
        # 记录历史
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
        print(f"\n继续FedAvg训练完成! 最佳准确率: {best_acc:.2f}%")
    
    return averaged_model, history, best_acc


def train_small_node_with_kd(student_model, teacher_model, trainloader, testloader,
                             num_epochs=100, lr=0.1, device='cuda',
                             temperature=4.0, alpha=0.7, verbose=True):
    """
    使用知识蒸馏训练小数据节点
    
    Args:
        student_model: 学生模型
        teacher_model: 教师模型
        trainloader: 训练数据加载器
        testloader: 测试数据加载器
        num_epochs: 训练轮数
        lr: 学习率
        device: 设备
        temperature: 蒸馏温度
        alpha: 蒸馏损失权重
        verbose: 是否打印详细信息
    
    Returns:
        student_model: 训练后的学生模型
        history: 历史记录
        best_acc: 最佳准确率
    """
    if verbose:
        print("\n" + "="*70)
        print(f"小数据节点KD训练 (T={temperature}, α={alpha})")
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
        
        # 打印关键epoch信息
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} - Test Acc: {test_acc:.2f}%, Best: {best_acc:.2f}%")
    
    if verbose:
        print(f"\nKD训练完成! 最佳准确率: {best_acc:.2f}%")
    
    return student_model, history, best_acc


def train_small_node_baseline(model_class, trainloader, testloader, num_epochs=100, 
                              lr=0.1, device='cuda', verbose=True):
    """
    训练基线模型（小数据节点独立训练，不使用KD）
    
    Args:
        model_class: 模型类
        trainloader: 训练数据加载器
        testloader: 测试数据加载器
        num_epochs: 训练轮数
        lr: 学习率
        device: 设备
        verbose: 是否打印详细信息
    
    Returns:
        model: 训练后的模型
        history: 历史记录
        best_acc: 最佳准确率
    """
    if verbose:
        print("\n" + "="*70)
        print("基线模型训练（小数据节点独立训练，不使用KD）")
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
        train_loss, train_acc = train_epoch_standard(
            model, trainloader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        # 打印关键epoch信息
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} - Test Acc: {test_acc:.2f}%, Best: {best_acc:.2f}%")
    
    if verbose:
        print(f"\n基线模型训练完成! 最佳准确率: {best_acc:.2f}%")
    
    return model, history, best_acc
