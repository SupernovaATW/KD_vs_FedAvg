"""
通用数据处理工具函数
用于05和06实验
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms


def split_dataset_for_nodes(trainset, num_large_nodes=4, large_to_small_ratio=10, seed=42):
    """
    将训练集分配给5个节点：4个大数据节点 + 1个小数据节点
    
    Args:
        trainset: 训练数据集
        num_large_nodes: 大数据节点数量
        large_to_small_ratio: 大数据节点与小数据节点的数据量比例
        seed: 随机种子
    
    Returns:
        large_nodes_indices: 大数据节点的数据索引列表
        small_node_indices: 小数据节点的数据索引
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
    
    # 小数据节点
    small_node_indices = all_indices[start_idx:start_idx + small_node_size].tolist()
    
    print(f"\n验证:")
    for i in range(num_large_nodes):
        print(f"大数据节点{i+1}: {len(large_nodes_indices[i])} 样本")
    print(f"小数据节点: {len(small_node_indices)} 样本")
    
    return large_nodes_indices, small_node_indices


def create_dataloaders(trainset, large_nodes_indices, small_node_indices, 
                      batch_size=128, num_workers=2):
    """
    为各个节点创建数据加载器
    
    Args:
        trainset: 训练数据集
        large_nodes_indices: 大数据节点的索引列表
        small_node_indices: 小数据节点的索引
        batch_size: 批次大小
        num_workers: 工作线程数
    
    Returns:
        large_trainloaders: 大数据节点的数据加载器列表
        small_trainloader: 小数据节点的数据加载器
    """
    # 为大数据节点创建数据加载器
    large_trainloaders = []
    for i, indices in enumerate(large_nodes_indices):
        subset = Subset(trainset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, 
                          num_workers=num_workers, pin_memory=True)
        large_trainloaders.append(loader)
    
    # 为小数据节点创建数据加载器
    small_subset = Subset(trainset, small_node_indices)
    small_trainloader = DataLoader(small_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
    
    return large_trainloaders, small_trainloader


def load_cifar10_data(data_root='./data', batch_size=128, num_workers=2):
    """
    加载CIFAR-10数据集
    
    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 工作线程数
    
    Returns:
        trainset: 训练集
        testset: 测试集
        testloader: 测试数据加载器
    """
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
    
    # 加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )
    
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    
    print(f"训练集大小: {len(trainset)}")
    print(f"测试集大小: {len(testset)}")
    
    return trainset, testset, testloader


def split_dataset_by_ratio(num_large_nodes=4, large_to_small_ratio=10,
                          batch_size=128, num_workers=2, data_root='./data'):
    """
    一站式函数：加载数据集并分配给各个节点
    
    Args:
        num_large_nodes: 大数据节点数量
        large_to_small_ratio: 大小节点数据量比例
        batch_size: 批次大小
        num_workers: 工作线程数
        data_root: 数据根目录
    
    Returns:
        large_trainloaders: 大数据节点的数据加载器列表
        small_trainloader: 小数据节点的数据加载器
        testloader: 测试数据加载器
    """
    # 加载数据集
    trainset, testset, testloader = load_cifar10_data(data_root, batch_size, num_workers)
    
    # 分配数据
    large_nodes_indices, small_node_indices = split_dataset_for_nodes(
        trainset, num_large_nodes, large_to_small_ratio
    )
    
    # 创建数据加载器
    large_trainloaders, small_trainloader = create_dataloaders(
        trainset, large_nodes_indices, small_node_indices, batch_size, num_workers
    )
    
    return large_trainloaders, small_trainloader, testloader
