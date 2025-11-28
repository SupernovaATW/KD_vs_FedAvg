import os
import torch
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')


def get_cifar10_dataloaders(batch_size=128, num_workers=2):
    """
    获取CIFAR-10数据集的训练和测试DataLoader
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

    # 下载和加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def get_split_cifar10_dataloaders(batch_size=128, num_workers=2, split_ratio=0.5):
    """
    将CIFAR-10训练集分成两部分，用于模拟两个独立的数据源
    这样可以更真实地模拟联邦学习场景
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

    # 下载和加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform_test)

    # 将训练集分成两部分
    train_size = len(trainset)
    split1_size = int(train_size * split_ratio)
    split2_size = train_size - split1_size
    
    trainset1, trainset2 = random_split(trainset, [split1_size, split2_size])

    trainloader1 = DataLoader(trainset1, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    trainloader2 = DataLoader(trainset2, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader1, trainloader2, testloader


def get_noniid_cifar10_dataloaders(batch_size=128, num_workers=2, alpha=0.5, num_clients=2,
                                   seed=7000, visualize=False, save_dir='logs',
                                   return_distribution=False, force_equal_size=True):
    """
    使用Dirichlet分布生成Non-IID的CIFAR-10数据集
    
    Args:
        batch_size: 批次大小
        num_workers: 数据加载器的工作进程数
        alpha: Dirichlet分布的浓度参数，越小数据越不均衡（Non-IID程度越高）
               - alpha -> 0: 高度Non-IID（每个客户端只有少数类别）
               - alpha -> ∞: 接近IID（均匀分布）
               - 常用值: 0.1, 0.5, 1.0
        num_clients: 客户端数量
        seed: 随机种子
        visualize: 是否打印详细的类别分布表
        save_dir: 兼容参数（不再使用）
    
    Returns:
        训练数据加载器列表和测试数据加载器
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

    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 下载和加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform_test)

    targets = np.array(trainset.targets)
    num_classes = 10
    client_indices = [[] for _ in range(num_clients)]
    target_counts = None
    remaining_capacity = None
    if force_equal_size:
        total_samples = len(trainset)
        base = total_samples // num_clients
        remainder = total_samples % num_clients
        target_counts = [base + (1 if i < remainder else 0) for i in range(num_clients)]
        remaining_capacity = np.array(target_counts, dtype=int)
    
    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        if len(idx_k) == 0:
            continue
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        if force_equal_size:
            assert remaining_capacity is not None
            desired = proportions * len(idx_k)
            base_alloc = np.minimum(np.floor(desired).astype(int), remaining_capacity)
            assigned = base_alloc.astype(int)
            leftover = len(idx_k) - assigned.sum()
            fractions = desired - np.floor(desired)
            capacity_after_base = remaining_capacity - assigned
            if leftover > 0:
                for client in np.argsort(-fractions):
                    if leftover == 0:
                        break
                    available = capacity_after_base[client]
                    if available <= 0:
                        continue
                    take = min(available, leftover)
                    assigned[client] += take
                    capacity_after_base[client] -= take
                    leftover -= take
            if leftover > 0:
                for client in range(num_clients):
                    if leftover == 0:
                        break
                    available = capacity_after_base[client]
                    if available <= 0:
                        continue
                    take = min(available, leftover)
                    assigned[client] += take
                    capacity_after_base[client] -= take
                    leftover -= take
            if leftover > 0:
                raise RuntimeError("无法在保持等量数据的情况下分配Non-IID样本，请检查参数。")
            start = 0
            for client, count in enumerate(assigned):
                if count <= 0:
                    continue
                selected = idx_k[start:start + count]
                client_indices[client].extend(selected.tolist())
                remaining_capacity[client] -= count
                start += count
        else:
            splits = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            client_idx_k = np.split(idx_k, splits)
            for i in range(num_clients):
                client_indices[i].extend(client_idx_k[i].tolist())
    
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    
    # 收集数据分布信息
    class_distributions = []
    for i in range(num_clients):
        client_targets = targets[client_indices[i]]
        unique, counts = np.unique(client_targets, return_counts=True)
        
        # 为可视化准备完整的类别分布
        full_dist = np.zeros(num_classes)
        for cls, count in zip(unique, counts):
            full_dist[cls] = count
        class_distributions.append(full_dist)
    
    distribution_report = None
    need_report = visualize or return_distribution
    if need_report:
        distribution_report = format_data_distribution_report(class_distributions, alpha)
        if visualize:
            print("\n" + distribution_report)
    else:
        print("\n=== Non-IID数据分布 (Dirichlet alpha={}) ===".format(alpha))
        for i in range(num_clients):
            print(f"客户端 {i+1}: 样本数={int(sum(class_distributions[i]))}")
        print()
    
    # 创建数据加载器
    trainloaders = []
    for i in range(num_clients):
        subset = Subset(trainset, client_indices[i])
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        trainloaders.append(loader)
    
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    if num_clients == 2:
        outputs = (trainloaders[0], trainloaders[1], testloader)
    else:
        outputs = (trainloaders, testloader)
    
    if return_distribution:
        return outputs + (distribution_report,)
    return outputs


def format_data_distribution_report(class_distributions, alpha):
    """格式化Non-IID数据分布表格文本"""
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    num_clients = len(class_distributions)
    lines = []
    lines.append("=" * 100)
    lines.append(f"Non-IID Data Distribution (Dirichlet α={alpha})")
    lines.append("=" * 100)
    lines.append("")
    header = "Client | " + " | ".join([f"Class{i}" for i in range(10)]) + " | Total"
    lines.append(header)
    lines.append("-" * 100)
    for i, dist in enumerate(class_distributions):
        row = f"  {i+1:2d}   | "
        row += " | ".join([f"{int(dist[j]):6d}" for j in range(10)])
        row += f" | {int(sum(dist)):6d}"
        lines.append(row)
    lines.append("-" * 100)
    total_per_class = np.sum(class_distributions, axis=0)
    row = " Total | "
    row += " | ".join([f"{int(total_per_class[j]):6d}" for j in range(10)])
    row += f" | {int(np.sum(total_per_class)):6d}"
    lines.append(row)
    lines.append("=" * 100)
    lines.append("")
    lines.append("Class Names:")
    for i, name in enumerate(class_names):
        lines.append(f"  Class {i}: {name}")
    lines.append("")
    lines.append("Data Distribution Percentage:")
    lines.append("-" * 100)
    header = "Client | " + " | ".join([f"Class{i}" for i in range(10)])
    lines.append(header)
    lines.append("-" * 100)
    for i, dist in enumerate(class_distributions):
        total = sum(dist)
        row = f"  {i+1:2d}   | "
        row += " | ".join([f"{(dist[j] / total * 100) if total else 0:5.1f}%" for j in range(10)])
        lines.append(row)
    lines.append("=" * 100)
    return "\n".join(lines)



