import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms


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
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

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
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    # 将训练集分成两部分
    train_size = len(trainset)
    split1_size = int(train_size * split_ratio)
    split2_size = train_size - split1_size
    
    trainset1, trainset2 = random_split(trainset, [split1_size, split2_size])

    trainloader1 = DataLoader(trainset1, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    trainloader2 = DataLoader(trainset2, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader1, trainloader2, testloader


if __name__ == '__main__':
    # 测试数据加载器
    trainloader, testloader = get_cifar10_dataloaders()
    print(f"训练集批次数: {len(trainloader)}")
    print(f"测试集批次数: {len(testloader)}")
    
    # 测试分割数据加载器
    trainloader1, trainloader2, testloader = get_split_cifar10_dataloaders()
    print(f"\n分割后:")
    print(f"训练集1批次数: {len(trainloader1)}")
    print(f"训练集2批次数: {len(trainloader2)}")
    print(f"测试集批次数: {len(testloader)}")
