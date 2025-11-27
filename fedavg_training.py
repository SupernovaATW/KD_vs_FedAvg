import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm


def train_epoch(model, trainloader, criterion, optimizer, device):
    """训练一个epoch"""
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


def evaluate(model, testloader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(testloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def federated_averaging(model1, model2):
    """
    联邦平均：将两个模型的参数平均
    """
    averaged_model = copy.deepcopy(model1)
    
    with torch.no_grad():
        for param_avg, param1, param2 in zip(
            averaged_model.parameters(),
            model1.parameters(),
            model2.parameters()
        ):
            param_avg.data = (param1.data + param2.data) / 2.0
    
    return averaged_model


def train_fedavg(model_class, trainloader1, trainloader2, testloader, 
                 num_epochs=100, lr=0.1, device='cuda'):
    """
    联邦平均训练方法：
    1. 初始化两个相同的模型
    2. 每轮在各自的数据上训练
    3. 将两个模型的参数平均
    4. 用平均后的参数更新两个模型
    """
    print("\n=== 开始联邦平均训练 ===")
    
    # 初始化两个模型
    model1 = model_class().to(device)
    model2 = model_class().to(device)
    
    # 确保两个模型初始参数相同
    model2.load_state_dict(model1.state_dict())
    
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(model1.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(model2.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=num_epochs)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=num_epochs)
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 在各自的数据上训练
        loss1, acc1 = train_epoch(model1, trainloader1, criterion, optimizer1, device)
        loss2, acc2 = train_epoch(model2, trainloader2, criterion, optimizer2, device)
        
        print(f"Model 1 - Loss: {loss1:.4f}, Acc: {acc1:.2f}%")
        print(f"Model 2 - Loss: {loss2:.4f}, Acc: {acc2:.2f}%")
        
        # 联邦平均
        averaged_model = federated_averaging(model1, model2)
        
        # 用平均后的参数更新两个模型
        model1.load_state_dict(averaged_model.state_dict())
        model2.load_state_dict(averaged_model.state_dict())
        
        # 更新学习率
        scheduler1.step()
        scheduler2.step()
        
        # 评估平均后的模型
        test_loss, test_acc = evaluate(averaged_model, testloader, criterion, device)
        print(f"Averaged Model - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # 记录历史
        history['train_loss'].append((loss1 + loss2) / 2)
        history['train_acc'].append((acc1 + acc2) / 2)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(averaged_model.state_dict(), 'best_fedavg_model.pth')
            print(f"✓ 保存最佳模型 (Acc: {best_acc:.2f}%)")
    
    print(f"\n联邦平均训练完成! 最佳测试准确率: {best_acc:.2f}%")
    
    return averaged_model, history, best_acc


if __name__ == '__main__':
    from models import ResNet18
    from data_loader import get_split_cifar10_dataloaders
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader1, trainloader2, testloader = get_split_cifar10_dataloaders(batch_size=128)
    
    model, history, best_acc = train_fedavg(
        ResNet18, trainloader1, trainloader2, testloader, 
        num_epochs=10, lr=0.1, device=device
    )
