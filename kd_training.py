import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


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


def train_epoch_kd(student_model, teacher_model, trainloader, optimizer, device, 
                   temperature=4.0, alpha=0.7):
    """
    使用知识蒸馏训练学生模型一个epoch
    
    Args:
        student_model: 学生模型
        teacher_model: 教师模型（已训练好）
        trainloader: 训练数据加载器
        optimizer: 优化器
        device: 设备
        temperature: 温度参数，用于软化输出分布
        alpha: 蒸馏损失的权重（1-alpha为硬标签损失的权重）
    """
    student_model.train()
    teacher_model.eval()
    
    running_loss = 0.0
    running_kd_loss = 0.0
    running_ce_loss = 0.0
    correct = 0
    total = 0
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction='batchmean')
    
    for inputs, targets in tqdm(trainloader, desc="KD Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # 学生模型的输出
        student_outputs = student_model(inputs)
        
        # 教师模型的输出（不需要梯度）
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
        
        # 硬标签损失（标准交叉熵）
        ce_loss = criterion_ce(student_outputs, targets)
        
        # 蒸馏损失（KL散度）
        # 使用温度参数软化输出分布
        soft_student = F.log_softmax(student_outputs / temperature, dim=1)
        soft_teacher = F.softmax(teacher_outputs / temperature, dim=1)
        kd_loss = criterion_kd(soft_student, soft_teacher) * (temperature ** 2)
        
        # 总损失：alpha * 蒸馏损失 + (1-alpha) * 硬标签损失
        loss = alpha * kd_loss + (1 - alpha) * ce_loss
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_kd_loss += kd_loss.item()
        running_ce_loss += ce_loss.item()
        
        _, predicted = student_outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(trainloader)
    avg_kd_loss = running_kd_loss / len(trainloader)
    avg_ce_loss = running_ce_loss / len(trainloader)
    accuracy = 100. * correct / total
    
    return avg_loss, avg_kd_loss, avg_ce_loss, accuracy


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


def train_teacher(model_class, trainloader, testloader, num_epochs=100, 
                  lr=0.1, device='cuda'):
    """
    训练教师模型
    """
    print("\n=== 训练教师模型 ===")
    
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练
        train_loss, train_acc = train_epoch_standard(model, trainloader, criterion, optimizer, device)
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        
        # 测试
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_teacher_model.pth')
            print(f"✓ 保存最佳教师模型 (Acc: {best_acc:.2f}%)")
    
    print(f"\n教师模型训练完成! 最佳测试准确率: {best_acc:.2f}%")
    
    return model, history, best_acc


def train_student_with_kd(student_class, teacher_model, trainloader, testloader, 
                          num_epochs=100, lr=0.1, device='cuda', 
                          temperature=4.0, alpha=0.7):
    """
    使用知识蒸馏训练学生模型
    """
    print("\n=== 使用知识蒸馏训练学生模型 ===")
    print(f"Temperature: {temperature}, Alpha: {alpha}")
    
    student_model = student_class().to(device)
    optimizer = optim.SGD(student_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_kd_loss': [], 'train_ce_loss': [],
        'train_acc': [], 'test_loss': [], 'test_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练
        train_loss, kd_loss, ce_loss, train_acc = train_epoch_kd(
            student_model, teacher_model, trainloader, optimizer, device, 
            temperature, alpha
        )
        print(f"Train - Loss: {train_loss:.4f}, KD: {kd_loss:.4f}, CE: {ce_loss:.4f}, Acc: {train_acc:.2f}%")
        
        # 测试
        test_loss, test_acc = evaluate(student_model, testloader, criterion, device)
        print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_kd_loss'].append(kd_loss)
        history['train_ce_loss'].append(ce_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student_model.state_dict(), 'best_student_model.pth')
            print(f"✓ 保存最佳学生模型 (Acc: {best_acc:.2f}%)")
    
    print(f"\n学生模型训练完成! 最佳测试准确率: {best_acc:.2f}%")
    
    return student_model, history, best_acc


def train_kd_pipeline(model_class, trainloader_teacher, trainloader_student, testloader, 
                     teacher_epochs=100, student_epochs=100,
                     lr=0.1, device='cuda', temperature=4.0, alpha=0.7):
    """
    完整的知识蒸馏流程：
    1. 先用教师数据集训练教师模型
    2. 再用学生数据集和教师模型指导学生模型训练
    
    Args:
        trainloader_teacher: 教师模型的训练数据（例如前50%数据）
        trainloader_student: 学生模型的训练数据（例如后50%数据）
    """
    print("\n" + "="*60)
    print("知识蒸馏训练流程（公平对比：教师和学生使用不同的数据集）")
    print("="*60)
    
    # 阶段1: 训练教师模型
    teacher_model, teacher_history, teacher_best_acc = train_teacher(
        model_class, trainloader_teacher, testloader, 
        num_epochs=teacher_epochs, lr=lr, device=device
    )
    
    # 阶段2: 训练学生模型
    student_model, student_history, student_best_acc = train_student_with_kd(
        model_class, teacher_model, trainloader_student, testloader,
        num_epochs=student_epochs, lr=lr, device=device,
        temperature=temperature, alpha=alpha
    )
    
    print("\n" + "="*60)
    print("知识蒸馏训练流程完成!")
    print(f"教师模型最佳准确率: {teacher_best_acc:.2f}%")
    print(f"学生模型最佳准确率: {student_best_acc:.2f}%")
    print("="*60)
    
    combined_history = {
        'teacher': teacher_history,
        'student': student_history
    }
    
    return teacher_model, student_model, combined_history, student_best_acc


if __name__ == '__main__':
    from models import ResNet18
    from data_loader import get_split_cifar10_dataloaders
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader_teacher, trainloader_student, testloader = get_split_cifar10_dataloaders(batch_size=128)
    
    teacher_model, student_model, history, best_acc = train_kd_pipeline(
        ResNet18, trainloader_teacher, trainloader_student, testloader,
        teacher_epochs=10, student_epochs=10,
        lr=0.1, device=device, temperature=4.0, alpha=0.7
    )
