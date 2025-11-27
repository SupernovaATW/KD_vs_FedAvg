# CIFAR-10 + ResNet18: 联邦平均 vs 知识蒸馏对比实验

本项目对比了在CIFAR-10数据集上使用ResNet18模型的两种训练策略：

1. **联邦平均 (FedAvg)**: 同时训练两个模型，每轮训练后将参数平均
2. **知识蒸馏 (Knowledge Distillation)**: 先训练一个教师模型，再用它指导学生模型训练

## 📋 项目结构

```
KD_vs_FedAvg/
├── models.py              # ResNet18模型定义
├── data_loader.py         # CIFAR-10数据加载器
├── fedavg_training.py     # 联邦平均训练实现
├── kd_training.py         # 知识蒸馏训练实现
├── main.py                # 主实验脚本
├── requirements.txt       # 依赖包列表
└── README.md              # 项目说明
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行完整对比实验

```bash
python main.py --epochs 100 --batch_size 128 --lr 0.1
```

### 3. 仅运行联邦平均

```bash
python main.py --run_fedavg --epochs 100
```

### 4. 仅运行知识蒸馏

```bash
python main.py --run_kd --epochs 100
```

## 📊 实验参数说明

### 基础参数
- `--epochs`: 训练轮数（默认100）
- `--batch_size`: 批次大小（默认128）
- `--lr`: 初始学习率（默认0.1）
- `--device`: 训练设备，cuda或cpu（默认cuda）
- `--num_workers`: 数据加载线程数（默认2）

### 知识蒸馏参数
- `--temperature`: 温度参数，用于软化输出分布（默认4.0）
- `--alpha`: 蒸馏损失权重，0-1之间（默认0.7）

### 运行选项
- `--run_fedavg`: 运行联邦平均实验
- `--run_kd`: 运行知识蒸馏实验
- `--no_plot`: 不生成对比图
- `--save_results`: 保存实验结果到JSON

### 输出路径
- `--plot_path`: 对比图保存路径（默认comparison_plot.png）
- `--results_path`: 结果JSON保存路径（默认experiment_results.json）

## 🔬 方法详解

### 联邦平均 (FedAvg)

**流程**:
1. 初始化两个相同的ResNet18模型
2. 将训练数据分成两部分（模拟两个客户端）
3. 每轮训练：
   - 两个模型分别在各自数据上训练
   - 训练完后将两个模型的参数平均
   - 用平均后的参数更新两个模型
4. 重复直到收敛

**特点**:
- 模拟联邦学习场景，数据不集中
- 两个模型协同学习
- 参数平均保证模型一致性

### 知识蒸馏 (Knowledge Distillation)

**流程**:
1. 阶段1 - 训练教师模型：
   - 在完整训练集上训练ResNet18
   - 获得性能优秀的教师模型
2. 阶段2 - 训练学生模型：
   - 初始化新的ResNet18作为学生
   - 使用蒸馏损失训练：
     - 软标签损失（KL散度，来自教师）
     - 硬标签损失（交叉熵，来自真实标签）
   - 总损失 = α × 软标签损失 + (1-α) × 硬标签损失

**特点**:
- 教师模型传授"知识"给学生
- 学生学习教师的输出分布，不仅是最终预测
- 温度参数软化概率分布，让学生学到更多信息

## 📈 实验输出

### 训练过程输出
- 每轮的训练/测试损失和准确率
- 最佳模型自动保存：
  - `best_fedavg_model.pth`: 联邦平均最佳模型
  - `best_teacher_model.pth`: 教师模型
  - `best_student_model.pth`: 学生模型

### 对比可视化
自动生成 `comparison_plot.png`，包含4个子图：
1. 训练准确率对比
2. 测试准确率对比
3. 训练损失对比
4. 测试损失对比

### 结果JSON
保存到 `experiment_results.json`，包含：
- 实验配置参数
- 两种方法的最佳准确率
- 完整训练历史（损失、准确率）

## 💡 使用示例

### 快速测试（10轮）
```bash
python main.py --epochs 10 --batch_size 128
```

### 完整训练（100轮，使用GPU）
```bash
python main.py --epochs 100 --batch_size 128 --device cuda
```

### 调整知识蒸馏参数
```bash
python main.py --run_kd --temperature 3.0 --alpha 0.5 --epochs 100
```

### CPU训练（较慢）
```bash
python main.py --epochs 20 --device cpu --batch_size 64
```

## 🔍 预期结果

在CIFAR-10上使用ResNet18，典型结果：
- **联邦平均**: 测试准确率约 90-93%
- **知识蒸馏**: 测试准确率约 91-94%

具体结果取决于：
- 训练轮数
- 学习率调度
- 数据增强
- 知识蒸馏的温度和alpha参数

## 📚 参考文献

1. **ResNet**: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
2. **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017
3. **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network", NIPS 2014 Workshop

## ⚙️ 系统要求

- Python 3.7+
- PyTorch 2.0+
- CUDA（可选，用于GPU加速）
- 至少8GB RAM（CPU训练）或4GB VRAM（GPU训练）

## 🐛 常见问题

**Q: CUDA out of memory错误？**
A: 减小batch_size，例如 `--batch_size 64`

**Q: 训练很慢？**
A: 确保使用GPU (`--device cuda`)，增加num_workers (`--num_workers 4`)

**Q: 如何只运行其中一种方法？**
A: 使用 `--run_fedavg` 或 `--run_kd` 单独指定

## 📝 License

MIT License
