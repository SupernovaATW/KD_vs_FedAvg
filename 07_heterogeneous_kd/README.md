# 07_heterogeneous_kd - 异构模型知识蒸馏实验

## 实验概述

本实验研究异构模型架构下的双向知识蒸馏在联邦学习场景中的效果。

## 架构设计

### 模型配置
- **大数据节点 (4个)**:
  - 节点1: ResNet8 (小模型)
  - 节点2: ResNet8 (小模型)
  - 节点3: ResNet18 (中型模型)
  - 节点4: ResNet18 (中型模型)
  
- **小数据节点 (1个)**:
  - ResNet34 (大模型)

### 训练流程

每个循环包含3个阶段:

1. **阶段1: 大节点本地训练**
   - 4个大节点各自在本地数据上独立训练
   - 不进行联邦平均 (FedAvg)
   - 训练完成后，所有模型发送给小节点

2. **阶段2: 小节点从大节点学习**
   - 小节点使用4个大节点模型的**平均logits**作为教师
   - 多教师知识蒸馏 (Multi-Teacher Knowledge Distillation)
   - 使用小节点的少量数据进行训练

3. **阶段3: 大节点从小节点学习**
   - 训练好的小节点模型发送回大节点
   - 每个大节点使用小节点模型作为教师
   - 单教师知识蒸馏 (Single-Teacher Knowledge Distillation)

## 主要特点

### 1. 异构模型架构
- 打破传统联邦学习中同构模型的限制
- 探索不同容量模型之间的知识传递
- 小节点使用更大的模型学习多个大节点的知识

### 2. 双向知识蒸馏
- **向上蒸馏**: 小节点从多个大节点学习 (知识聚合)
- **向下蒸馏**: 大节点从小节点学习 (知识回流)

### 3. 多教师集成
- 小节点同时从4个异构教师学习
- 使用平均logits作为软标签
- 充分利用不同模型的互补性

### 4. 参数搜索
支持对以下参数进行网格搜索:
- `small_temperature`: 小节点学习的温度参数
- `small_alpha`: 小节点蒸馏损失权重
- `large_temperature`: 大节点学习的温度参数
- `large_alpha`: 大节点蒸馏损失权重

## 文件结构

```
07_heterogeneous_kd/
├── config.py              # 配置和命令行参数
├── experiment_runner.py   # 实验运行器（核心训练逻辑）
├── main.py               # 主程序（参数搜索）
├── quick_test.py         # 快速测试脚本
├── visualize_results.py  # 结果可视化
├── results_utils.py      # 结果处理工具
└── hetero_results/       # 实验结果目录
    ├── hetero_results_*.json     # 完整实验结果
    ├── hetero_summary_*.csv      # 汇总结果
    ├── hetero_intermediate_*.json # 中间结果
    └── *.png                     # 可视化图表
```

## 使用方法

### 1. 快速测试

验证代码是否正常工作（使用少量epochs）:

```bash
cd 07_heterogeneous_kd
python quick_test.py
```

### 2. 运行完整实验

使用默认参数运行:

```bash
python main.py
```

自定义参数运行:

```bash
python main.py \
    --num-cycles 20 \
    --local-epochs 30 \
    --small-node-epochs 30 \
    --large-node-kd-epochs 30 \
    --small-temperatures 2.0 4.0 6.0 8.0 \
    --small-alphas 0.3 0.5 0.7 0.9 \
    --large-temperatures 2.0 4.0 6.0 8.0 \
    --large-alphas 0.3 0.5 0.7 0.9 \
    --lr 0.1 \
    --seed 42
```

### 3. 可视化结果

```bash
python visualize_results.py hetero_results/hetero_results_TIMESTAMP.json
```

这将生成:
- `param_heatmaps.png`: 参数热力图
- `training_progress.png`: 训练进度图
- `param_comparison.png`: 参数对比图
- `summary_report.txt`: 文本摘要报告

## 命令行参数

### 数据参数
- `--num-large-nodes`: 大节点数量 (默认: 4)
- `--large-to-small-ratio`: 大小节点数据量比例 (默认: 10)
- `--batch-size`: 批次大小 (默认: 128)
- `--num-workers`: 数据加载线程数 (默认: 2)

### 训练参数
- `--local-epochs`: 大节点本地训练epochs (默认: 20)
- `--small-node-epochs`: 小节点KD训练epochs (默认: 20)
- `--large-node-kd-epochs`: 大节点KD训练epochs (默认: 20)
- `--num-cycles`: 循环次数 (默认: 10)
- `--lr`: 学习率 (默认: 0.1)

### KD参数
- `--small-temperatures`: 小节点Temperature列表 (默认: [2.0, 4.0, 6.0])
- `--small-alphas`: 小节点Alpha列表 (默认: [0.5, 0.7, 0.9])
- `--large-temperatures`: 大节点Temperature列表 (默认: [2.0, 4.0, 6.0])
- `--large-alphas`: 大节点Alpha列表 (默认: [0.5, 0.7, 0.9])

### 其他参数
- `--seed`: 随机种子 (默认: 42)
- `--output-dir`: 输出目录 (默认: hetero_results)
- `--no-visualize`: 跳过可视化生成

## 实验输出

### JSON结果文件
包含完整的实验配置和每个循环的详细结果:
- 配置信息
- 每个循环的模型准确率
- 训练统计信息

### CSV汇总文件
便于快速查看和比较不同参数组合的结果。

### 可视化图表
- 参数热力图: 显示不同参数组合的效果
- 训练进度图: 显示最佳参数下的训练过程
- 参数对比图: 单独分析Temperature和Alpha的影响

## 技术细节

### 多教师知识蒸馏
```python
# 计算所有教师的平均logits
avg_teacher_logits = mean([teacher1(x), teacher2(x), teacher3(x), teacher4(x)])

# 蒸馏损失
soft_student = log_softmax(student(x) / T)
soft_teacher = softmax(avg_teacher_logits / T)
kd_loss = KL_divergence(soft_student, soft_teacher) * T^2

# 总损失
total_loss = α * kd_loss + (1-α) * ce_loss
```

### 模型参数量对比
- ResNet8: ~1.1M 参数
- ResNet18: ~11M 参数  
- ResNet34: ~21M 参数

## 依赖项

所有依赖项已在项目根目录的 `requirements.txt` 中定义。

主要依赖:
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- tqdm

## 相关实验

- `05_knowledge_transfer`: 同构模型的知识迁移循环
- `06_kd_params_study`: 同构模型的KD参数研究

## 注意事项

1. **GPU内存**: 实验同时加载多个模型，需要较大的GPU内存
2. **训练时间**: 参数搜索会运行多个实验组合，需要较长时间
3. **中间结果**: 程序会定期保存中间结果，防止数据丢失
4. **日志输出**: 训练过程会显示详细的中间信息，便于监控进度

## 预期结果

- 异构模型能够有效进行知识传递
- 多教师蒸馏优于单教师蒸馏
- 双向知识流动提升所有节点的性能
- 最佳KD参数可能因模型大小不同而异
