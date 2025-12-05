## 项目概览

本仓库围绕 **CIFAR-10 + ResNet18** 构建了四个可独立运行的小项目，用于研究联邦平均 (FedAvg) 与知识蒸馏 (Knowledge Distillation) 在 IID/Non-IID 设置下的表现。

```
KD_vs_FedAvg/
├── common/                       # 共享模块（模型、数据、训练例程）
│   ├── __init__.py
│   ├── data_loader.py           # CIFAR-10加载、IID/Non-IID划分
│   ├── fedavg_training.py       # FedAvg训练流程
│   ├── kd_training.py           # 知识蒸馏训练流程
│   ├── models.py                # ResNet18模型定义
│   ├── data_utils.py            # 通用数据处理工具
│   ├── training_utils.py        # 通用训练工具函数
│   └── experiment_utils.py      # 通用实验运行工具
├── 01_iid_comparison/
│   ├── main.py                  # IID场景：FedAvg vs KD 对比
│   └── iid_comparison_results/  # 结果留存目录
├── 02_iid_kd_tune/
│   ├── tune_kd_params.py        # IID场景的KD参数调优
│   └── kd_iid_tuning_results/
├── 03_non_iid_kd_tune/
│   ├── tune_kd_params_noniid.py # Non-IID场景的KD参数调优
│   └── kd_non_iid_tuning_results/
├── 04_non_iid_comparison/
│   ├── compare_noniid.py        # Non-IID场景：FedAvg vs KD 对比
│   └── comparison_outputs/
├── 05_knowledge_transfer/       # 循环知识迁移实验
│   ├── main.py                  # 主程序入口
│   ├── config.py                # 配置参数
│   ├── experiment_runner.py     # 实验运行逻辑
│   ├── results_utils.py         # 结果可视化
│   └── transfer_results/        # 实验结果
├── 06_kd_params_study/          # KD参数研究实验（循环迭代版）
│   ├── main.py                  # 主程序入口
│   ├── config.py                # 配置参数
│   ├── experiment_runner.py     # 实验运行逻辑
│   ├── visualize_results.py     # 结果可视化
│   └── param_study_results/     # 实验结果
├── data/                        # CIFAR-10数据集
├── logs/                        # 日志与可视化文件夹
├── requirements.txt
└── README.md
```

> ✅ 运行任意子项目前，请在仓库根目录执行命令（保证 `common/` 可以被 Python 找到）。

## 环境准备

```bash
pip install -r requirements.txt
```

## 共享模块简介 (`common/`)

| 文件 | 功能 |
| --- | --- |
| `models.py` | ResNet18 定义（可扩展为其它网络） |
| `data_loader.py` | CIFAR-10 加载、IID拆分、Dirichlet Non-IID划分 |
| `fedavg_training.py` | FedAvg 训练/评估流程 |
| `kd_training.py` | 教师-学生蒸馏训练流水线 |
| `data_utils.py` | 通用数据处理工具（节点数据分配、DataLoader创建） |
| `training_utils.py` | 通用训练工具（标准训练、联邦平均、KD训练、基线训练） |
| `experiment_utils.py` | 通用实验运行工具（循环知识迁移实验流程） |

所有子项目仅需从 `common` 导入，不再复制粘贴公共逻辑。

## 六个子项目

### 01_iid_comparison

- **脚本**：`python 01_iid_comparison/main.py [args...]`
- **用途**：在 IID 数据上直接对比 FedAvg 与 KD。
- **常用参数**：
   - `--epochs`、`--batch_size`、`--lr`
   - `--device cuda|cpu`
   - `--run_fedavg` / `--run_kd` 控制单独运行
- **输出**：
   - 训练日志 → `logs/experiment_*.log`
   - JSON结果 → `01_iid_comparison/iid_comparison_results/`（可自定义）
   - 可选的对比图（PNG）

### 02_iid_kd_tune

- **脚本**：`python 02_iid_kd_tune/tune_kd_params.py [args...]`
- **用途**：网格遍历 Temperature × Alpha，在 IID 下寻找最优 KD 组合。
- **亮点**：
   - 每次实验记录教师/学生表现与提升幅度
   - 结果写入 `logs/kd_tuning_*.(csv|json)`
- **示例**：
   ```bash
   python 02_iid_kd_tune/tune_kd_params.py \
         --epochs 50 --temperatures 3 4 5 --alphas 0.5 0.7 0.9
   ```

### 03_non_iid_kd_tune

- **脚本**：`python 03_non_iid_kd_tune/tune_kd_params_noniid.py [args...]`
- **用途**：在 Dirichlet Non-IID 拆分下调参。
- **新增能力**：
   - `--dirichlet_alphas 1.0 0.5 0.1`（默认）一次性跑多种异质程度
   - 每个 α 的 Non-IID 数据分布会以表格形式写入 `logs/data_distribution_alpha*_*.txt`
- **输出**：统一落在 `logs/`（CSV、JSON、日志）。

### 04_non_iid_comparison

- **脚本**：`python 04_non_iid_comparison/compare_noniid.py [args...]`
- **用途**：在多组 Dirichlet α × KD α 上，对比 FedAvg 与 KD 的性能差异。
- **行为说明**：
   - 外层遍历 `--dirichlet_alphas`（默认 `[1.0, 0.5, 0.1]`）
   - 内层遍历 `--alphas`（默认 `[1.0, 0.5, 0.1]`）
   - 每个 Dirichlet α 都会生成独立的对比柱状图与训练曲线，文件名自带标记，例如 `comparison_noniid_alpha0p5_*.png`
   - 结果 CSV/JSON 会汇总所有组合，可在同一个文件里筛选。
- **示例**：
   ```bash
   python 04_non_iid_comparison/compare_noniid.py \
         --epochs 80 --temperature 4 --alphas 1.0 0.5 0.1 \
         --dirichlet_alphas 1.0 0.5 0.1
   ```

### 05_knowledge_transfer

- **脚本**：`python 05_knowledge_transfer/main.py [args...]`
- **用途**：循环知识迁移实验，研究大数据节点和小数据节点之间的知识互相促进。
- **实验流程**：
   1. 阶段1: 4个大数据节点进行初始FedAvg训练
   2. 阶段2: 小数据节点从聚合模型进行KD学习
   3. 阶段3: 使用学生模型继续FedAvg训练
   4. 重复阶段2和3，形成知识迁移循环
- **核心参数**：
   - `--num-cycles`: 知识迁移循环次数（默认5）
   - `--initial-fedavg-rounds`: 初始FedAvg通信轮次（默认50）
   - `--kd-rounds`: KD后的FedAvg通信轮次（默认25）
   - `--small-node-epochs`: 小数据节点KD训练epoch数（默认100）
   - `--temperature`: 蒸馏温度（默认4.0）
   - `--alpha`: 蒸馏损失权重（默认0.7）
- **输出**：
   - 详细结果：`transfer_results/exp_*/transfer_results.json`
   - 结果摘要：`transfer_results/exp_*/transfer_summary.csv`
   - 可视化图表：准确率演变、性能对比、循环提升效果
- **示例**：
   ```bash
   python 05_knowledge_transfer/main.py \
         --num-cycles 3 \
         --initial-fedavg-rounds 40 \
         --kd-rounds 20 \
         --small-node-epochs 80 \
         --temperature 4.5 \
         --alpha 0.65
   ```

### 06_kd_params_study

- **脚本**：`python 06_kd_params_study/main.py [args...]`
- **用途**：系统研究不同KD参数（Temperature和Alpha）在循环迭代中的效果。
- **实验设计**：
   - 参数网格：Temperature × Alpha的笛卡尔积
   - 对每组参数执行完整的循环知识迁移实验
   - 默认Temperature: [1, 2, 3, 4, 5, 6, 8, 10]
   - 默认Alpha: [0.1, 0.3, 0.5, 0.7, 0.9]
- **核心参数**：
   - `--temperatures`: Temperature参数列表
   - `--alphas`: Alpha参数列表
   - `--num-cycles`: 每组参数的循环次数（默认5）
   - `--initial-fedavg-rounds`: 初始FedAvg轮次（默认50）
   - `--kd-rounds`: KD后FedAvg轮次（默认25）
   - `--small-node-epochs`: 小节点KD训练epoch（默认100）
- **输出**：
   - CSV摘要：`param_study_results/kd_params_study_*.csv`
   - 详细JSON：`param_study_results/kd_params_study_detailed_*.json`
   - 可视化：热力图、参数影响折线图、性能对比
- **示例**：
   ```bash
   # 快速测试（少量参数组合）
   python 06_kd_params_study/main.py \
         --temperatures 2 4 6 \
         --alphas 0.5 0.7 \
         --num-cycles 2 \
         --initial-fedavg-rounds 10 \
         --kd-rounds 5
   
   # 完整参数扫描
   python 06_kd_params_study/main.py \
         --temperatures 1 2 3 4 5 6 8 10 \
         --alphas 0.1 0.3 0.5 0.7 0.9 \
         --num-cycles 5
   ```

## 输出与日志

- **01-04项目统一日志目录**：`logs/`
   - `experiment_*.log` / `kd_tuning_*.log` / `compare_noniid_*.log`
   - 数据分布表：`data_distribution_alpha*_*.txt`
- **05项目输出**：`05_knowledge_transfer/transfer_results/`
   - 每次实验创建独立的时间戳目录 `exp_YYYYMMDD_HHMMSS/`
   - JSON详细结果、CSV摘要、可视化图表（PNG）
- **06项目输出**：`06_kd_params_study/param_study_results/`
   - 参数扫描结果CSV和详细JSON
   - 热力图、折线图等可视化（PNG）
- **结果文件**：
   - CSV/JSON 会带时间戳方便追踪
   - 图像 (PNG) 默认也存放在对应目录

> 所有脚本都配置了输出目录参数，如需自定义可通过 `--output-dir` 或 `--log_dir` 指定。

## 运行提示

- 建议使用 GPU (`--device cuda`)，否则需适度降低 `--epochs` 或 `--batch_size`。
- 当 `Dirichlet α` 较小时（如 0.1），各客户端类别分布会非常不均衡；可通过生成的表格快速检查。
- 若遇到显存不足，降低 `--batch_size` 或使用 `--epochs` 较小的快速实验模式。

## 参考文献

1. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
2. McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017
3. Hinton et al., *Distilling the Knowledge in a Neural Network*, NIPS 2014 Workshop

## License

MIT License
