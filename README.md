## 项目概览

本仓库围绕 **CIFAR-10 + ResNet18** 构建了四个可独立运行的小项目，用于研究联邦平均 (FedAvg) 与知识蒸馏 (Knowledge Distillation) 在 IID/Non-IID 设置下的表现。

```
KD_vs_FedAvg/
├── common/                       # 共享模块（模型、数据、训练例程）
│   ├── __init__.py
│   ├── data_loader.py
│   ├── fedavg_training.py
│   ├── kd_training.py
│   └── models.py
├── 01_iid_comparison/
│   ├── main.py                   # IID场景：FedAvg vs KD 对比
│   └── iid_comparison_results/   # 结果留存目录
├── 02_iid_kd_tune/
│   ├── tune_kd_params.py         # IID场景的KD参数调优
│   └── kd_iid_tuning_results/
├── 03_non_iid_kd_tune/
│   ├── tune_kd_params_noniid.py  # Non-IID场景的KD参数调优（支持多Dirichlet α）
│   └── kd_non_iid_tuning_results/
├── 04_non_iid_comparison/
│   ├── compare_noniid.py         # Non-IID场景：FedAvg vs KD 对比（多Dirichlet α）
│   └── comparison_outputs/
├── data/                         # CIFAR-10（二进制格式，需提前下载/解压）
├── logs/                         # 所有脚本默认写入的日志与可视化文件夹
├── old_results/                  # 旧版结果归档
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
| `data_loader.py` | CIFAR-10 加载、IID拆分、Dirichlet Non-IID划分（含表格化分布输出） |
| `fedavg_training.py` | FedAvg 训练/评估流程 |
| `kd_training.py` | 教师-学生蒸馏训练流水线 |

所有子项目仅需从 `common` 导入，不再复制粘贴公共逻辑。

## 四个子项目

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

## 输出与日志

- 统一日志目录：`logs/`
   - `experiment_*.log` / `kd_tuning_*.log` / `compare_noniid_*.log`
   - 数据分布表：`data_distribution_alpha*_*.txt`
- 结果文件：
   - CSV/JSON 会带时间戳方便追踪
   - 图像 (PNG) 默认也存放在 `logs/`

> 所有脚本都配置了 `--log_dir` 参数，如需分项目保存可手动传入自定义路径。

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
