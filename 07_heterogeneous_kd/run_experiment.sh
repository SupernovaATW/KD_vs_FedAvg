#!/bin/bash

# 异构模型知识蒸馏实验启动脚本

echo "=================================="
echo "异构模型知识蒸馏实验"
echo "=================================="

# 激活conda环境（如果需要）
# conda activate FedDistill

# 设置默认参数
NUM_CYCLES=10
LOCAL_EPOCHS=20
SMALL_EPOCHS=20
LARGE_KD_EPOCHS=20
LR=0.1
SEED=42

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --cycles)
            NUM_CYCLES="$2"
            shift 2
            ;;
        --local-epochs)
            LOCAL_EPOCHS="$2"
            shift 2
            ;;
        --small-epochs)
            SMALL_EPOCHS="$2"
            shift 2
            ;;
        --large-kd-epochs)
            LARGE_KD_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --quick-test)
            echo "运行快速测试..."
            python quick_test.py
            exit 0
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --quick-test         运行快速测试（2 epochs, 1 cycle）"
            echo "  --cycles N           设置循环次数（默认: 10）"
            echo "  --local-epochs N     设置大节点本地训练epochs（默认: 20）"
            echo "  --small-epochs N     设置小节点KD训练epochs（默认: 20）"
            echo "  --large-kd-epochs N  设置大节点KD训练epochs（默认: 20）"
            echo "  --lr RATE            设置学习率（默认: 0.1）"
            echo "  --seed N             设置随机种子（默认: 42）"
            echo "  --help               显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看可用选项"
            exit 1
            ;;
    esac
done

# 显示配置
echo "配置:"
echo "  循环次数: $NUM_CYCLES"
echo "  大节点本地训练: $LOCAL_EPOCHS epochs"
echo "  小节点KD训练: $SMALL_EPOCHS epochs"
echo "  大节点KD训练: $LARGE_KD_EPOCHS epochs"
echo "  学习率: $LR"
echo "  随机种子: $SEED"
echo "=================================="

# 运行实验
python main.py \
    --num-cycles $NUM_CYCLES \
    --local-epochs $LOCAL_EPOCHS \
    --small-node-epochs $SMALL_EPOCHS \
    --large-node-kd-epochs $LARGE_KD_EPOCHS \
    --lr $LR \
    --seed $SEED \
    --small-temperatures 2.0 4.0 6.0 \
    --small-alphas 0.5 0.7 0.9 \
    --large-temperatures 2.0 4.0 6.0 \
    --large-alphas 0.5 0.7 0.9

echo ""
echo "=================================="
echo "实验完成!"
echo "=================================="
