#!/bin/bash
# SwarmAgentic 快速启动脚本
# 用于快速验证环境配置和运行测试

set -e  # 遇到错误立即退出

echo "=========================================="
echo "SwarmAgentic 环境验证和快速测试"
echo "=========================================="
echo ""

# 检查是否在项目根目录
if [ ! -f "requirements.txt" ]; then
    echo "❌ 错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到conda，请先安装conda"
    exit 1
fi

echo "✓ Conda已安装"

# 检查环境是否存在
if conda env list | grep -q "^swarm "; then
    echo "✓ Swarm环境已存在"
else
    echo "⚠️  Swarm环境不存在，正在创建..."
    conda create -n swarm python=3.11 -y
    echo "✓ Swarm环境创建成功"
fi

# 激活环境
echo ""
echo "激活conda环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate swarm

# 检查Python版本
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python版本: $PYTHON_VERSION"

# 检查依赖是否安装
echo ""
echo "检查依赖包..."
if python -c "import openai, langchain, numpy, pandas" 2>/dev/null; then
    echo "✓ 主要依赖包已安装"
else
    echo "⚠️  正在安装依赖包..."
    pip install -r requirements.txt
    echo "✓ 依赖包安装完成"
fi

# 检查API密钥
echo ""
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  警告: OPENAI_API_KEY 环境变量未设置"
    echo "   请运行: export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    read -p "是否现在设置API密钥? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "请输入您的OpenAI API密钥: " api_key
        export OPENAI_API_KEY="$api_key"
        echo "✓ API密钥已设置 (仅当前会话有效)"
    fi
else
    echo "✓ OPENAI_API_KEY 已设置"
fi

# 显示菜单
echo ""
echo "=========================================="
echo "请选择要运行的任务:"
echo "=========================================="
echo "1. MGSM (数学推理) - 快速测试 (2次迭代, 10个样本)"
echo "2. Creative Writing (创意写作) - 快速测试 (2次迭代)"
echo "3. TravelPlanner (旅行规划) - 完整训练 (5次迭代)"
echo "4. 仅验证环境 (不运行训练)"
echo "5. 退出"
echo ""

read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        echo ""
        echo "开始MGSM快速测试..."
        cd mgsm
        python pso.py --max_iteration 2 --dataset_size 10
        echo ""
        echo "✓ MGSM测试完成!"
        echo "运行评估: cd mgsm && python test.py --particle_idx -1"
        ;;
    2)
        echo ""
        echo "开始Creative Writing快速测试..."
        cd creative_writing
        python pso.py --max_iteration 2 --dataset data/data_100_random_text.txt
        echo ""
        echo "✓ Creative Writing测试完成!"
        echo "运行评估: cd creative_writing && python test.py --particle_idx -1"
        ;;
    3)
        echo ""
        echo "开始TravelPlanner训练..."
        cd travelplanner/swarm
        python pso.py --max_iteration 5 --dataset data/train_45.jsonl --ref_info data/train_ref_info.jsonl
        echo ""
        echo "✓ TravelPlanner训练完成!"
        echo "运行评估: cd travelplanner/swarm && python test.py --particle_idx -1"
        ;;
    4)
        echo ""
        echo "✓ 环境验证完成!"
        echo ""
        echo "环境信息:"
        echo "  - Conda环境: swarm"
        echo "  - Python版本: $PYTHON_VERSION"
        echo "  - 工作目录: $(pwd)"
        if [ -n "$OPENAI_API_KEY" ]; then
            echo "  - API密钥: 已设置 (${OPENAI_API_KEY:0:10}...)"
        else
            echo "  - API密钥: 未设置"
        fi
        ;;
    5)
        echo "退出"
        exit 0
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "完成!"
echo "=========================================="
