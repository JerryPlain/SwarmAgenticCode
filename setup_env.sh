#!/bin/bash
# SwarmAgentic 一键环境配置脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "SwarmAgentic 一键环境配置"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查是否在项目根目录
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ 错误: 请在项目根目录运行此脚本${NC}"
    exit 1
fi

PROJECT_DIR=$(pwd)
echo -e "${GREEN}✓ 项目目录: $PROJECT_DIR${NC}"

# 检查conda是否安装
echo ""
echo "步骤 1/5: 检查Conda安装..."
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ 未找到conda，请先安装conda:${NC}"
    echo "   macOS/Linux: https://docs.conda.io/en/latest/miniconda.html"
    echo "   Windows: https://www.anaconda.com/download"
    exit 1
fi
echo -e "${GREEN}✓ Conda已安装: $(conda --version)${NC}"

# 初始化conda
eval "$(conda shell.bash hook)"

# 检查并创建环境
echo ""
echo "步骤 2/5: 检查/创建conda环境..."
if conda env list | grep -q "^swarm "; then
    echo -e "${YELLOW}⚠️  Swarm环境已存在，将使用现有环境${NC}"
else
    echo "正在创建swarm环境 (Python 3.11)..."
    conda create -n swarm python=3.11 -y
    echo -e "${GREEN}✓ Swarm环境创建成功${NC}"
fi

# 激活环境
echo ""
echo "步骤 3/5: 激活环境并安装依赖..."
conda activate swarm

# 检查Python版本
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python版本: $PYTHON_VERSION${NC}"

# 升级pip
echo "升级pip..."
pip install --upgrade pip -q

# 安装依赖
echo "安装项目依赖..."
if pip install -r requirements.txt; then
    echo -e "${GREEN}✓ 依赖包安装完成${NC}"
else
    echo -e "${RED}❌ 依赖包安装失败${NC}"
    exit 1
fi

# 验证关键包
echo ""
echo "步骤 4/5: 验证关键包..."
python -c "import openai, langchain, numpy, pandas, tqdm" 2>/dev/null && \
    echo -e "${GREEN}✓ 所有关键包验证通过${NC}" || \
    (echo -e "${RED}❌ 包验证失败${NC}" && exit 1)

# 检查API密钥
echo ""
echo "步骤 5/5: 检查API密钥配置..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠️  OPENAI_API_KEY 环境变量未设置${NC}"
    echo ""
    echo "请选择设置方式:"
    echo "1. 现在设置 (仅当前会话有效)"
    echo "2. 稍后手动设置"
    echo ""
    read -p "请输入选项 (1/2): " choice
    
    if [ "$choice" = "1" ]; then
        read -p "请输入您的OpenAI API密钥: " api_key
        export OPENAI_API_KEY="$api_key"
        echo -e "${GREEN}✓ API密钥已设置 (仅当前会话有效)${NC}"
        echo ""
        echo -e "${YELLOW}提示: 要永久设置，请运行:${NC}"
        echo "  echo 'export OPENAI_API_KEY=\"$api_key\"' >> ~/.zshrc"
        echo "  source ~/.zshrc"
    else
        echo -e "${YELLOW}提示: 稍后请运行以下命令设置API密钥:${NC}"
        echo "  export OPENAI_API_KEY='your-api-key-here'"
    fi
else
    masked_key="${OPENAI_API_KEY:0:10}...${OPENAI_API_KEY: -4}"
    echo -e "${GREEN}✓ API密钥已设置 ($masked_key)${NC}"
fi

# 运行环境检查
echo ""
echo "=========================================="
echo "运行环境检查..."
echo "=========================================="
python check_env.py

echo ""
echo "=========================================="
echo -e "${GREEN}🎉 环境配置完成！${NC}"
echo "=========================================="
echo ""
echo "下一步操作:"
echo "1. 运行快速测试: ./quick_start.sh"
echo "2. 或手动运行任务:"
echo "   conda activate swarm"
echo "   cd mgsm && python pso.py --max_iteration 2 --dataset_size 10"
echo ""
echo "详细指南请查看: 配置和复现指南.md"
echo ""
