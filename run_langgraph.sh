#!/bin/bash

# TrendRadar LangGraph 运行脚本
# 使用方法: ./run_langgraph.sh

# 切换到脚本所在目录
cd "$(dirname "$0")"

echo "🚀 启动 TrendRadar LangGraph Agent..."
echo ""

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo "⚠️  警告: 未找到 .env 文件"
    echo "请确保已在项目根目录创建 .env 文件并配置 API Key"
    echo ""
    read -p "是否继续运行? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查 Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ 错误: 未找到 Python，请先安装 Python 3.8+"
    exit 1
fi

echo "使用 Python: $PYTHON_CMD"
echo ""

# 运行程序
$PYTHON_CMD trend_radar_langgraph.py

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 运行完成！"
    echo ""
    echo "📁 报告位置: output_langgraph/"
    if [ -d "output_langgraph" ]; then
        echo "最新报告:"
        ls -lt output_langgraph/*.html 2>/dev/null | head -1 | awk '{print $NF}'
    fi
else
    echo ""
    echo "❌ 运行失败，请检查错误信息"
fi

