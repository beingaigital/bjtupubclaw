# TrendRadar LangGraph 版

这是 TrendRadar 的 LangGraph 重构版本，旨在提供更灵活、可扩展的工作流管理。

## 🚀 快速开始

### 一键运行（3 步）

1. **安装依赖**
   ```bash
   pip install langchain langgraph langchain-openai pyyaml requests pytz
   ```

2. **创建 `.env` 文件**（在项目根目录）
   ```bash
   KIMI_API_KEY="sk-你的API密钥"
   KIMI_BASE_URL="https://api.moonshot.cn/v1"
   KIMI_MODEL_NAME="moonshot-v1-8k"
   ```

3. **运行程序**
   
   **方式 A: 直接运行（推荐）**
   ```bash
   # 在项目根目录执行
   cd /Users/biaowenhuang/Documents/TrendRadar
   python3 trend_radar_langgraph.py
   ```
   
   **方式 B: 使用运行脚本（更便捷）**
   ```bash
   # 在项目根目录执行
   ./run_langgraph.sh
   ```

报告将自动生成在 `output_langgraph` 目录下！

## 架构说明

本项目使用 [LangGraph](https://python.langchain.com/docs/langgraph) 构建了一个多节点的有向无环图 (DAG) 工作流：

1.  **SpiderNode**: 负责从多平台抓取热搜数据。
2.  **InsightNode**: 使用 LLM 对抓取的数据进行深度分析和情感提取。
3.  **ForumNode**: 模拟多角色（分析师、媒体、事实核查员）对热点话题进行讨论。
4.  **ReportNode**: 汇总所有信息，生成 HTML 格式的日报。

### 状态管理 (TrendState)

工作流通过 `TrendState` 在各节点间传递数据，主要字段包括：
- `news_data`: 原始新闻列表
- `analysis_result`: 结构化分析结果
- `forum_discussion`: 论坛讨论文本
- `html_report`: 最终报告内容

## 环境准备

### 步骤 1: 安装依赖

首先确保已安装 Python 3.8+，然后安装必要的依赖包：

```bash
# 安装 LangGraph 相关依赖
pip install langchain langgraph langchain-openai pyyaml requests pytz

# 或者如果项目有 requirements.txt，也可以先安装基础依赖
pip install -r requirements.txt
pip install langchain langgraph langchain-openai
```

### 步骤 2: 配置环境变量

有两种方式配置环境变量：

#### 方式一：使用 .env 文件（推荐）

在项目根目录创建 `.env` 文件，添加以下配置：

```bash
# 方式 1: 使用统一的 KIMI 配置（推荐，代码会自动映射到各个引擎）
KIMI_API_KEY="sk-..."
KIMI_BASE_URL="https://api.moonshot.cn/v1"
KIMI_MODEL_NAME="moonshot-v1-8k"

# 方式 2: 分别配置各个引擎（更灵活）
# 舆情分析模型
INSIGHT_ENGINE_API_KEY="sk-..."
INSIGHT_ENGINE_BASE_URL="https://api.moonshot.cn/v1"
INSIGHT_ENGINE_MODEL_NAME="moonshot-v1-8k"

# 报告生成模型
REPORT_ENGINE_API_KEY="sk-..."
REPORT_ENGINE_BASE_URL="https://aihubmix.com/v1"
REPORT_ENGINE_MODEL_NAME="moonshot-v1-8k"

# 论坛讨论模型
QUERY_ENGINE_API_KEY="sk-..."
QUERY_ENGINE_BASE_URL="https://api.deepseek.com"
QUERY_ENGINE_MODEL_NAME="deepseek-chat"
```

#### 方式二：使用系统环境变量

在终端中设置环境变量：

```bash
# macOS/Linux
export INSIGHT_ENGINE_API_KEY="sk-..."
export INSIGHT_ENGINE_BASE_URL="https://api.moonshot.cn/v1"
export REPORT_ENGINE_API_KEY="sk-..."
export REPORT_ENGINE_BASE_URL="https://aihubmix.com/v1"
export QUERY_ENGINE_API_KEY="sk-..."
export QUERY_ENGINE_BASE_URL="https://api.deepseek.com"

# Windows (PowerShell)
$env:INSIGHT_ENGINE_API_KEY="sk-..."
$env:INSIGHT_ENGINE_BASE_URL="https://api.moonshot.cn/v1"
# ... 其他变量类似
```

### 步骤 3: 检查配置文件

确保 `config/config.yaml` 文件存在且配置正确。主要检查：
- `crawler.enable_crawler`: 是否启用爬虫功能
- `platforms`: 需要抓取的平台列表

## 运行

### 基本运行

在项目根目录执行：

**macOS/Linux:**
```bash
# 方式 1: 使用 python3（推荐）
python3 trend_radar_langgraph.py

# 方式 2: 使用运行脚本
./run_langgraph.sh
```

**Windows:**
```bash
python trend_radar_langgraph.py
```

> 💡 **提示**: 如果 `python` 命令不可用，请使用 `python3`。可以通过 `which python3` 或 `python3 --version` 检查 Python 是否已安装。

### 运行流程说明

程序会按以下顺序执行：

1. **SpiderNode** → 从配置的平台抓取热搜数据
2. **InsightNode** → 使用 LLM 分析数据并提取情感
3. **ForumNode** → 模拟多角色讨论热点话题
4. **ReportNode** → 生成 HTML 格式的日报

### 输出结果

运行成功后，报告将生成在 `output_langgraph` 目录下，文件命名格式为：
```
output_langgraph/YYYY年MM月DD日_HH时MM分.html
```

### 常见问题排查

1. **依赖缺失错误**
   ```bash
   ModuleNotFoundError: No module named 'langgraph'
   ```
   解决：运行 `pip install langgraph langchain langchain-openai`

2. **API Key 错误**
   ```
   Authentication failed
   ```
   解决：检查 `.env` 文件中的 API Key 是否正确，或检查环境变量是否设置

3. **配置文件错误**
   ```
   FileNotFoundError: config/config.yaml
   ```
   解决：确保在项目根目录运行，且 `config/config.yaml` 文件存在

4. **网络连接问题**
   ```
   Connection timeout
   ```
   解决：检查网络连接，或配置代理（在 `config/config.yaml` 中设置 `crawler.use_proxy: true`）

## 扩展指南

由于使用了 LangGraph，您可以轻松地：
- **添加循环**: 例如，如果分析结果不满意，自动返回上一步重试。
- **并行执行**: 同时运行多个分析节点，最后汇总结果。
- **人工介入**: 在生成报告前暂停流程，等待人工确认（Human-in-the-loop）。
