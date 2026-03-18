# BJTUPubClaw · 热点助手与舆情分析

来自北京交通大学传播系的实用舆情工具：最快 30 秒部署的热点助手 —— 告别无效刷屏，只看真正关心的新闻资讯；同时提供热点事件的深度解析与简版论坛讨论。核心工作流基于 TrendRadar LangGraph 版本改造而来。

## 功能概览

- 聚合多平台热榜，自动去重与排序
- 可选接入大模型，生成趋势总结、专题研判与论坛讨论
- 一键生成自包含 HTML 报告（可直接打开或托管）
- 留存数据快照，便于复盘与追踪
- **📊 平台对比模式**：选择 2-3 个平台并排对比展示，支持三种布局方式

## 📊 平台对比模式（新增！）

快速对比不同平台的热点差异：

```bash
# 对比微博、知乎、抖音
python3 main.py --compare weibo,zhihu,douyin

# 使用网格布局
python3 main.py --compare weibo,zhihu,douyin --layout grid

# 使用标签页布局，显示 Top 10
python3 main.py --compare weibo,zhihu,douyin --layout tabs --top-n 10
```

详细说明请查看 [COMPARE_MODE.md](COMPARE_MODE.md)

## 快速开始

1) 克隆仓库

```bash
git clone https://github.com/beingaigital/bjtupubclaw
```

2) 创建并激活虚拟环境（推荐）

- macOS / Linux:
```bash
cd bjtupubclaw
python3 -m venv .venv
source .venv/bin/activate
```

- Windows (PowerShell):
```powershell
cd bjtupubclaw
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

3) 安装依赖

```bash
pip install -r requirements.txt
```

4) 配置环境变量（.env）

在仓库根目录新建 `.env` 文件，至少设置一个可用的 API Key。支持以下任一密钥，程序会自动兼容并映射：

```dotenv
# 至少设置一个即可
# OPENAI 或第三方兼容接口
OPENAI_API_KEY=sk-xxxxx

# 或选用以下任意一个
# Kimi
# KIMI_API_KEY=xxxxx
# Qwen（DashScope）
# QWEN_API_KEY=xxxxx
# DASHSCOPE_API_KEY=xxxxx

# 可选：自定义 Base URL / 模型名（不设置将采用合理默认值）
# OPENAI_BASE_URL=https://api.openai.com/v1
# KIMI_BASE_URL=https://api.moonshot.cn/v1
# QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# KIMI_MODEL_NAME=moonshot-v1-8k
# QWEN_MODEL_NAME=qwen-plus
```

说明：
- 程序会自动读取项目根目录的 `.env`（见 [env.py](file:///Users/biaowenhuang/Documents/bjtupubclaw/env.py)），也可通过环境变量 `ENV_FILE` 指定其他路径。
- 未配置任何 Key 时，程序会自动降级：仍生成报告，但分析为基础版本。

5) 运行

运行方式 A（推荐）：在仓库根目录直接执行脚本
```bash
python3 main.py
```
运行方式 B：在仓库所在目录的上一级目录执行
```bash
python3 -m bjtupubclaw.main
```

运行方式 C：指定自定义配置文件
```bash
python3 -m bjtupubclaw.main --config /absolute/path/to/config.yaml
```

运行方式 D：启用平台对比模式
```bash
# 对比微博、知乎、抖音（并排布局）
python3 main.py --compare weibo,zhihu,douyin

# 网格布局
python3 main.py --compare weibo,zhihu,douyin --layout grid

# 标签页布局
python3 main.py --compare weibo,zhihu,douyin --layout tabs

# 自定义显示条数
python3 main.py --compare weibo,zhihu --top-n 10
```

说明：
- 入口见 [main.py](file:///Users/biaowenhuang/Documents/bjtupubclaw/main.py)，内部调用了基于 LangGraph 的 [agent.py](file:///Users/biaowenhuang/Documents/bjtupubclaw/agent.py) 的 `run`。
- 主要配置文件为 `config/config.yaml`，根目录的 `config.yaml` 内容与其保持一致，便于通过 `--config` 参数指定绝对路径。
- 对比模式详细用法请查看 [COMPARE_MODE.md](COMPARE_MODE.md)


## 输出目录

- 报告（LangGraph 工作流）：`output_langgraph/index.html` 与 `output_langgraph/bjtupubclaw_report_YYYYMMDD_HHMMSS.html`
- 快照与历史数据：`data_langgraph_hourly/`（按小时存储各平台原始数据）、`data_langgraph/`（合并快照）

## 进阶

- 无 Key 跑通：若暂时没有可用的 API Key，直接运行也能生成报告（降级为基础分析）。
- 托管到静态站点：将 `output/index.html` 上传至任意静态托管（如 GitHub Pages / Vercel 静态站）。

## 故障排查

- pip 安装出现证书错误（macOS 常见）：
  - 报错示例：`SSLCertVerificationError: certificate verify failed: unable to get local issuer certificate`
  - 临时解决（一次性）：
    - macOS/Linux:
      ```bash
      SSL_CERT_FILE="$(python3 -c 'import pip._vendor.certifi as c; print(c.where())')" python3 -m pip install -r requirements.txt
      ```
    - Windows PowerShell:
      ```powershell
      setx SSL_CERT_FILE "$(python -c \"import pip._vendor.certifi as c; print(c.where())\")"
      python -m pip install -r requirements.txt
      ```
  - 长期配置（推荐）：
    - 将下行加入你的 shell 配置文件（如 `~/.zshrc`），重启终端：
      ```bash
      export SSL_CERT_FILE="$(python3 -c 'import pip._vendor.certifi as c; print(c.where())')"
      ```
    - 若使用 python.org 安装包，运行：`/Applications/Python 3.x/Install Certificates.command`
  - 企业代理环境：请将公司代理根证书导入系统信任，并配置 `PIP_CERT` 或 `REQUESTS_CA_BUNDLE` 指向该证书。
  - 验证：
    ```bash
    python3 -m pip --version
    python3 -c "import certifi,sys; print(sys.executable); print(certifi.where())"
    ```

- ModuleNotFoundError（如 `No module named 'requests'`）：
  - 确保已激活虚拟环境并使用同一个解释器安装依赖：
    - macOS/Linux:
      ```bash
      source .venv/bin/activate
      python3 -m pip install -r requirements.txt
      ```
    - Windows PowerShell:
      ```powershell
      .venv\Scripts\Activate.ps1
      python -m pip install -r requirements.txt
      ```
  - 如仍有冲突，强制使用虚拟环境内解释器：
    - macOS/Linux: `.venv/bin/python -m pip install -r requirements.txt`
    - Windows: `.venv\Scripts\python.exe -m pip install -r requirements.txt`

- 相对导入错误（`attempted relative import with no known parent package`）：
  - 请选择以下任一方式运行：
    - 仓库上一级目录：`python3 -m bjtupubclaw.main`
    - 仓库根目录：`python3 main.py`
  - 本仓库已做兼容处理，二者皆可运行。

## 致谢

- 代码基础与灵感来源：TrendRadar 项目
