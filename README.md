# BJTU Public Law & Public Opinion Lab (bjtupubclaw)

基于原 TrendRadar 项目二次开发的 **舆情抓取 + 分析 + 可视化** 工具，使用 LangGraph 组织抓取流程，并输出现代化 HTML 舆情日报界面。

## 核心入口

- `bjtupubclaw.py`：当前推荐的启动入口（内部调用 `trend_radar_langgraph.py` 的 `main()`）。
- `trend_radar_langgraph.py`：LangGraph 工作流主文件，定义了抓取、分析、讨论与报告生成的节点与状态结构。

## 运行方式（简要）

```bash
python bjtupubclaw.py
```

更多详细配置（如平台列表、抓取频率、通知方式等），请参考 `README_LangGraph.md` 与 `config/config.yaml` 中的说明。

