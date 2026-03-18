# 📊 平台对比模式使用说明

平台对比模式允许你选择 2-3 个平台进行并排对比展示，方便快速比较不同平台的热点差异。

## 🚀 快速开始

### 方式一：命令行参数（推荐）

临时启用对比模式，无需修改配置文件：

```bash
# 对比微博、知乎、抖音三个平台
python3 main.py --compare weibo,zhihu,douyin

# 对比微博和百度（2 个平台）
python3 main.py --compare weibo,baidu

# 使用网格布局
python3 main.py --compare weibo,zhihu,douyin --layout grid

# 使用标签页布局，每个平台显示 Top 10
python3 main.py --compare weibo,zhihu,douyin --layout tabs --top-n 10
```

### 方式二：配置文件（固定配置）

在 `config.yaml` 中永久配置对比模式：

```yaml
compare_mode:
  enabled: true  # 启用对比模式
  platforms:     # 选择要对比的平台
    - weibo
    - zhihu
    - douyin
  layout: "side-by-side"  # 布局方式
  top_n: 20               # 每个平台显示的条数
```

然后直接运行：
```bash
python3 main.py
```

### 方式三：组合使用

配置文件中设置默认对比配置，命令行参数可以临时覆盖：

```bash
# 即使 config.yaml 中 enabled=true，命令行指定 --compare 会覆盖配置
python3 main.py --compare toutiao,baidu,zhihu --layout grid
```

## 📐 布局方式

| 布局 | 参数 | 说明 | 适用场景 |
|------|------|------|----------|
| 并排 | `side-by-side` (默认) | 平台垂直排列，一个接一个 | 详细阅读每个平台 |
| 网格 | `grid` | 2-3 个平台并排显示 | 快速横向对比 |
| 标签页 | `tabs` | 点击标签切换平台 | 节省空间，专注单一平台 |

## 🎯 可用平台 ID

以下平台可用于对比：

| ID | 平台名称 |
|----|----------|
| `weibo` | 微博热搜 |
| `zhihu` | 知乎热榜 |
| `douyin` | 抖音热榜 |
| `baidu` | 百度热搜 |
| `toutiao` | 今日头条 |
| `bilibili-hot-search` | B 站热搜 |
| `tieba` | 贴吧 |
| `thepaper` | 澎湃新闻 |
| `ifeng` | 凤凰网 |
| `cls-hot` | 财联社热门 |
| `wallstreetcn-hot` | 华尔街见闻 |

## 💡 使用技巧

1. **推荐对比组合**：
   - 社交媒体对比：`weibo,zhihu,douyin`
   - 新闻门户对比：`toutiao,baidu,thepaper`
   - 财经新闻对比：`cls-hot,wallstreetcn`

2. **网格布局最佳实践**：
   - 2 个平台：网格布局效果最佳
   - 3 个平台：并排布局更易阅读

3. **标签页布局**：
   - 适合需要详细查看单个平台内容时
   - 可以在不滚动的情况下快速切换

4. **调整显示数量**：
   - 默认显示 Top 20
   - 快速浏览可用 `--top-n 10`
   - 详细分析可用 `--top-n 50`

## 🖼️ 示例截图

运行对比模式后，HTML 报告顶部会显示对比模式指示器：
```
📊 平台对比模式：微博热搜，知乎热榜，抖音热榜
```

各平台卡片会根据选择的布局方式进行排列。

## ⚙️ 环境变量

命令行参数通过以下环境变量传递给内部处理：

- `COMPARE_MODE`: `true` / `false`
- `COMPARE_PLATFORMS`: 逗号分隔的平台 ID 列表
- `COMPARE_LAYOUT`: 布局方式
- `COMPARE_TOP_N`: 每个平台显示的条数

## 🔧 故障排除

**问题：对比模式没有生效**
- 检查平台 ID 是否正确（使用英文 ID，不是中文名称）
- 确认至少选择了 2 个平台
- 查看终端输出，确认有 "📊 平台对比模式已启用" 的日志

**问题：某些平台没有显示**
- 确认该平台在抓取时有数据
- 检查平台 ID 拼写是否正确
- 如果所有对比平台都没有数据，会自动降级到普通模式

---

**版本**: v1.0  
**最后更新**: 2026-03-17