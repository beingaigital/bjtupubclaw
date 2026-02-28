# coding=utf-8
import json
import os
import time
import webbrowser
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TypedDict, Annotated
import operator
import pytz
import requests

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

def load_env_file(env_path: Optional[str] = None) -> None:
    candidate_paths = []
    if env_path:
        candidate_paths.append(Path(env_path))
    else:
        candidate_paths.append(Path(os.environ.get("ENV_FILE", ".env")))
        candidate_paths.append(Path.cwd() / ".env")
        candidate_paths.append(Path(__file__).resolve().parent / ".env")
        for base in [Path.cwd(), Path(__file__).resolve().parent]:
            current = base
            for _ in range(4):
                candidate_paths.append(current / ".env")
                if current.parent == current:
                    break
                current = current.parent
    unique_paths = []
    for p in candidate_paths:
        if p not in unique_paths:
            unique_paths.append(p)
    path = next((p for p in unique_paths if p.exists()), None)
    if not path:
        return
    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"]
    last_error = None
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    if key.startswith("export "):
                        key = key[7:].strip()
                    key = key.lstrip("\ufeff")
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            return
        except Exception as e:
            last_error = e
    if last_error:
        print(f"读取 .env 失败: {last_error}")

def apply_env_aliases() -> None:
    if "INSIGHT_ENGINE_API_KEY" not in os.environ and os.environ.get("KIMI_API_KEY"):
        os.environ["INSIGHT_ENGINE_API_KEY"] = os.environ["KIMI_API_KEY"]
    if "REPORT_ENGINE_API_KEY" not in os.environ and os.environ.get("KIMI_API_KEY"):
        os.environ["REPORT_ENGINE_API_KEY"] = os.environ["KIMI_API_KEY"]
    if "QUERY_ENGINE_API_KEY" not in os.environ and os.environ.get("KIMI_API_KEY"):
        os.environ["QUERY_ENGINE_API_KEY"] = os.environ["KIMI_API_KEY"]
    if "OPENAI_API_KEY" not in os.environ and os.environ.get("KIMI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["KIMI_API_KEY"]
    if "INSIGHT_ENGINE_BASE_URL" not in os.environ and os.environ.get("KIMI_BASE_URL"):
        os.environ["INSIGHT_ENGINE_BASE_URL"] = os.environ["KIMI_BASE_URL"]
    if "REPORT_ENGINE_BASE_URL" not in os.environ and os.environ.get("KIMI_BASE_URL"):
        os.environ["REPORT_ENGINE_BASE_URL"] = os.environ["KIMI_BASE_URL"]
    if "QUERY_ENGINE_BASE_URL" not in os.environ and os.environ.get("KIMI_BASE_URL"):
        os.environ["QUERY_ENGINE_BASE_URL"] = os.environ["KIMI_BASE_URL"]
    if "OPENAI_BASE_URL" not in os.environ and os.environ.get("KIMI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.environ["KIMI_BASE_URL"]
    if "INSIGHT_ENGINE_MODEL_NAME" not in os.environ:
        if os.environ.get("KIMI_MODEL_NAME"):
            os.environ["INSIGHT_ENGINE_MODEL_NAME"] = os.environ["KIMI_MODEL_NAME"]
        elif os.environ.get("KIMI_MODEL"):
            os.environ["INSIGHT_ENGINE_MODEL_NAME"] = os.environ["KIMI_MODEL"]
    if "REPORT_ENGINE_MODEL_NAME" not in os.environ:
        if os.environ.get("KIMI_MODEL_NAME"):
            os.environ["REPORT_ENGINE_MODEL_NAME"] = os.environ["KIMI_MODEL_NAME"]
        elif os.environ.get("KIMI_MODEL"):
            os.environ["REPORT_ENGINE_MODEL_NAME"] = os.environ["KIMI_MODEL"]
    if "QUERY_ENGINE_MODEL_NAME" not in os.environ:
        if os.environ.get("KIMI_MODEL_NAME"):
            os.environ["QUERY_ENGINE_MODEL_NAME"] = os.environ["KIMI_MODEL_NAME"]
        elif os.environ.get("KIMI_MODEL"):
            os.environ["QUERY_ENGINE_MODEL_NAME"] = os.environ["KIMI_MODEL"]

load_env_file()
apply_env_aliases()

# === 配置加载 ===
def load_config():
    """加载配置文件"""
    config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
    if not Path(config_path).exists():
        if Path("../config/config.yaml").exists():
            config_path = "../config/config.yaml"
        else:
            print(f"配置文件 {config_path} 不存在，使用默认配置")
            return {"crawler": {"request_interval": 1000}, "platforms": []}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# === 基础工具函数 ===
def get_beijing_time():
    return datetime.now(pytz.timezone("Asia/Shanghai"))

def ensure_directory_exists(directory: str):
    Path(directory).mkdir(parents=True, exist_ok=True)


def html_escape(text: str) -> str:
    """HTML转义"""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _is_docker_env() -> bool:
    """检测是否在 Docker 容器内运行（容器内不自动打开浏览器）"""
    if os.environ.get("DOCKER_CONTAINER") == "true":
        return True
    return os.path.exists("/.dockerenv")


# === 状态定义 ===
class TrendState(TypedDict):
    """定义工作流状态"""
    # 原始新闻数据
    news_data: Dict[str, Any]
    # 结构化分析结果
    analysis_result: Dict[str, Any]
    # 论坛讨论内容
    forum_discussion: str
    # 最终HTML报告
    html_report: str
    # 执行过程中的消息记录（可选，用于调试）
    messages: Annotated[List[BaseMessage], operator.add]
    # 错误信息
    error: Optional[str]

# === 节点定义 ===

class SpiderNode:
    """抓取节点"""
    def __init__(self):
        self.sources = {
            "weibo": "微博热搜",
            "zhihu": "知乎热榜",
            "bilibili-hot-search": "B站热搜",
            "toutiao": "今日头条",
            "douyin": "抖音热榜",
            "36kr": "36氪",
            "sspai": "少数派"
        }
        if CONFIG and "platforms" in CONFIG:
            self.sources = {p['id']: p.get('name', p['id']) for p in CONFIG['platforms']}

    def fetch_data(self, id_value: str) -> Optional[str]:
        url = f"https://newsnow.busiyi.world/api/s?id={id_value}&latest"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://newsnow.busiyi.world/"
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            print(f"Error fetching {id_value}: {e}")
        return None

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [SpiderNode] 开始抓取新闻 ---")
        news_list: List[Dict[str, Any]] = []
        for source_id, source_name in self.sources.items():
            print(f"正在抓取: {source_name} ({source_id})")
            raw_data = self.fetch_data(source_id)
            if raw_data:
                try:
                    data = json.loads(raw_data)
                    items = data.get("items", [])
                    # 为了提高样本量，这里尽量多保留一些热搜项（例如前 30 条）
                    top_items = items[:30]
                    for idx, item in enumerate(top_items, 1):
                        news_list.append({
                            "source_id": source_id,
                            "source": source_name,
                            "source_name": source_name,
                            "title": item.get("title"),
                            # 平台内排行（1 表示热搜第1）
                            "rank": idx,
                            "url": item.get("url", ""),
                            "mobile_url": item.get("mobileUrl", ""),
                            "hot_value": item.get("hotValue", 0)
                        })
                except Exception as e:
                    print(f"解析 {source_name} 失败: {e}")
            time.sleep(1)

        print(f"本次抓取完成，共 {len(news_list)} 条新闻")
        if len(news_list) == 0:
            print("⚠️  警告: 未能抓取到任何新闻，可能是 API 服务问题或网络连接问题")

        # 将本次抓取结果保存为快照，并合并最近 24 小时内的历史快照，扩大样本量
        try:
            snapshot_dir = Path("data_langgraph")
            ensure_directory_exists(str(snapshot_dir))
            timestamp = get_beijing_time()
            snapshot_name = f"snapshot_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            snapshot_path = snapshot_dir / snapshot_name
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": timestamp.isoformat(),
                        "items": news_list,
                    },
                    f,
                    ensure_ascii=False,
                )
        except Exception as e:
            print(f"保存快照失败: {e}")

        # 合并最近 24 小时的快照
        aggregated: Dict[tuple, Dict[str, Any]] = {}
        now = get_beijing_time()
        lookback_hours = 24
        cutoff = now - timedelta(hours=lookback_hours)

        try:
            for p in snapshot_dir.glob("snapshot_*.json"):
                try:
                    ts_str = p.stem.replace("snapshot_", "")
                    dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                except Exception:
                    continue
                if dt < cutoff:
                    continue
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    items = data.get("items") or []
                    for item in items:
                        key = (
                            item.get("source_id") or item.get("source") or "",
                            item.get("title") or "",
                        )
                        if not key[1]:
                            continue
                        existing = aggregated.get(key)
                        if existing is None:
                            aggregated[key] = dict(item)
                        else:
                            # 保留更靠前的排名和更高的热度
                            rank_new = item.get("rank")
                            rank_old = existing.get("rank")
                            if rank_new is not None and (
                                rank_old is None or rank_new < rank_old
                            ):
                                existing["rank"] = rank_new
                            hot_new = item.get("hot_value")
                            hot_old = existing.get("hot_value")
                            if hot_new is not None and (
                                hot_old is None or hot_new > hot_old
                            ):
                                existing["hot_value"] = hot_new
            merged_list = list(aggregated.values()) if aggregated else news_list
            print(f"合并最近{lookback_hours}小时快照后，共 {len(merged_list)} 条去重新闻")
        except Exception as e:
            print(f"合并历史快照失败，退回使用本次抓取结果: {e}")
            merged_list = news_list

        return {"news_data": {"news_list": merged_list}}

class InsightNode:
    """分析节点"""
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.environ.get("INSIGHT_ENGINE_API_KEY"),
            base_url=os.environ.get("INSIGHT_ENGINE_BASE_URL", "https://api.moonshot.cn/v1"),
            model=os.environ.get("INSIGHT_ENGINE_MODEL_NAME", "moonshot-v1-8k"),
            temperature=0.7
        )

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [InsightNode] 开始分析舆情 ---")
        news_list = state.get("news_data", {}).get("news_list", [])
        if not news_list:
            print("⚠️  警告: 没有新闻数据可分析，跳过分析步骤")
            return {"error": "No news to analyze", "analysis_result": {"top_topics": [], "summary": "今日无新闻数据"}}

        news_text = "\n".join([f"- [{n['source']}] {n['title']}" for n in news_list])
        
        system_prompt = "你是一个资深的舆情分析专家。"
        user_prompt = f"""
        请分析以下一定时间范围内采集到的热点新闻列表：
        {news_text}

        任务（请重点合并同一事件的不同表述，并计算舆情热度值和舆情类别）：
        1. 将高度相关、实际上描述同一事件的不同标题进行合并，例如：
           - “美以联合袭击伊朗”“伊朗反击”“以色列宣布袭击伊朗”“伊朗首都德黑兰爆炸现场”“伊朗局势”等，应归为同一重大事件（如“美伊战争风险升高”）。
        2. 以“事件”为单位，提取最重要的舆情事件，**最多输出 8 个事件**，尽量覆盖不同领域（政治、国际、社会、文娱、科技、财经等），并特别关注如下议题：
           - 台海局势（两岸、台海、台湾相关冲突或选举）
           - 中美关系（中美博弈、贸易科技战、外交摩擦）
           - AI / 人工智能发展与监管
           - 重大社会舆情（群体性事件、公共安全、治安案件等）
           - 危机事件（战争风险、爆炸、重大事故、金融危机等）
        3. 每个事件需要包含以下字段：
           - topic: 事件名称，应当是一个**具体单一事件**的标题，而不是抽象的类别或总结性表述。
             例如：使用“2026年考研国家线公布”“五粮液集团董事长被查”“美以空袭伊朗”等具体事件标题，
             而不要使用“中国经济社会发展成绩单”“重大社会舆情”“文娱类舆论”等泛泛的概念性标题。
           - sentiment: 情感倾向（正面/负面/中性）。
           - comment: 对该事件的简要点评，结合不同标题的信息。
           - heat_score: 舆情热度值（0~100 的整数或一位小数），参考规则：
               a) 同一事件相关的标题数量越多，热度越高；
               b) 单条信息在平台中的排名越靠前（如热搜第1），该标题对热度的贡献越大；
               c) 不同平台赋予权重（从高到低）：微博、百度热搜、抖音、知乎、今日头条、贴吧、澎湃新闻、财联社热门、凤凰网、华尔街见闻，其它平台权重更低。
           - category: 该事件所属舆情类别，必须从以下固定选项中选择其一（使用完整名称）：
             ["经济类舆论", "突发事件舆论", "法治类舆论", "文娱类舆论", "科教类舆论", "国际关系类舆论", "健康类舆论", "治理类舆论", "民生类舆论", "生态环境类舆论", "其他"]
             特别规则（请严格遵守）：
               · 各类考试、招生、分数线、国家线等（如“考研国家线发布”“高考政策调整”）一律归入“科教类舆论”；
               · 人工智能、AI、大模型、算法监管等相关内容一律归入“科教类舆论”，不要归入“其他”或“文娱类舆论”。
        4. 总结整体舆情趋势，给出一个 summary 字段。

        请严格以JSON格式返回，不要包含Markdown代码块标记，格式如下：
        {{
            "top_topics": [
                {{"topic": "话题1（合并后的事件）", "sentiment": "负面", "comment": "点评...", "heat_score": 92.5, "category": "国际关系类舆论"}},
                ...
            ],
            "summary": "整体趋势总结..."
        }}
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            content = response.content
            # 清理可能的 Markdown 标记
            json_str = content.replace("```json", "").replace("```", "").strip()
            analysis_result = json.loads(json_str)
            return {"analysis_result": analysis_result}
        except Exception as e:
            print(f"InsightNode Error: {e}")
            return {"error": str(e), "analysis_result": {"raw": content}}

class ForumNode:
    """论坛讨论节点"""
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.environ.get("QUERY_ENGINE_API_KEY"),
            base_url=os.environ.get("QUERY_ENGINE_BASE_URL", "https://api.moonshot.cn/v1"),
            model=os.environ.get("QUERY_ENGINE_MODEL_NAME", "moonshot-v1-8k"),
            temperature=0.7
        )

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [ForumNode] 开始论坛讨论 ---")
        analysis_result = state.get("analysis_result", {})
        
        topic = "今日热点"
        if "top_topics" in analysis_result and isinstance(analysis_result["top_topics"], list):
            if len(analysis_result["top_topics"]) > 0:
                topic = analysis_result["top_topics"][0].get("topic", "今日热点")
        
        print(f"讨论话题: {topic}")
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是一名资深舆情分析师，擅长从多个视角对重大舆情事件进行结构化剖析，给出清晰、简明的要点总结，而不是还原聊天记录。",
            ),
            ("human", """
            请围绕“{topic}”生成一段 **重大事件剖析** 的总结性内容，而不是对话脚本。

            要求：
            1. 从三个固定视角分别进行小结，每个视角 2-4 句话：
               - Insight（深度观察）: 事件背后的深层原因、潜在影响、结构性变化。
               - Media（媒体观点）: 主流媒体与社交舆论如何报道和解读，整体情绪倾向。
               - Query（关键事实）: 目前已经确认的关键事实、尚存的不确定点。
            2. 在最后再给出一个“综合判断”，用 3-5 句话，说明该事件的核心风险/机遇、未来可能演变方向，以及需要重点关注的人群或领域。
            3. 直接输出一段可阅读的中文分析文本即可，可以使用清晰的小标题（如“【Insight 深度观察】”），但不要使用 Markdown 代码块或列表标记。
            """)
        ])
        
        chain = prompt | self.llm
        try:
            response = chain.invoke({"topic": topic})
            return {"forum_discussion": response.content}
        except Exception as e:
            print(f"ForumNode Error: {e}")
            return {"forum_discussion": "讨论生成失败"}

def render_langgraph_html_report(
    news_list: List[Dict],
    analysis_result: Dict,
    forum_discussion: str,
) -> str:
    """使用与 main.py 一致的 HTML 模板生成舆情报告，保留来源与链接、支持保存为图片"""
    now = get_beijing_time()
    start_time = now - timedelta(hours=12)
    total_news = len(news_list)
    top_topics = analysis_result.get("top_topics") or []
    summary = analysis_result.get("summary") or "暂无总结"
    # 按类别对事件进行分组，便于在深度分析中分栏展示（采用《十大舆情分类》标准）
    category_order = [
        "经济类舆论",
        "突发事件舆论",
        "法治类舆论",
        "文娱类舆论",
        "科教类舆论",
        "国际关系类舆论",
        "健康类舆论",
        "治理类舆论",
        "民生类舆论",
        "生态环境类舆论",
        "其他",
    ]
    topics_by_category: Dict[str, List[Dict[str, Any]]] = {c: [] for c in category_order}
    for t in top_topics:
        cat = t.get("category") or "其他"
        if cat not in topics_by_category:
            cat = "其他"
        topics_by_category[cat].append(t)

    # 按类别的“最大舆情热度”从高到低排序，只展示有热点事件的类别
    def _category_heat(cat: str) -> float:
        topics = topics_by_category.get(cat) or []
        max_heat = 0.0
        for tt in topics:
            try:
                h = float(tt.get("heat_score") or 0)
            except (TypeError, ValueError):
                h = 0.0
            if h > max_heat:
                max_heat = h
        return max_heat

    category_display_order = [c for c in category_order if topics_by_category.get(c)]
    category_display_order.sort(
        key=lambda c: (-_category_heat(c), category_order.index(c))
    )

    # 平台权重与排序（用于原始信息表格 & 热度理解）
    # 这里假定配置中的 source_id 使用这些英文标识；若不匹配则自动归为“其他”
    platform_order = [
        "weibo",          # 微博
        "baidu-hot",      # 百度热搜（示例 ID，需与实际配置对应）
        "douyin",         # 抖音
        "zhihu",          # 知乎
        "toutiao",        # 今日头条
        "tieba",          # 贴吧
        "thepaper",       # 澎湃新闻
        "cls",            # 财联社热门
        "ifeng",          # 凤凰网
        "wallstreetcn",   # 华尔街见闻
    ]
    platform_index = {pid: idx for idx, pid in enumerate(platform_order)}

    # 按来源分组新闻（同时保留 source_id，便于排序和权重）
    platforms: Dict[str, Dict[str, Any]] = {}
    for n in news_list:
        source_id = n.get("source_id") or n.get("source") or "other"
        source_name = n.get("source_name") or n.get("source") or source_id
        if source_id not in platforms:
            platforms[source_id] = {
                "name": source_name,
                "items": []
            }
        platforms[source_id]["items"].append(n)

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>舆情日报 - 热点新闻分析</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js" integrity="sha512-BNaRQnYJYiPSqHHDb58B0yaPfCu+Wgds8Gp/gU33kqBtgNS4tSPHuGibyoeqMV/TJlSKda6FXzoEyYGjTe+vXA==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 16px; background: #fafafa; color: #333; line-height: 1.5; }
        .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 16px rgba(0,0,0,0.06); }
        .header { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: white; padding: 32px 24px; text-align: center; position: relative; }
        .save-buttons { position: absolute; top: 16px; right: 16px; display: flex; gap: 8px; }
        .save-btn { background: rgba(255, 255, 255, 0.2); border: 1px solid rgba(255, 255, 255, 0.3); color: white; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.2s ease; backdrop-filter: blur(10px); white-space: nowrap; }
        .save-btn:hover { background: rgba(255, 255, 255, 0.3); border-color: rgba(255, 255, 255, 0.5); transform: translateY(-1px); }
        .save-btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .header-title { font-size: 22px; font-weight: 700; margin: 0 0 20px 0; }
        .header-info { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-size: 14px; opacity: 0.95; }
        .info-item { text-align: center; }
        .info-label { display: block; font-size: 12px; opacity: 0.8; margin-bottom: 4px; }
        .info-value { font-weight: 600; font-size: 16px; }
        .content { padding: 24px; }
        .section { margin-bottom: 32px; }
        .section:last-child { margin-bottom: 0; }
        .section-title { font-size: 16px; font-weight: 600; color: #1a1a1a; margin: 0 0 16px 0; padding-bottom: 8px; border-bottom: 1px solid #f0f0f0; }
        .section-body { font-size: 14px; color: #374151; line-height: 1.6; white-space: pre-wrap; }
        .topic-category { margin-bottom: 18px; font-size: 14px; font-weight: 600; color: #4b5563; }
        .topic-item { margin-bottom: 16px; padding: 12px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #4f46e5; }
        .topic-name { font-weight: 600; color: #1e293b; margin-bottom: 4px; }
        .topic-meta { font-size: 12px; color: #64748b; margin-bottom: 6px; }
        .topic-comment { font-size: 13px; color: #475569; }
        .source-group { margin-bottom: 24px; }
        .source-title { color: #666; font-size: 13px; font-weight: 600; margin: 0 0 12px 0; padding-bottom: 6px; border-bottom: 1px solid #f5f5f5; }
        .news-item { margin-bottom: 16px; padding: 12px 0; border-bottom: 1px solid #f5f5f5; display: flex; gap: 12px; align-items: flex-start; }
        .news-item:last-child { border-bottom: none; }
        .news-num { color: #999; font-size: 12px; font-weight: 600; min-width: 20px; text-align: center; flex-shrink: 0; background: #f1f5f9; border-radius: 50%; width: 22px; height: 22px; display: flex; align-items: center; justify-content: center; }
        .news-content { flex: 1; min-width: 0; }
        .news-link { color: #2563eb; text-decoration: none; }
        .news-link:hover { text-decoration: underline; }
        .news-link:visited { color: #7c3aed; }
        .news-source { color: #666; font-size: 12px; margin-bottom: 4px; }
        .news-hot { font-size: 11px; color: #dc2626; font-weight: 500; }
        .raw-table-wrapper { overflow-x: auto; }
        .raw-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .raw-table thead { background: #f1f5f9; }
        .raw-table th, .raw-table td { padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; }
        .raw-table th { color: #4b5563; font-weight: 600; font-size: 12px; }
        .raw-table tbody tr:hover { background: #f9fafb; }
        .raw-table .platform-cell { white-space: nowrap; color: #374151; font-weight: 500; }
        .raw-table .rank-cell { width: 60px; color: #6b7280; }
        .raw-table .title-cell { color: #111827; }
        .raw-table .hot-badge { margin-left: 6px; font-size: 11px; color: #b91c1c; }
        .footer { margin-top: 24px; padding: 20px 24px; background: #f8f9fa; border-top: 1px solid #e5e7eb; text-align: center; }
        .footer-content { font-size: 13px; color: #6b7280; line-height: 1.6; }
        .project-name { font-weight: 600; color: #374151; }
        @media (max-width: 480px) { body { padding: 12px; } .header { padding: 24px 20px; } .content { padding: 20px; } .header-info { grid-template-columns: 1fr; } .save-buttons { position: static; margin-bottom: 16px; justify-content: center; flex-direction: column; width: 100%; } .save-btn { width: 100%; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="save-buttons">
                <button class="save-btn" onclick="saveAsImage()">保存为图片</button>
                <button class="save-btn" onclick="saveAsMultipleImages()">分段保存</button>
            </div>
            <div class="header-title">舆情日报 · 热点新闻分析</div>
            <div class="header-info">
                <div class="info-item"><span class="info-label">报告类型</span><span class="info-value">舆情分析</span></div>
                <div class="info-item"><span class="info-label">新闻总数</span><span class="info-value">""" + str(total_news) + """ 条</span></div>
                <div class="info-item"><span class="info-label">热点话题</span><span class="info-value">""" + str(len(top_topics)) + """ 个</span></div>
                <div class="info-item"><span class="info-label">时间范围</span><span class="info-value">""" + start_time.strftime("%m-%d %H:%M") + " ~ " + now.strftime("%m-%d %H:%M") + """</span></div>
            </div>
        </div>
        <div class="content">
            <div class="section">
                <div class="section-title">今日热点概览</div>
                <div class="section-body">""" + html_escape(summary) + """</div>
            </div>
            <div class="section">
                <div class="section-title">深度分析</div>
"""

    # 按类别分组展示深度分析事件（类别按热度排序，只有有热点的类别才展示）
    for cat in category_display_order:
        cat_topics = topics_by_category.get(cat) or []
        html += '<div class="topic-category">' + html_escape(cat) + '</div>'
        for t in cat_topics:
            topic = t.get("topic") or ""
            sentiment = t.get("sentiment") or ""
            comment = t.get("comment") or ""
            heat_score = t.get("heat_score")
            html += '<div class="topic-item"><div class="topic-name">' + html_escape(topic) + '</div>'
            meta_parts: List[str] = []
            if sentiment:
                meta_parts.append("情感: " + sentiment)
            if heat_score is not None:
                meta_parts.append("舆情热度: " + str(heat_score))
            if meta_parts:
                html += '<div class="topic-meta">' + html_escape(" · ".join(meta_parts)) + '</div>'
            html += '<div class="topic-comment">' + html_escape(comment) + '</div></div>'

    html += """
            </div>
            <div class="section">
                <div class="section-title">重大事件剖析</div>
                <div class="section-body">""" + html_escape(forum_discussion or "暂无重大事件剖析") + """</div>
            </div>
            <div class="section">
                <div class="section-title">原始信息抓取（来源与链接）</div>
                <div class="raw-table-wrapper">
                    <table class="raw-table">
                        <thead>
                            <tr>
                                <th>平台</th>
                                <th>排行</th>
                                <th>信息 & 链接</th>
                            </tr>
                        </thead>
                        <tbody>
"""

    # 将所有新闻按平台顺序和平台内排行排序，生成统一列表
    # 平台顺序：微博、百度热搜、抖音、知乎、今日头条、贴吧、澎湃新闻、财联社热门、凤凰网、华尔街见闻，其它平台排在最后
    def _platform_sort_key(item: Dict[str, Any]) -> int:
        sid = item.get("source_id") or item.get("source") or "other"
        return platform_index.get(sid, len(platform_index))

    rows: List[Dict[str, Any]] = []
    for sid, info in platforms.items():
        for n in info["items"]:
            n_copy = dict(n)
            # 确保包含 ID 与展示名
            n_copy["source_id"] = sid
            n_copy["source_name"] = info["name"]
            rows.append(n_copy)

    rows.sort(key=lambda n: (_platform_sort_key(n), n.get("rank") or 9999))

    for n in rows:
        platform_name = n.get("source_name") or n.get("source") or "其他"
        rank = n.get("rank")
        rank_text = str(rank) if rank is not None else "-"
        title = n.get("title") or ""
        link_url = n.get("mobile_url") or n.get("url") or ""
        hot_val = n.get("hot_value")

        html += '<tr>'
        html += '<td class="platform-cell">' + html_escape(platform_name) + '</td>'
        html += '<td class="rank-cell">' + html_escape(rank_text) + '</td>'
        html += '<td class="title-cell">'
        if link_url:
            html += '<a href="' + html_escape(link_url) + '" target="_blank" class="news-link">' + html_escape(title) + '</a>'
        else:
            html += html_escape(title)
        if hot_val is not None and hot_val != 0:
            html += '<span class="hot-badge">热度 ' + html_escape(str(hot_val)) + '</span>'
        html += '</td></tr>'

    html += """
                        </tbody>
                    </table>
                </div>
            </div>
"""

    html += """
            </div>
        </div>
        <div class="footer">
            <div class="footer-content">
                由 <span class="project-name">BJTU舆情实验室</span> 生成
            </div>
        </div>
    </div>
    <script>
        async function saveAsImage() {
            const button = event.target;
            const originalText = button.textContent;
            try {
                button.textContent = '生成中...';
                button.disabled = true;
                window.scrollTo(0, 0);
                await new Promise(r => setTimeout(r, 200));
                const buttons = document.querySelector('.save-buttons');
                buttons.style.visibility = 'hidden';
                await new Promise(r => setTimeout(r, 100));
                const container = document.querySelector('.container');
                const canvas = await html2canvas(container, { backgroundColor: '#ffffff', scale: 1.5, useCORS: true, allowTaint: false, imageTimeout: 10000, logging: false });
                buttons.style.visibility = 'visible';
                const link = document.createElement('a');
                const now = new Date();
                link.download = 'TrendRadar_舆情日报_' + now.getFullYear() + String(now.getMonth()+1).padStart(2,'0') + String(now.getDate()).padStart(2,'0') + '_' + String(now.getHours()).padStart(2,'0') + String(now.getMinutes()).padStart(2,'0') + '.png';
                link.href = canvas.toDataURL('image/png', 1.0);
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                button.textContent = '保存成功!';
                setTimeout(function(){ button.textContent = originalText; button.disabled = false; }, 2000);
            } catch (e) {
                const buttons = document.querySelector('.save-buttons');
                if (buttons) buttons.style.visibility = 'visible';
                button.textContent = '保存失败';
                setTimeout(function(){ button.textContent = originalText; button.disabled = false; }, 2000);
            }
        }
        async function saveAsMultipleImages() {
            const button = event.target;
            const originalText = button.textContent;
            const container = document.querySelector('.container');
            const scale = 1.5;
            const maxHeight = 5000 / scale;
            try {
                button.textContent = '分析中...';
                button.disabled = true;
                const sections = container.querySelectorAll('.section');
                const header = container.querySelector('.header');
                const footer = container.querySelector('.footer');
                const buttons = document.querySelector('.save-buttons');
                buttons.style.visibility = 'hidden';
                const images = [];
                let seg = 0;
                for (let i = 0; i < sections.length; i++) {
                    button.textContent = '生成中 (' + (i+1) + '/' + sections.length + ')...';
                    const temp = document.createElement('div');
                    temp.className = 'container';
                    temp.style.cssText = 'position:absolute;left:-9999px;top:0;width:' + container.offsetWidth + 'px;background:white;';
                    temp.appendChild(header.cloneNode(true));
                    temp.appendChild(sections[i].cloneNode(true));
                    temp.appendChild(footer.cloneNode(true));
                    document.body.appendChild(temp);
                    await new Promise(r => setTimeout(r, 100));
                    const canvas = await html2canvas(temp, { backgroundColor: '#ffffff', scale: scale, useCORS: true, logging: false });
                    document.body.removeChild(temp);
                    images.push(canvas.toDataURL('image/png', 1.0));
                }
                buttons.style.visibility = 'visible';
                const now = new Date();
                const base = 'TrendRadar_舆情日报_' + now.getFullYear() + String(now.getMonth()+1).padStart(2,'0') + String(now.getDate()).padStart(2,'0') + '_' + String(now.getHours()).padStart(2,'0') + String(now.getMinutes()).padStart(2,'0');
                for (let i = 0; i < images.length; i++) {
                    const a = document.createElement('a');
                    a.download = base + '_part' + (i+1) + '.png';
                    a.href = images[i];
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    await new Promise(r => setTimeout(r, 100));
                }
                button.textContent = '已保存 ' + images.length + ' 张图片!';
                setTimeout(function(){ button.textContent = originalText; button.disabled = false; }, 2000);
            } catch (e) {
                const buttons = document.querySelector('.save-buttons');
                if (buttons) buttons.style.visibility = 'visible';
                button.textContent = '保存失败';
                setTimeout(function(){ button.textContent = originalText; button.disabled = false; }, 2000);
            }
        }
        document.addEventListener('DOMContentLoaded', function(){ window.scrollTo(0, 0); });
    </script>
</body>
</html>"""
    return html


class ReportNode:
    """报告生成节点：使用与 main.py 一致的 HTML 模板，保留来源与链接，支持保存为图片与浏览器打开"""
    def __init__(self):
        pass

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [ReportNode] 开始生成报告 ---")
        analysis_result = state.get("analysis_result", {})
        news_data = state.get("news_data", {})
        discussion = state.get("forum_discussion", "")

        news_list = news_data.get("news_list") or []

        if state.get("error") or not news_list:
            print("⚠️  警告: 没有新闻数据，生成空报告")
            empty_html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>舆情日报 - 无数据</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1 class="text-center">舆情日报</h1>
        <div class="alert alert-warning mt-4">
            <h4>⚠️ 今日无新闻数据</h4>
            <p>可能的原因：</p>
            <ul>
                <li>API 服务暂时不可用</li>
                <li>网络连接问题</li>
                <li>所有平台均未返回数据</li>
            </ul>
            <p class="mb-0">请检查网络连接或稍后重试。</p>
        </div>
    </div>
</body>
</html>"""
            return {"html_report": empty_html}

        html_content = render_langgraph_html_report(news_list, analysis_result, discussion)
        return {"html_report": html_content}

# === 构建图 ===
def build_graph():
    workflow = StateGraph(TrendState)

    # 添加节点
    workflow.add_node("spider", SpiderNode())
    workflow.add_node("insight", InsightNode())
    workflow.add_node("forum", ForumNode())
    workflow.add_node("report", ReportNode())

    # 定义边
    workflow.set_entry_point("spider")
    workflow.add_edge("spider", "insight")
    workflow.add_edge("insight", "forum")
    workflow.add_edge("forum", "report")
    workflow.add_edge("report", END)

    return workflow.compile()

# === 主程序入口 ===
def main():
    print("=== TrendRadar LangGraph 版启动 ===")
    
    app = build_graph()
    
    # 初始状态
    initial_state = {
        "messages": [],
        "news_data": {},
        "analysis_result": {},
        "forum_discussion": "",
        "html_report": "",
        "error": None
    }
    
    # 执行工作流
    try:
        final_state = app.invoke(initial_state)

        if final_state.get("html_report"):
            output_dir = Path("output_langgraph")
            ensure_directory_exists(str(output_dir))
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            output_path = output_dir / filename

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_state["html_report"])

            # 同时写入 index.html，便于直接打开目录时查看最新报告（与 main.py 行为一致）
            index_path = output_dir / "index.html"
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(final_state["html_report"])

            print(f"\n✅ 流程执行成功！报告已保存: {output_path}")

            # 非 Docker 环境下自动在浏览器中打开报告（与 main.py 一致）
            if not _is_docker_env():
                file_url = "file://" + str(output_path.resolve())
                print(f"正在打开报告: {file_url}")
                try:
                    webbrowser.open(file_url)
                except Exception as e:
                    print(f"自动打开浏览器失败: {e}，请手动打开: {output_path}")
            else:
                print(f"（Docker 环境不自动打开浏览器）报告路径: {output_path}")
        else:
            print("\n❌ 流程执行完成，但未生成报告内容。")
            if final_state.get("error"):
                print(f"错误信息: {final_state['error']}")

    except Exception as e:
        print(f"\n❌ 流程执行异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
