# coding=utf-8
import json
import os
import time
import webbrowser
import yaml
import re
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


# === 数据存储和加载工具函数 ===
def get_hourly_data_dir() -> Path:
    """获取按小时存储数据的目录"""
    data_dir = Path("data_langgraph_hourly")
    ensure_directory_exists(str(data_dir))
    return data_dir


def save_hourly_data(
    platform_id: str,
    platform_name: str,
    items: List[Dict[str, Any]],
    timestamp: Optional[datetime] = None,
) -> Path:
    """保存每小时的数据到文件"""
    if timestamp is None:
        timestamp = get_beijing_time()

    data_dir = get_hourly_data_dir()
    # 按日期和小时组织目录结构：YYYYMMDD/HH/
    date_str = timestamp.strftime("%Y%m%d")
    hour_str = timestamp.strftime("%H")
    hour_dir = data_dir / date_str / hour_str
    ensure_directory_exists(str(hour_dir))

    # 文件名：platform_id_timestamp.json
    filename = f"{platform_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    file_path = hour_dir / filename

    data = {
        "platform_id": platform_id,
        "platform_name": platform_name,
        "timestamp": timestamp.isoformat(),
        "items": items,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return file_path


def load_past_hours_data(lookback_hours: int = 12) -> List[Dict[str, Any]]:
    """加载过去N小时的所有平台数据"""
    data_dir = get_hourly_data_dir()
    now = get_beijing_time()
    cutoff = now - timedelta(hours=lookback_hours)

    all_items: List[Dict[str, Any]] = []

    # 遍历所有日期和小时目录
    for date_dir in sorted(data_dir.glob("20*"), reverse=True):
        date_str = date_dir.name
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            continue

        for hour_dir in sorted(date_dir.glob("*"), reverse=True):
            hour_str = hour_dir.name
            try:
                hour_obj = int(hour_str)
                if hour_obj < 0 or hour_obj > 23:
                    continue
            except ValueError:
                continue

            # 构建完整时间戳
            file_timestamp = datetime.combine(date_obj.date(), datetime.min.time().replace(hour=hour_obj))
            file_timestamp = pytz.timezone("Asia/Shanghai").localize(file_timestamp)

            if file_timestamp < cutoff:
                continue

            # 加载该小时目录下的所有文件
            for json_file in hour_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        items = data.get("items", [])
                        # 为每个item添加时间戳信息
                        for item in items:
                            item["_fetch_timestamp"] = data.get("timestamp")
                            item["_platform_id"] = data.get("platform_id")
                            item["_platform_name"] = data.get("platform_name")
                        all_items.extend(items)
                except Exception as e:
                    print(f"加载文件失败 {json_file}: {e}")

    return all_items


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

    news_data: Dict[str, Any]
    raw_items: Annotated[List[Dict[str, Any]], operator.add]
    analysis_result: Dict[str, Any]
    classification_stats: Dict[str, Any]
    forum_discussion: str
    html_report: str
    messages: Annotated[List[BaseMessage], operator.add]
    error: Optional[str]
    platform_summaries: Dict[str, Any]


# === 节点定义 ===
class BaseFetchNode:
    """基础抓取节点"""

    def __init__(self, platform_id: str, platform_name: str):
        self.platform_id = platform_id
        self.platform_name = platform_name

    def fetch_data(self) -> Optional[str]:
        """抓取数据"""
        url = f"https://newsnow.busiyi.world/api/s?id={self.platform_id}&latest"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://newsnow.busiyi.world/",
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            print(f"Error fetching {self.platform_name} ({self.platform_id}): {e}")
        return None

    def parse_data(self, raw_data: str) -> List[Dict[str, Any]]:
        """解析数据"""
        items = []
        try:
            data = json.loads(raw_data)
            raw_items = data.get("items", [])
            top_items = raw_items[:10]
            for idx, item in enumerate(top_items, 1):
                items.append(
                    {
                        "source_id": self.platform_id,
                        "source": self.platform_name,
                        "source_name": self.platform_name,
                        "title": item.get("title"),
                        "rank": idx,
                        "url": item.get("url", ""),
                        "mobile_url": item.get("mobileUrl", ""),
                        "hot_value": item.get("hotValue", 0),
                    }
                )
        except Exception as e:
            print(f"解析 {self.platform_name} 数据失败: {e}")
        return items

    def __call__(self, state: TrendState) -> TrendState:
        """执行抓取"""
        print(f"--- [Fetch{self.platform_name}Node] 开始抓取 {self.platform_name} ---")
        timestamp = get_beijing_time()

        raw_data = self.fetch_data()
        if not raw_data:
            print(f"⚠️  警告: {self.platform_name} 抓取失败")
            return {"raw_items": []}

        items = self.parse_data(raw_data)
        print(f"✅ {self.platform_name} 抓取完成，共 {len(items)} 条新闻")

        try:
            save_path = save_hourly_data(self.platform_id, self.platform_name, items, timestamp)
            print(f"   数据已保存: {save_path}")
        except Exception as e:
            print(f"   保存数据失败: {e}")

        return {"raw_items": items}


class FetchWeiboNode(BaseFetchNode):
    def __init__(self):
        super().__init__("weibo", "微博")


class FetchZhihuNode(BaseFetchNode):
    def __init__(self):
        super().__init__("zhihu", "知乎")


class FetchToutiaoNode(BaseFetchNode):
    def __init__(self):
        super().__init__("toutiao", "今日头条")


class FetchBaiduNode(BaseFetchNode):
    def __init__(self):
        super().__init__("baidu", "百度热搜")


class FetchDouyinNode(BaseFetchNode):
    def __init__(self):
        super().__init__("douyin", "抖音")


class FetchBilibiliNode(BaseFetchNode):
    def __init__(self):
        super().__init__("bilibili-hot-search", "bilibili 热搜")


class FetchThepaperNode(BaseFetchNode):
    def __init__(self):
        super().__init__("thepaper", "澎湃新闻")


class FetchTiebaNode(BaseFetchNode):
    def __init__(self):
        super().__init__("tieba", "贴吧")


class FetchIfengNode(BaseFetchNode):
    def __init__(self):
        super().__init__("ifeng", "凤凰网")


class FetchClsNode(BaseFetchNode):
    def __init__(self):
        super().__init__("cls-hot", "财联社热门")


class FetchWallstreetcnNode(BaseFetchNode):
    def __init__(self):
        super().__init__("wallstreetcn-hot", "华尔街见闻")


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
            "sspai": "少数派",
        }
        if CONFIG and "platforms" in CONFIG:
            self.sources = {p["id"]: p.get("name", p["id"]) for p in CONFIG["platforms"]}

    def fetch_data(self, id_value: str) -> Optional[str]:
        url = f"https://newsnow.busiyi.world/api/s?id={id_value}&latest"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://newsnow.busiyi.world/",
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
                    top_items = items[:10]
                    for idx, item in enumerate(top_items, 1):
                        news_list.append(
                            {
                                "source_id": source_id,
                                "source": source_name,
                                "source_name": source_name,
                                "title": item.get("title"),
                                "rank": idx,
                                "url": item.get("url", ""),
                                "mobile_url": item.get("mobileUrl", ""),
                                "hot_value": item.get("hotValue", 0),
                            }
                        )
                except Exception as e:
                    print(f"解析 {source_name} 失败: {e}")
            time.sleep(1)

        print(f"本次抓取完成，共 {len(news_list)} 条新闻")
        if len(news_list) == 0:
            print("⚠️  警告: 未能抓取到任何新闻，可能是 API 服务问题或网络连接问题")

        try:
            snapshot_dir = Path("data_langgraph")
            ensure_directory_exists(str(snapshot_dir))
            timestamp = get_beijing_time()
            snapshot_name = f"snapshot_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            snapshot_path = snapshot_dir / snapshot_name
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump({"timestamp": timestamp.isoformat(), "items": news_list}, f, ensure_ascii=False)
        except Exception as e:
            print(f"保存快照失败: {e}")

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
                        key = (item.get("source_id") or item.get("source") or "", item.get("title") or "")
                        if not key[1]:
                            continue
                        existing = aggregated.get(key)
                        if existing is None:
                            aggregated[key] = dict(item)
                        else:
                            rank_new = item.get("rank")
                            rank_old = existing.get("rank")
                            if rank_new is not None and (rank_old is None or rank_new < rank_old):
                                existing["rank"] = rank_new
                            hot_new = item.get("hot_value")
                            hot_old = existing.get("hot_value")
                            if hot_new is not None and (hot_old is None or hot_new > hot_old):
                                existing["hot_value"] = hot_new
            merged_list = list(aggregated.values()) if aggregated else news_list
            print(f"合并最近{lookback_hours}小时快照后，共 {len(merged_list)} 条去重新闻")
        except Exception as e:
            print(f"合并历史快照失败，退回使用本次抓取结果: {e}")
            merged_list = news_list

        return {"news_data": {"news_list": merged_list}}


class NormalizeNewsNode:
    """清洗、去重、排序节点"""

    def __init__(self):
        pass

    def clean_title(self, title: str) -> str:
        """清洗标题：去除多余空格、特殊字符等"""
        if not title:
            return ""
        title = title.strip()
        title = re.sub(r"\s+", " ", title)
        return title

    def normalize_news(self, raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清洗、去重、排序新闻"""
        if not raw_items:
            return []

        cleaned_items = []
        for item in raw_items:
            title = item.get("title", "")
            cleaned_title = self.clean_title(title)
            if cleaned_title:
                item_copy = dict(item)
                item_copy["title"] = cleaned_title
                cleaned_items.append(item_copy)

        seen = {}
        deduplicated_items = []
        for item in cleaned_items:
            key = (item.get("source_id") or item.get("source") or "", item.get("title") or "")
            if not key[1]:
                continue

            existing = seen.get(key)
            if existing is None:
                seen[key] = item
                deduplicated_items.append(item)
            else:
                rank_new = item.get("rank")
                rank_old = existing.get("rank")
                if rank_new is not None and (rank_old is None or rank_new < rank_old):
                    existing["rank"] = rank_new

                hot_new = item.get("hot_value")
                hot_old = existing.get("hot_value")
                if hot_new is not None and (hot_old is None or hot_new > hot_old):
                    existing["hot_value"] = hot_new

        platform_order = ["weibo", "baidu", "douyin", "zhihu", "toutiao", "tieba", "thepaper", "cls-hot", "ifeng", "wallstreetcn-hot"]
        platform_index = {pid: idx for idx, pid in enumerate(platform_order)}

        platform_groups = {}
        for item in deduplicated_items:
            sid = item.get("source_id") or item.get("source") or "other"
            if sid not in platform_groups:
                platform_groups[sid] = []
            platform_groups[sid].append(item)

        final_items = []
        for sid, items in platform_groups.items():
            items.sort(key=lambda x: x.get("rank") or 9999)
            final_items.extend(items[:10])

        def sort_key(item: Dict[str, Any]) -> tuple:
            source_id = item.get("source_id") or item.get("source") or "other"
            platform_rank = platform_index.get(source_id, len(platform_index))
            item_rank = item.get("rank") or 9999
            return (platform_rank, item_rank)

        sorted_items = sorted(final_items, key=sort_key)
        return sorted_items

    def __call__(self, state: TrendState) -> TrendState:
        """执行清洗、去重、排序"""
        print("--- [NormalizeNewsNode] 开始清洗、去重、排序 ---")

        news_data = state.get("news_data", {})
        raw_items = news_data.get("raw_items", [])

        print("正在加载过去12小时的历史数据...")
        historical_items = load_past_hours_data(lookback_hours=12)
        print(f"从历史数据中加载了 {len(historical_items)} 条新闻")

        all_items = raw_items + historical_items
        normalized_items = self.normalize_news(all_items)

        print(f"✅ 清洗、去重、排序完成，共 {len(normalized_items)} 条新闻")

        return {"news_data": {"news_list": normalized_items, "raw_items": []}, "raw_items": []}


class StartFetchNode:
    def __init__(self):
        pass

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [StartFetchNode] 开始并行抓取所有平台 ---")
        return {"raw_items": []}


class MergeFetchNode:
    def __init__(self):
        pass

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [MergeFetchNode] 合并所有抓取结果 ---")
        raw_items = state.get("raw_items", [])
        print(f"✅ 所有平台抓取完成，共 {len(raw_items)} 条原始新闻")
        return {"news_data": {"raw_items": raw_items}}


class InsightNode:
    """分析节点"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.environ.get("INSIGHT_ENGINE_API_KEY"),
            base_url=os.environ.get("INSIGHT_ENGINE_BASE_URL", "https://api.moonshot.cn/v1"),
            model=os.environ.get("INSIGHT_ENGINE_MODEL_NAME", "moonshot-v1-8k"),
            temperature=0.7,
        )

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [InsightNode] 开始分析舆情 ---")
        news_list = state.get("news_data", {}).get("news_list", [])
        if not news_list:
            print("⚠️  警告: 没有新闻数据可分析，跳过分析步骤")
            return {"error": "No news to analyze", "analysis_result": {"top_topics": [], "summary": "今日无新闻数据"}}

        def news_priority_score(news_item: Dict[str, Any]) -> float:
            source_id = news_item.get("source_id", "").lower()
            rank = news_item.get("rank", 9999)
            hot_value = news_item.get("hot_value", 0)

            platform_weights = {
                "weibo": 10.0, "baidu": 9.0, "douyin": 8.0, "zhihu": 7.0,
                "toutiao": 6.0, "tieba": 5.0, "thepaper": 4.0, "cls-hot": 3.0,
                "ifeng": 2.0, "wallstreetcn-hot": 1.0,
            }
            platform_weight = platform_weights.get(source_id, 0.5)
            rank_weight = max(0, 30 - rank) / 30.0
            hot_weight = min(1.0, hot_value / 1000000.0) if hot_value else 0

            score = platform_weight * 0.5 + rank_weight * 0.3 + hot_weight * 0.2
            return score

        sorted_news = sorted(news_list, key=news_priority_score, reverse=True)
        max_news_count = 120
        selected_news = sorted_news[:max_news_count]

        if len(news_list) > max_news_count:
            print(f"📊 从 {len(news_list)} 条新闻中筛选出前 {max_news_count} 条高优先级新闻进行分析")

        news_text = "\n".join([f"- {n['title']}" for n in selected_news])

        system_prompt = "你是一个专业的舆情分析助手，擅长从新闻标题中提取热点事件并进行分类分析。"
        user_prompt = f"""
请分析以下热点新闻标题列表，提取最重要的舆情事件：

{news_text}

任务要求：
1. 合并描述同一事件的不同标题，提取核心事件
2. 输出最多8个最重要的事件，覆盖不同领域
3. 每个事件包含字段：
   - topic: 具体事件标题（不要使用抽象类别）
   - sentiment: 情感倾向（正面/负面/中性）
   - comment: 简要点评（1-2句话）
   - heat_score: 热度值（0-100，基于相关标题数量和重要性）
   - category: 类别（从以下选项选择）：
     ["经济类舆论", "突发事件舆论", "法治类舆论", "文娱类舆论", "科教类舆论", "国际关系类舆论", "健康类舆论", "治理类舆论", "民生类舆论", "生态环境类舆论", "其他"]
4. 提供整体趋势summary（2-3句话）

请以JSON格式返回（不要使用Markdown代码块）：
{{
    "top_topics": [
        {{"topic": "事件标题", "sentiment": "负面", "comment": "点评", "heat_score": 85.0, "category": "类别"}},
        ...
    ],
    "summary": "趋势总结"
}}
"""

        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            time.sleep(0.5)
            response = self.llm.invoke(messages)
            content = response.content

            json_str = content.replace("```json", "").replace("```", "").strip()
            try:
                analysis_result = json.loads(json_str)
            except json.JSONDecodeError as json_err:
                print(f"⚠️  JSON解析失败，尝试修复: {json_err}")
                try:
                    topics_match = re.search(r'"top_topics"\s*:\s*\[', json_str)
                    if topics_match:
                        start_pos = topics_match.end()
                        complete_objects = []
                        brace_count = 0
                        in_string = False
                        escape_next = False
                        obj_start = None

                        for i in range(start_pos, len(json_str)):
                            char = json_str[i]
                            if escape_next:
                                escape_next = False
                                continue
                            if char == "\\":
                                escape_next = True
                                continue
                            if char == '"' and not escape_next:
                                in_string = not in_string
                                continue
                            if not in_string:
                                if char == "{":
                                    if brace_count == 0:
                                        obj_start = i
                                    brace_count += 1
                                elif char == "}":
                                    brace_count -= 1
                                    if brace_count == 0 and obj_start is not None:
                                        complete_objects.append((obj_start, i + 1))
                                        obj_start = None

                        if complete_objects:
                            obj_texts = [json_str[start:end] for start, end in complete_objects]
                            partial_json = '{"top_topics": [' + ",".join(obj_texts) + '], "summary": "JSON响应被截断，仅提取了部分数据"}'
                            try:
                                analysis_result = json.loads(partial_json)
                                print(f"⚠️  JSON被截断，已修复并提取了 {len(analysis_result.get('top_topics', []))} 个话题")
                            except Exception as parse_err:
                                raise json_err
                        else:
                            raise json_err
                    else:
                        raise json_err
                except Exception as fix_err:
                    print(f"⚠️  JSON解析失败且无法修复: {json_err}")
                    summary_text = "分析过程中JSON响应被截断，无法完整解析"
                    if "summary" in json_str.lower():
                        summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', json_str)
                        if summary_match:
                            summary_text = summary_match.group(1) + " (部分数据)"
                    analysis_result = {"top_topics": [], "summary": summary_text}

            return {"analysis_result": analysis_result}
        except Exception as e:
            print(f"InsightNode Error: {e}")
            if "content" in locals():
                return {"error": str(e), "analysis_result": {"raw": content}}
            else:
                return {"error": str(e), "analysis_result": {"raw": None, "error_detail": "LLM response not received"}}


class ClassifyNode:
    """十一类舆情分类统计节点"""

    def __init__(self):
        self.category_order = [
            "经济类舆论", "突发事件舆论", "法治类舆论", "文娱类舆论", "科教类舆论",
            "国际关系类舆论", "健康类舆论", "治理类舆论", "民生类舆论", "生态环境类舆论", "其他",
        ]

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [ClassifyNode] 开始十一类舆情分类统计 ---")
        analysis_result = state.get("analysis_result", {})
        top_topics = analysis_result.get("top_topics") or []

        if not top_topics:
            return {
                "classification_stats": {
                    "topics_by_category": {c: [] for c in self.category_order},
                    "category_display_order": [],
                    "category_heat_map": {c: 0.0 for c in self.category_order},
                }
            }

        topics_by_category: Dict[str, List[Dict[str, Any]]] = {c: [] for c in self.category_order}
        for t in top_topics:
            cat = t.get("category") or "其他"
            if cat not in topics_by_category:
                cat = "其他"
            topics_by_category[cat].append(t)

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

        category_heat_map = {cat: _category_heat(cat) for cat in self.category_order}
        category_display_order = [c for c in self.category_order if topics_by_category.get(c)]
        category_display_order.sort(key=lambda c: (-category_heat_map[c], self.category_order.index(c)))

        return {
            "classification_stats": {
                "topics_by_category": topics_by_category,
                "category_display_order": category_display_order,
                "category_heat_map": category_heat_map,
            }
        }


class ForumNode:
    """论坛讨论节点"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.environ.get("QUERY_ENGINE_API_KEY"),
            base_url=os.environ.get("QUERY_ENGINE_BASE_URL", "https://api.moonshot.cn/v1"),
            model=os.environ.get("QUERY_ENGINE_MODEL_NAME", "moonshot-v1-8k"),
            temperature=0.7,
        )

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [ForumNode] 开始论坛讨论 ---")
        analysis_result = state.get("analysis_result", {})

        topic = "今日热点"
        if "top_topics" in analysis_result and isinstance(analysis_result["top_topics"], list):
            if len(analysis_result["top_topics"]) > 0:
                topic = analysis_result["top_topics"][0].get("topic", "今日热点")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一名资深舆情分析师，擅长从多个视角对重大舆情事件进行结构化剖析，给出清晰、简明的要点总结，而不是还原聊天记录。",
                ),
                (
                    "human",
                    """
            请围绕“{topic}”生成一段 **重大事件剖析** 的总结性内容，而不是对话脚本。

            要求：
            1. 从三个固定视角分别进行小结，每个视角 2-4 句话：
               - Insight（深度观察）: 事件背后的深层原因、潜在影响、结构性变化。
               - Media（媒体观点）: 主流媒体与社交舆论如何报道和解读，整体情绪倾向。
               - Query（关键事实）: 目前已经确认的关键事实、尚存的不确定点。
            2. 在最后再给出一个“综合判断”，用 3-5 句话，说明该事件的核心风险/机遇、未来可能演变方向，以及需要重点关注的人群或领域。
            3. 直接输出一段可阅读的中文分析文本即可，可以使用清晰的小标题（如“【Insight 深度观察】”），但不要使用 Markdown 代码块或列表标记。
            """,
                ),
            ]
        )

        chain = prompt | self.llm
        try:
            response = chain.invoke({"topic": topic})
            return {"forum_discussion": response.content}
        except Exception as e:
            return {"forum_discussion": "讨论生成失败"}


class PlatformSummaryNode:
    """各平台关键词与精选评论生成节点（加强防报错版）"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.environ.get("INSIGHT_ENGINE_API_KEY"),
            base_url=os.environ.get("INSIGHT_ENGINE_BASE_URL", "https://api.siliconflow.cn/v1"),
            model=os.environ.get("INSIGHT_ENGINE_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
            temperature=0.7,
        )

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [PlatformSummaryNode] 生成各平台关键词与精选评论 ---")
        news_list = state.get("news_data", {}).get("news_list", [])
        if not news_list:
            return {"platform_summaries": {}}

        # 分平台整理 Top 10 标题
        platform_groups = {}
        for item in news_list:
            sid = item.get("source_id") or item.get("source") or "other"
            if sid not in platform_groups:
                platform_groups[sid] = []
            if len(platform_groups[sid]) < 10:
                platform_groups[sid].append(item.get("title", ""))

        # 构建发给大模型的 Prompt
        prompt_text = "请阅读以下各大平台的热搜榜单标题，为每个平台提取3个核心热点关键词，并模拟一条符合该平台网民风格的精选短评（15字左右，接地气、一针见血）。\n\n"
        for sid, titles in platform_groups.items():
            prompt_text += f"【平台ID: {sid}】\n"
            for t in titles:
                prompt_text += f"- {t}\n"
            prompt_text += "\n"

        prompt_text += """
任务要求：
1. 必须且只能输出 JSON 格式，不要包含任何前缀、后缀、或 markdown 代码块标记。
2. JSON 的最外层 key 必须是上述提供的【平台ID】。
3. 数据结构示例：
{
    "weibo": {
        "keywords": ["关键词1", "关键词2", "关键词3"],
        "comment": "这是一条模拟的精选评论"
    },
    "zhihu": {
        "keywords": ["关键词A", "关键词B", "关键词C"],
        "comment": "利益相关，简单分析一下背后的逻辑"
    }
}
"""
        
        # 准备一个兜底数据结构，防止大模型API崩了导致UI显示不出来
        fallback_data = {}
        for sid in platform_groups.keys():
            fallback_data[sid] = {
                "keywords": ["分析中", "请稍候"],
                "comment": "大模型暂未返回内容，请检查网络或API限流情况。"
            }

        try:
            messages = [
                SystemMessage(content="你是一个资深的互联网舆情编辑，必须严格输出纯 JSON 格式数据。"),
                HumanMessage(content=prompt_text)
            ]
            response = self.llm.invoke(messages)
            content = response.content
            
            # 使用强大的正则提取，防范大模型乱加标记
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                clean_json_str = match.group(0)
            else:
                clean_json_str = content
                
            summaries = json.loads(clean_json_str)
            print("✅ 关键词与评论生成成功")
            return {"platform_summaries": summaries}
            
        except Exception as e:
            print(f"⚠️ 生成各平台总结失败，已使用兜底数据。报错信息: {e}")
            return {"platform_summaries": fallback_data}


def render_langgraph_html_report(
    news_list: List[Dict],
    analysis_result: Dict,
    forum_discussion: str,
    classification_stats: Optional[Dict[str, Any]] = None,
    platform_summaries: Optional[Dict[str, Any]] = None,
) -> str:
    """使用与 main.py 一致的 HTML 模板生成舆情报告，保留来源与链接、支持保存为图片"""
    now = get_beijing_time()
    start_time = now - timedelta(hours=12)
    total_news = len(news_list)
    top_topics = analysis_result.get("top_topics") or []
    summary = analysis_result.get("summary") or "暂无总结"

    if classification_stats:
        topics_by_category = classification_stats.get("topics_by_category", {})
        category_display_order = classification_stats.get("category_display_order", [])
    else:
        category_order = [
            "经济类舆论", "突发事件舆论", "法治类舆论", "文娱类舆论", "科教类舆论", 
            "国际关系类舆论", "健康类舆论", "治理类舆论", "民生类舆论", "生态环境类舆论", "其他"
        ]
        topics_by_category: Dict[str, List[Dict[str, Any]]] = {c: [] for c in category_order}
        for t in top_topics:
            cat = t.get("category") or "其他"
            if cat not in topics_by_category:
                cat = "其他"
            topics_by_category[cat].append(t)

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
        category_display_order.sort(key=lambda c: (-_category_heat(c), category_order.index(c)))

    platform_order = [
        "weibo", "baidu", "douyin", "zhihu", "toutiao", 
        "tieba", "thepaper", "cls-hot", "ifeng", "wallstreetcn-hot"
    ]
    platform_index = {pid: idx for idx, pid in enumerate(platform_order)}

    platforms: Dict[str, Dict[str, Any]] = {}
    for n in news_list:
        source_id = n.get("source_id") or n.get("source") or "other"
        source_name = n.get("source_name") or n.get("source") or source_id
        if source_id not in platforms:
            platforms[source_id] = {"name": source_name, "items": []}
        platforms[source_id]["items"].append(n)

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="color-scheme" content="light">
    <title>舆情日报 - 热点新闻分析</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js" integrity="sha512-BNaRQnYJYiPSqHHDb58B0yaPfCu+Wgds8Gp/gU33kqBtgNS4tSPHuGibyoeqMV/TJlSKda6FXzoEyYGjTe+vXA==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <style>
        * { box-sizing: border-box; }
        :root { color-scheme: light; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 16px; background: #f6f7fb; color: #111827; line-height: 1.5; }
        .container { max-width: 720px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 16px rgba(17,24,39,0.06); border: 1px solid #eef2f7; }
        .header { background: #ffffff; color: #111827; padding: 24px 24px 20px; text-align: center; position: relative; border-bottom: 1px solid #eef2f7; }
        .save-buttons { position: absolute; top: 16px; right: 16px; display: flex; gap: 8px; }
        .save-btn { background: #ffffff; border: 1px solid #dbe3ee; color: #111827; padding: 8px 12px; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 600; transition: all 0.2s ease; white-space: nowrap; box-shadow: 0 1px 2px rgba(17,24,39,0.04); }
        .save-btn:hover { background: #f8fafc; border-color: #cbd5e1; transform: translateY(-1px); }
        .save-btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .header-title { font-size: 22px; font-weight: 800; margin: 0 0 16px 0; letter-spacing: 0.2px; }
        .header-info { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 14px; opacity: 1; }
        .info-item { text-align: center; }
        .info-label { display: block; font-size: 12px; color: #6b7280; margin-bottom: 4px; }
        .info-value { font-weight: 600; font-size: 16px; }
        .content { padding: 24px; }
        .section { margin-bottom: 32px; }
        .section:last-child { margin-bottom: 0; }
        .section-title { font-size: 16px; font-weight: 800; color: #111827; margin: 0 0 16px 0; padding-bottom: 10px; border-bottom: 1px solid #eef2f7; }
        .section-body { font-size: 14px; color: #374151; line-height: 1.6; white-space: pre-wrap; }
        .topic-category { margin-bottom: 18px; font-size: 14px; font-weight: 600; color: #4b5563; }
        .topic-item { margin-bottom: 16px; padding: 12px; background: #f8fafc; border-radius: 10px; border: 1px solid #eef2f7; border-left: 4px solid #2563eb; }
        .topic-name { font-weight: 600; color: #1e293b; margin-bottom: 4px; }
        .topic-meta { font-size: 12px; color: #64748b; margin-bottom: 6px; }
        .topic-comment { font-size: 13px; color: #475569; }
        
        /* 卡片化新样式 */
        .platform-cards { display: grid; gap: 16px; margin-top: 10px; }
        .p-card { background: #fff; border: 1px solid #eef2f7; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(17,24,39,0.04); }
        .p-card-header { background: #f8fafc; padding: 12px 16px; font-weight: 800; border-bottom: 1px solid #eef2f7; color: #1e293b; font-size: 15px; }
        
        /* 新增：关键词和评论样式 */
        .p-summary { background: #fdfdfd; padding: 12px 16px; border-bottom: 1px solid #eef2f7; font-size: 13px; color: #4b5563; }
        .p-keywords { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
        .p-tag { background: #e0e7ff; color: #3730a3; padding: 4px 8px; border-radius: 6px; font-weight: 600; font-size: 12px; }
        .p-comment { background: #f3f4f6; padding: 10px 12px; border-radius: 8px; font-style: italic; color: #374151; border-left: 4px solid #cbd5e1; line-height: 1.5; }
        .p-comment::before { content: "💬 AI锐评："; font-weight: bold; font-style: normal; color: #64748b; }
        
        .p-list { list-style: none; margin: 0; padding: 0; }
        .p-item { padding: 12px 16px; border-bottom: 1px solid #f1f5f9; display: flex; gap: 12px; align-items: flex-start; }
        .p-item:hover { background: #f8fafc; }
        .p-item:last-child { border-bottom: none; }
        .p-rank { width: 26px; height: 26px; background: #eef2ff; color: #4f46e5; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 800; flex-shrink: 0; }
        .p-rank.top3 { background: #fee2e2; color: #dc2626; }
        .p-title { font-size: 14px; color: #374151; text-decoration: none; line-height: 1.5; flex: 1; }
        .p-title:hover { color: #2563eb; }
        .hot-badge { font-size: 11px; color: #ef4444; background: #fef2f2; padding: 2px 6px; border-radius: 4px; margin-left: 6px; white-space: nowrap; display: inline-block; }
        
        .footer { margin-top: 24px; padding: 20px 24px; background: #f8fafc; border-top: 1px solid #eef2f7; text-align: center; }
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
                <div class="info-item"><span class="info-label">新闻总数</span><span class="info-value">"""
    html += str(total_news)
    html += """ 条</span></div>
                <div class="info-item"><span class="info-label">热点话题</span><span class="info-value">"""
    html += str(len(top_topics))
    html += """ 个</span></div>
                <div class="info-item"><span class="info-label">时间范围</span><span class="info-value">"""
    html += start_time.strftime("%m-%d %H:%M") + " ~ " + now.strftime("%m-%d %H:%M")
    html += """</span></div>
            </div>
        </div>
        <div class="content">
            <div class="section">
                <div class="section-title">今日热点概览</div>
                <div class="section-body">"""
    html += html_escape(summary)
    html += """</div>
            </div>
            <div class="section">
                <div class="section-title">深度分析</div>
"""

    for cat in category_display_order:
        cat_topics = topics_by_category.get(cat) or []
        html += '<div class="topic-category">' + html_escape(cat) + "</div>"
        for t in cat_topics:
            topic = t.get("topic") or ""
            sentiment = t.get("sentiment") or ""
            comment = t.get("comment") or ""
            heat_score = t.get("heat_score")
            html += '<div class="topic-item"><div class="topic-name">' + html_escape(topic) + "</div>"
            meta_parts: List[str] = []
            if sentiment:
                meta_parts.append("情感: " + sentiment)
            if heat_score is not None:
                meta_parts.append("舆情热度: " + str(heat_score))
            if meta_parts:
                html += '<div class="topic-meta">' + html_escape(" · ".join(meta_parts)) + "</div>"
            html += '<div class="topic-comment">' + html_escape(comment) + "</div></div>"

    html += """
            </div>
            <div class="section">
                <div class="section-title">重大事件剖析</div>
                <div class="section-body">"""
    html += html_escape(forum_discussion or "暂无重大事件剖析")
    html += """</div>
            </div>
            <div class="section">
                <div class="section-title">各平台热搜榜单 (Top 10)</div>
                <div class="platform-cards">
"""

    ordered_sids = sorted(platforms.keys(), key=lambda k: platform_index.get(k, 999))
    
    for sid in ordered_sids:
        info = platforms[sid]
        items = info["items"]
        if not items:
            continue
            
        items.sort(key=lambda x: x.get("rank") or 9999)
        top_10_items = items[:10]
        
        html += f'<div class="p-card"><div class="p-card-header">📌 {html_escape(info["name"])}</div>'
        
        # --- 新增：在这里把大模型生成的关键词和评论插入到卡片顶部 ---
        summary_data = (platform_summaries or {}).get(sid)
        if summary_data:
            keywords = summary_data.get("keywords", [])
            comment = summary_data.get("comment", "")
            if keywords or comment:
                html += '<div class="p-summary">'
                if keywords:
                    html += '<div class="p-keywords">'
                    for kw in keywords:
                        html += f'<span class="p-tag">#{html_escape(str(kw))}</span>'
                    html += '</div>'
                if comment:
                    html += f'<div class="p-comment">{html_escape(comment)}</div>'
                html += '</div>'
        # --------------------------------------------------------
        
        html += '<ul class="p-list">'
        
        for n in top_10_items:
            rank = n.get("rank")
            rank_text = str(rank) if rank is not None else "-"
            top_class = " top3" if rank in [1, 2, 3] else ""
            title = html_escape(n.get("title") or "")
            link = html_escape(n.get("mobile_url") or n.get("url") or "")
            hot_val = n.get("hot_value")
            
            html += f'<li class="p-item"><div class="p-rank{top_class}">{rank_text}</div>'
            if link:
                html += f'<a href="{link}" target="_blank" class="p-title">{title}'
            else:
                html += f'<span class="p-title">{title}'
                
            if hot_val is not None and hot_val != 0:
                html += f'<span class="hot-badge">热度 {html_escape(str(hot_val))}</span>'
                
            if link:
                html += '</a>'
            else:
                html += '</span>'
                
            html += '</li>'
            
        html += '</ul></div>'

    html += """
                </div>
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
                link.download = 'BJTUPubClaw_舆情日报_' + now.getFullYear() + String(now.getMonth()+1).padStart(2,'0') + String(now.getDate()).padStart(2,'0') + '_' + String(now.getHours()).padStart(2,'0') + String(now.getMinutes()).padStart(2,'0') + '.png';
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
                const base = 'BJTUPubClaw_舆情日报_' + now.getFullYear() + String(now.getMonth()+1).padStart(2,'0') + String(now.getDate()).padStart(2,'0') + '_' + String(now.getHours()).padStart(2,'0') + String(now.getMinutes()).padStart(2,'0');
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
    """报告生成节点"""

    def __init__(self):
        pass

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [ReportNode] 开始生成报告 ---")
        analysis_result = state.get("analysis_result", {})
        news_data = state.get("news_data", {})
        discussion = state.get("forum_discussion", "")
        classification_stats = state.get("classification_stats")
        
        # --- 获取平台总结数据 ---
        platform_summaries = state.get("platform_summaries", {})

        news_list = news_data.get("news_list") or []

        if not news_list:
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

        # --- 把平台总结传入渲染函数 ---
        html_content = render_langgraph_html_report(news_list, analysis_result, discussion, classification_stats, platform_summaries)
        return {"html_report": html_content}


# === 构建图 ===
def build_graph():
    workflow = StateGraph(TrendState)

    platforms = []
    if CONFIG and "platforms" in CONFIG:
        platforms = CONFIG["platforms"]
    else:
        platforms = [
            {"id": "weibo", "name": "微博"},
            {"id": "zhihu", "name": "知乎"},
            {"id": "toutiao", "name": "今日头条"},
            {"id": "baidu", "name": "百度热搜"},
            {"id": "douyin", "name": "抖音"},
            {"id": "bilibili-hot-search", "name": "bilibili 热搜"},
            {"id": "thepaper", "name": "澎湃新闻"},
            {"id": "tieba", "name": "贴吧"},
            {"id": "ifeng", "name": "凤凰网"},
            {"id": "cls-hot", "name": "财联社热门"},
            {"id": "wallstreetcn-hot", "name": "华尔街见闻"},
        ]

    fetch_node_map = {
        "weibo": FetchWeiboNode,
        "zhihu": FetchZhihuNode,
        "toutiao": FetchToutiaoNode,
        "baidu": FetchBaiduNode,
        "douyin": FetchDouyinNode,
        "bilibili-hot-search": FetchBilibiliNode,
        "thepaper": FetchThepaperNode,
        "tieba": FetchTiebaNode,
        "ifeng": FetchIfengNode,
        "cls-hot": FetchClsNode,
        "wallstreetcn-hot": FetchWallstreetcnNode,
    }

    workflow.add_node("start_fetch", StartFetchNode())

    fetch_node_names = []
    for platform in platforms:
        platform_id = platform["id"]
        if platform_id in fetch_node_map:
            node_name = f"fetch_{platform_id}"
            fetch_node_names.append(node_name)
            workflow.add_node(node_name, fetch_node_map[platform_id]())

    workflow.add_node("platform_summary", PlatformSummaryNode())
    workflow.add_node("merge_fetch", MergeFetchNode())
    workflow.add_node("normalize", NormalizeNewsNode())
    workflow.add_node("insight", InsightNode())
    workflow.add_node("classify", ClassifyNode())
    workflow.add_node("forum", ForumNode())
    workflow.add_node("report", ReportNode())

    if fetch_node_names:
        workflow.set_entry_point("start_fetch")
        for node_name in fetch_node_names:
            workflow.add_edge("start_fetch", node_name)
            workflow.add_edge(node_name, "merge_fetch")
        workflow.add_edge("merge_fetch", "normalize")
    else:
        workflow.add_node("spider", SpiderNode())
        workflow.set_entry_point("spider")
        workflow.add_edge("spider", "normalize")

    workflow.add_edge("normalize", "insight")
    workflow.add_edge("insight", "classify")
    workflow.add_edge("classify", "forum")
    workflow.add_edge("forum", "platform_summary")
    workflow.add_edge("platform_summary", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


def run(config_path: Optional[str] = None) -> str:
    global CONFIG
    if config_path:
        os.environ["CONFIG_PATH"] = config_path
    CONFIG = load_config()

    print("=== BJTUPubClaw LangGraph 版启动 ===")

    app = build_graph()

    initial_state: TrendState = {
        "messages": [],
        "news_data": {},
        "raw_items": [],
        "analysis_result": {},
        "classification_stats": {},
        "forum_discussion": "",
        "html_report": "",
        "error": None,
        "platform_summaries": {}, # --- 新增状态初始化 ---
    }

    final_state = app.invoke(initial_state)

    report_path = ""
    if final_state.get("html_report"):
        output_dir = Path("output_langgraph")
        ensure_directory_exists(str(output_dir))
        filename = f"bjtupubclaw_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        output_path = output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_state["html_report"])

        index_path = output_dir / "index.html"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(final_state["html_report"])

        report_path = str(output_path.resolve())
        print(f"\n✅ 流程执行成功！报告已保存: {report_path}")

        if not _is_docker_env():
            file_url = "file://" + report_path
            print(f"正在打开报告: {file_url}")
            try:
                webbrowser.open(file_url)
            except Exception as e:
                print(f"自动打开浏览器失败: {e}，请手动打开: {report_path}")
        else:
            print(f"（Docker 环境不自动打开浏览器）报告路径: {report_path}")
    else:
        print("\n❌ 流程执行完成，但未生成报告内容。")
        if final_state.get("error"):
            print(f"错误信息: {final_state['error']}")
    return report_path


def main():
    try:
        run()
    except Exception as e:
        print(f"\n❌ 流程执行异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()