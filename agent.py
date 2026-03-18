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

# 📊 全局 CONFIG 变量，供 render_langgraph_html_report 使用
CONFIG = None


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

    # 原始新闻数据
    news_data: Dict[str, Any]
    # 原始抓取的新闻列表（用于并行节点合并）
    raw_items: Annotated[List[Dict[str, Any]], operator.add]
    # 结构化分析结果
    analysis_result: Dict[str, Any]
    # 十一类舆情分类统计结果
    classification_stats: Dict[str, Any]
    # 论坛讨论内容
    forum_discussion: str
    # 最终HTML报告
    html_report: str
    # 执行过程中的消息记录（可选，用于调试）
    messages: Annotated[List[BaseMessage], operator.add]
    # 错误信息
    error: Optional[str]


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
            # 为了提高样本量，这里尽量多保留一些热搜项（例如前 30 条）
            top_items = raw_items[:30]
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
            # 返回空列表，LangGraph会自动合并
            return {"raw_items": []}

        items = self.parse_data(raw_data)
        print(f"✅ {self.platform_name} 抓取完成，共 {len(items)} 条新闻")

        # 保存到每小时数据目录
        try:
            save_path = save_hourly_data(self.platform_id, self.platform_name, items, timestamp)
            print(f"   数据已保存: {save_path}")
        except Exception as e:
            print(f"   保存数据失败: {e}")

        # 返回抓取的数据（LangGraph会自动合并到raw_items）
        return {"raw_items": items}


# 为每个平台创建Fetch节点
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
                    # 为了提高样本量，这里尽量多保留一些热搜项（例如前 30 条）
                    top_items = items[:30]
                    for idx, item in enumerate(top_items, 1):
                        news_list.append(
                            {
                                "source_id": source_id,
                                "source": source_name,
                                "source_name": source_name,
                                "title": item.get("title"),
                                # 平台内排行（1 表示热搜第1）
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

        # 将本次抓取结果保存为快照，并合并最近 24 小时内的历史快照，扩大样本量
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
                        key = (item.get("source_id") or item.get("source") or "", item.get("title") or "")
                        if not key[1]:
                            continue
                        existing = aggregated.get(key)
                        if existing is None:
                            aggregated[key] = dict(item)
                        else:
                            # 保留更靠前的排名和更高的热度
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
        # 去除首尾空格
        title = title.strip()
        # 去除多个连续空格
        import re

        title = re.sub(r"\s+", " ", title)
        return title

    def normalize_news(self, raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清洗、去重、排序新闻"""
        if not raw_items:
            return []

        # 1. 清洗：清理标题
        cleaned_items = []
        for item in raw_items:
            title = item.get("title", "")
            cleaned_title = self.clean_title(title)
            if cleaned_title:
                item_copy = dict(item)
                item_copy["title"] = cleaned_title
                cleaned_items.append(item_copy)

        # 2. 去重：基于 (source_id, title) 去重
        seen = {}
        deduplicated_items = []
        for item in cleaned_items:
            key = (item.get("source_id") or item.get("source") or "", item.get("title") or "")
            if not key[1]:  # 标题为空则跳过
                continue

            existing = seen.get(key)
            if existing is None:
                seen[key] = item
                deduplicated_items.append(item)
            else:
                # 保留更靠前的排名和更高的热度
                rank_new = item.get("rank")
                rank_old = existing.get("rank")
                if rank_new is not None and (rank_old is None or rank_new < rank_old):
                    existing["rank"] = rank_new

                hot_new = item.get("hot_value")
                hot_old = existing.get("hot_value")
                if hot_new is not None and (hot_old is None or hot_new > hot_old):
                    existing["hot_value"] = hot_new

        # 3. 排序：按平台权重和平台内排名排序
        platform_order = ["weibo", "baidu", "douyin", "zhihu", "toutiao", "tieba", "thepaper", "cls-hot", "ifeng", "wallstreetcn-hot"]
        platform_index = {pid: idx for idx, pid in enumerate(platform_order)}

        def sort_key(item: Dict[str, Any]) -> tuple:
            source_id = item.get("source_id") or item.get("source") or "other"
            platform_rank = platform_index.get(source_id, len(platform_index))
            item_rank = item.get("rank") or 9999
            return (platform_rank, item_rank)

        sorted_items = sorted(deduplicated_items, key=sort_key)

        return sorted_items

    def __call__(self, state: TrendState) -> TrendState:
        """执行清洗、去重、排序"""
        print("--- [NormalizeNewsNode] 开始清洗、去重、排序 ---")

        # 从状态中获取原始数据
        news_data = state.get("news_data", {})
        raw_items = news_data.get("raw_items", [])

        # 加载过去12小时的历史数据
        print("正在加载过去12小时的历史数据...")
        historical_items = load_past_hours_data(lookback_hours=12)
        print(f"从历史数据中加载了 {len(historical_items)} 条新闻")

        # 合并当前抓取的数据和历史数据
        all_items = raw_items + historical_items

        # 执行清洗、去重、排序
        normalized_items = self.normalize_news(all_items)

        print(f"✅ 清洗、去重、排序完成，共 {len(normalized_items)} 条新闻")

        return {"news_data": {"news_list": normalized_items, "raw_items": []}, "raw_items": []}


class StartFetchNode:
    """开始抓取的入口节点"""

    def __init__(self):
        pass

    def __call__(self, state: TrendState) -> TrendState:
        """初始化抓取状态"""
        print("--- [StartFetchNode] 开始并行抓取所有平台 ---")
        return {"raw_items": []}


class MergeFetchNode:
    """合并所有Fetch节点的结果"""

    def __init__(self):
        pass

    def __call__(self, state: TrendState) -> TrendState:
        """合并所有Fetch节点的结果"""
        print("--- [MergeFetchNode] 合并所有抓取结果 ---")
        # raw_items已经通过Annotated自动合并了
        raw_items = state.get("raw_items", [])
        print(f"✅ 所有平台抓取完成，共 {len(raw_items)} 条原始新闻")
        # 将raw_items转移到news_data中
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

        # 优化1: 智能筛选新闻，减少输入量，避免触发API限制
        # 优先选择：高排名、高热度、重要平台的新闻
        def news_priority_score(news_item: Dict[str, Any]) -> float:
            """计算新闻优先级分数"""
            source_id = news_item.get("source_id", "").lower()
            rank = news_item.get("rank", 9999)
            hot_value = news_item.get("hot_value", 0)

            # 平台权重（从高到低）
            platform_weights = {
                "weibo": 10.0,
                "baidu": 9.0,
                "douyin": 8.0,
                "zhihu": 7.0,
                "toutiao": 6.0,
                "tieba": 5.0,
                "thepaper": 4.0,
                "cls-hot": 3.0,
                "ifeng": 2.0,
                "wallstreetcn-hot": 1.0,
            }
            platform_weight = platform_weights.get(source_id, 0.5)

            # 排名权重（排名越靠前分数越高）
            rank_weight = max(0, 30 - rank) / 30.0

            # 热度权重（归一化到0-1）
            hot_weight = min(1.0, hot_value / 1000000.0) if hot_value else 0

            # 综合分数
            score = platform_weight * 0.5 + rank_weight * 0.3 + hot_weight * 0.2
            return score

        # 按优先级排序并限制数量（最多120条，避免输入过长）
        sorted_news = sorted(news_list, key=news_priority_score, reverse=True)
        max_news_count = 120  # 限制新闻数量，减少输入长度
        selected_news = sorted_news[:max_news_count]

        if len(news_list) > max_news_count:
            print(f"📊 从 {len(news_list)} 条新闻中筛选出前 {max_news_count} 条高优先级新闻进行分析")

        # 优化2: 简化数据格式，只保留必要信息
        news_text = "\n".join([f"- {n['title']}" for n in selected_news])  # 移除来源信息，减少字符数

        # 优化3: 简化Prompt，移除敏感关键词示例，使用更中性的表述
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

            # 优化4: 添加小延迟，避免请求频率过高
            time.sleep(0.5)

            response = self.llm.invoke(messages)
            content = response.content

            # 清理可能的 Markdown 标记
            json_str = content.replace("```json", "").replace("```", "").strip()

            # 尝试解析JSON，如果失败则尝试修复不完整的JSON
            try:
                analysis_result = json.loads(json_str)
            except json.JSONDecodeError as json_err:
                # 尝试修复不完整的JSON
                print(f"⚠️  JSON解析失败，尝试修复: {json_err}")
                try:
                    import re

                    # 方法1: 尝试找到所有完整的topic对象（通过平衡大括号）
                    # 查找 "top_topics": [ 之后的内容
                    topics_match = re.search(r'"top_topics"\s*:\s*\[', json_str)
                    if topics_match:
                        start_pos = topics_match.end()
                        # 查找所有完整的topic对象
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
                                        # 找到了一个完整的对象
                                        complete_objects.append((obj_start, i + 1))
                                        obj_start = None

                        # 如果找到了完整的对象，尝试构建有效的JSON
                        if complete_objects:
                            # 提取所有完整对象的文本
                            obj_texts = [json_str[start:end] for start, end in complete_objects]
                            # 构建有效的JSON
                            partial_json = '{"top_topics": [' + ",".join(obj_texts) + '], "summary": "JSON响应被截断，仅提取了部分数据"}'
                            try:
                                analysis_result = json.loads(partial_json)
                                print(f"⚠️  JSON被截断，已修复并提取了 {len(analysis_result.get('top_topics', []))} 个话题")
                            except Exception as parse_err:
                                print(f"⚠️  修复后的JSON仍无法解析: {parse_err}")
                                raise json_err
                        else:
                            raise json_err
                    else:
                        raise json_err
                except Exception as fix_err:
                    # 如果修复也失败，返回一个默认结构，但保留错误信息
                    print(f"⚠️  JSON解析失败且无法修复: {json_err}")
                    # 尝试至少提取一些文本信息作为summary
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
        # 十一类舆情分类标准（采用《十大舆情分类》标准）
        self.category_order = [
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

    def __call__(self, state: TrendState) -> TrendState:
        print("--- [ClassifyNode] 开始十一类舆情分类统计 ---")
        analysis_result = state.get("analysis_result", {})
        top_topics = analysis_result.get("top_topics") or []

        if not top_topics:
            print("⚠️  警告: 没有分析结果可分类，跳过分类统计步骤")
            return {
                "classification_stats": {
                    "topics_by_category": {c: [] for c in self.category_order},
                    "category_display_order": [],
                    "category_heat_map": {c: 0.0 for c in self.category_order},
                }
            }

        # 按类别对事件进行分组
        topics_by_category: Dict[str, List[Dict[str, Any]]] = {c: [] for c in self.category_order}
        for t in top_topics:
            cat = t.get("category") or "其他"
            if cat not in topics_by_category:
                cat = "其他"
            topics_by_category[cat].append(t)

        # 计算每个类别的最大舆情热度
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

        # 按类别的"最大舆情热度"从高到低排序，只展示有热点事件的类别
        category_display_order = [c for c in self.category_order if topics_by_category.get(c)]
        category_display_order.sort(key=lambda c: (-category_heat_map[c], self.category_order.index(c)))

        print(f"分类统计完成，共 {len(category_display_order)} 个类别有热点事件")

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

        print(f"讨论话题: {topic}")

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
            print(f"ForumNode Error: {e}")
            return {"forum_discussion": "讨论生成失败"}


def render_langgraph_html_report(
    news_list: List[Dict],
    analysis_result: Dict,
    forum_discussion: str,
    classification_stats: Optional[Dict[str, Any]] = None,
) -> str:
    """使用与 main.py 一致的 HTML 模板生成舆情报告，保留来源与链接、支持保存为图片"""
    now = get_beijing_time()
    start_time = now - timedelta(hours=12)
    total_news = len(news_list)
    top_topics = analysis_result.get("top_topics") or []
    summary = analysis_result.get("summary") or "暂无总结"

    # 📊 读取对比模式配置
    compare_mode_enabled = False
    compare_platforms = []
    compare_layout = "side-by-side"
    compare_top_n = 20
    
    # 优先使用环境变量（命令行参数）
    if os.environ.get("COMPARE_MODE", "").lower() == "true":
        compare_mode_enabled = True
        compare_platforms = [p.strip() for p in os.environ.get("COMPARE_PLATFORMS", "").split(",") if p.strip()]
        compare_layout = os.environ.get("COMPARE_LAYOUT", "side-by-side")
        try:
            compare_top_n = int(os.environ.get("COMPARE_TOP_N", "20"))
        except ValueError:
            compare_top_n = 20
    # 其次使用配置文件
    elif CONFIG and CONFIG.get("compare_mode", {}).get("enabled"):
        compare_mode_enabled = True
        compare_platforms = CONFIG.get("compare_mode", {}).get("platforms", [])
        compare_layout = CONFIG.get("compare_mode", {}).get("layout", "side-by-side")
        compare_top_n = CONFIG.get("compare_mode", {}).get("top_n", 20)
    
    if compare_mode_enabled and compare_platforms:
        print(f"📊 平台对比模式已启用：{', '.join(compare_platforms)}")
        print(f"   布局：{compare_layout}, Top {compare_top_n}")

    # 使用 ClassifyNode 生成的分类统计结果，如果没有则回退到原有逻辑
    if classification_stats:
        topics_by_category = classification_stats.get("topics_by_category", {})
        category_display_order = classification_stats.get("category_display_order", [])
    else:
        # 回退逻辑：如果没有分类统计结果，则使用原有计算方式（保持向后兼容）
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

    # 平台权重与排序（用于原始信息表格 & 热度理解）
    # 这里假定配置中的 source_id 使用这些英文标识；若不匹配则自动归为“其他”
    platform_order = [
        "weibo",  # 微博
        "baidu-hot",  # 百度热搜（示例 ID，需与实际配置对应）
        "douyin",  # 抖音
        "zhihu",  # 知乎
        "toutiao",  # 今日头条
        "tieba",  # 贴吧
        "thepaper",  # 澎湃新闻
        "cls",  # 财联社热门
        "ifeng",  # 凤凰网
        "wallstreetcn",  # 华尔街见闻
    ]
    platform_index = {pid: idx for idx, pid in enumerate(platform_order)}

    # 按来源分组新闻（同时保留 source_id，便于排序和权重）
    platforms: Dict[str, Dict[str, Any]] = {}
    for n in news_list:
        source_id = n.get("source_id") or n.get("source") or "other"
        source_name = n.get("source_name") or n.get("source") or source_id
        if source_id not in platforms:
            platforms[source_id] = {"name": source_name, "items": []}
        platforms[source_id]["items"].append(n)

    # 📊 对比模式：过滤平台
    if compare_mode_enabled and compare_platforms:
        filtered_platforms = {}
        for pid in compare_platforms:
            if pid in platforms:
                filtered_platforms[pid] = platforms[pid]
        # 如果没有匹配的平台，保留所有平台（降级到普通模式）
        if filtered_platforms:
            platforms = filtered_platforms
            print(f"   对比平台数：{len(platforms)}")

    # 对比模式标题
    compare_title = f"平台对比 ({len(platforms)})" if compare_mode_enabled and compare_platforms else "热点新闻分析"
    
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>舆情日报 - """ + html_escape(compare_title) + """</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js" integrity="sha512-BNaRQnYJYiPSqHHDb58B0yaPfCu+Wgds8Gp/gU33kqBtgNS4tSPHuGibyoeqMV/TJlSKda6FXzoEyYGjTe+vXA==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 16px; background: #fafafa; color: #333; line-height: 1.5; }
        .container { max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 16px rgba(0,0,0,0.06); }
        .header { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: white; padding: 32px 24px; text-align: center; position: relative; }
        .theme-toggle { position: absolute; top: 16px; left: 16px; background: rgba(255, 255, 255, 0.2); border: 1px solid rgba(255, 255, 255, 0.3); color: white; width: 40px; height: 40px; border-radius: 8px; cursor: pointer; font-size: 18px; transition: all 0.2s ease; backdrop-filter: blur(10px); display: flex; align-items: center; justify-content: center; }
        .theme-toggle:hover { background: rgba(255, 255, 255, 0.3); transform: scale(1.1); }
        .save-buttons { position: absolute; top: 16px; right: 16px; display: flex; gap: 8px; }
        .save-btn { background: rgba(255, 255, 255, 0.2); border: 1px solid rgba(255, 255, 255, 0.3); color: white; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.2s ease; backdrop-filter: blur(10px); white-space: nowrap; }
        .save-btn:hover { background: rgba(255, 255, 255, 0.3); border-color: rgba(255, 255, 255, 0.5); transform: translateY(-1px); }
        .save-btn:disabled { opacity: 0.6; cursor: not-allowed; }
        
        /* 深色模式 */
        .dark-mode { background: #0f172a; color: #e2e8f0; }
        .dark-mode .container { background: #1e293b; box-shadow: 0 2px 16px rgba(0,0,0,0.3); }
        .dark-mode .header { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-bottom: 1px solid #374151; }
        .dark-mode .header-info { opacity: 0.9; }
        .dark-mode .info-label { color: #94a3b8; }
        .dark-mode .info-value { color: #e2e8f0; }
        .dark-mode .section-title { color: #e2e8f0; border-bottom-color: #374151; }
        .dark-mode .section-body { color: #cbd5e1; }
        .dark-mode .platform-save-tip { background: #334155; color: #94a3b8; }
        .dark-mode .platform-card { background: #1e293b; border-color: #374151; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
        .dark-mode .platform-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.5); }
        .dark-mode .platform-header { background: linear-gradient(135deg, #334155 0%, #475569 100%); border-bottom-color: #475569; }
        .dark-mode .platform-name { color: #e2e8f0; }
        .dark-mode .platform-count { background: #4f46e5; color: #e0e7ff; }
        .dark-mode .platform-save-btn { background: #4f46e5; }
        .dark-mode .platform-save-btn:hover { background: #6366f1; }
        .dark-mode .news-item { border-bottom-color: #334155; }
        .dark-mode .news-item:hover { background: #334155; }
        .dark-mode .news-rank { background: #475569; color: #94a3b8; }
        .dark-mode .news-link { color: #818cf8; }
        .dark-mode .news-link:hover { color: #a5b4fc; }
        .dark-mode .news-link:visited { color: #c4b5fd; }
        .dark-mode .hot-badge { background: #451a1a; color: #fca5a5; }
        .dark-mode .topic-item { background: #334155; border-left-color: #6366f1; }
        .dark-mode .topic-name { color: #e2e8f0; }
        .dark-mode .topic-meta { color: #94a3b8; }
        .dark-mode .topic-comment { color: #cbd5e1; }
        
        /* 📊 平台对比模式样式 */
        .compare-mode-indicator { position: absolute; top: 16px; left: 50%; transform: translateX(-50%); background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(10px); padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 600; border: 1px solid rgba(255, 255, 255, 0.3); }
        .compare-grid { display: grid; gap: 16px; }
        .compare-grid.grid-2 { grid-template-columns: 1fr 1fr; }
        .compare-grid.grid-3 { grid-template-columns: repeat(3, 1fr); }
        .compare-side-by-side { display: flex; flex-direction: column; gap: 16px; }
        .compare-tabs { margin-bottom: 16px; }
        .compare-tab-buttons { display: flex; gap: 8px; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; margin-bottom: 16px; }
        .compare-tab-btn { padding: 8px 16px; border: none; background: #f3f4f6; border-radius: 6px 6px 0 0; cursor: pointer; font-weight: 500; transition: all 0.2s; }
        .compare-tab-btn:hover { background: #e5e7eb; }
        .compare-tab-btn.active { background: #4f46e5; color: white; }
        .compare-tab-content { display: none; }
        .compare-tab-content.active { display: block; }
        
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
        
        /* 平台卡片样式 */
        .platform-save-tip { font-size: 13px; color: #64748b; background: #f1f5f9; padding: 8px 16px; border-radius: 0 0 8px 8px; margin-bottom: 20px; }
        .platform-card { margin-bottom: 24px; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid #e5e7eb; transition: all 0.2s ease; }
        .platform-card:last-child { margin-bottom: 0; }
        .platform-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.12); transform: translateY(-2px); }
        .platform-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 18px; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-bottom: 1px solid #e5e7eb; }
        .platform-left { display: flex; align-items: center; gap: 10px; flex: 1; }
        .platform-icon { font-size: 18px; }
        .platform-name { font-weight: 700; color: #1e293b; font-size: 15px; }
        .platform-right { display: flex; align-items: center; gap: 10px; }
        .platform-count { font-size: 12px; color: #64748b; background: #e0e7ff; padding: 3px 10px; border-radius: 12px; font-weight: 600; }
        .platform-save-btn { background: #2563eb; color: white; border: none; border-radius: 6px; padding: 6px 10px; font-size: 14px; cursor: pointer; transition: all 0.2s; font-weight: 600; }
        .platform-save-btn:hover { background: #1d4ed8; transform: scale(1.05); }
        .platform-save-btn:active { transform: scale(0.95); }
        .platform-news-list { padding: 8px 0; }
        
        .news-item { margin-bottom: 8px; padding: 12px 18px; border-bottom: 1px solid #f1f5f9; display: flex; gap: 12px; align-items: flex-start; transition: background 0.15s; }
        .news-item:last-child { border-bottom: none; }
        .news-item:hover { background: #f8fafc; }
        .news-rank { font-size: 13px; font-weight: 700; color: #94a3b8; min-width: 32px; text-align: center; flex-shrink: 0; background: #f1f5f9; padding: 2px 6px; border-radius: 6px; }
        .news-content { flex: 1; min-width: 0; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; line-height: 1.4; }
        .news-item:last-child { border-bottom: none; }
        .news-num { color: #999; font-size: 12px; font-weight: 600; min-width: 20px; text-align: center; flex-shrink: 0; background: #f1f5f9; border-radius: 50%; width: 22px; height: 22px; display: flex; align-items: center; justify-content: center; }
        .news-content { flex: 1; min-width: 0; }
        .news-link { color: #2563eb; text-decoration: none; font-weight: 500; }
        .news-link:hover { text-decoration: underline; color: #1d4ed8; }
        .news-link:visited { color: #7c3aed; }
        .news-source { color: #666; font-size: 12px; margin-bottom: 4px; }
        .news-hot { font-size: 11px; color: #dc2626; font-weight: 500; }
        .hot-badge { font-size: 11px; color: #dc2626; background: #fef2f2; padding: 2px 6px; border-radius: 4px; white-space: nowrap; }
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
        
        /* 对比模式响应式 */
        @media (max-width: 768px) {
            .compare-grid.grid-2, .compare-grid.grid-3 { grid-template-columns: 1fr; }
            .compare-mode-indicator { position: static; transform: none; display: inline-block; margin-bottom: 12px; }
        }
        @media (max-width: 480px) { body { padding: 12px; } .header { padding: 24px 20px; } .content { padding: 20px; } .header-info { grid-template-columns: 1fr; } .save-buttons { position: static; margin-bottom: 16px; justify-content: center; flex-direction: column; width: 100%; } .save-btn { width: 100%; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <button class="theme-toggle" onclick="toggleTheme()" title="切换深色/浅色模式">🌙</button>
            <div class="save-buttons">
                <button class="save-btn" onclick="saveAsImage()">保存为图片</button>
                <button class="save-btn" onclick="saveAsMultipleImages()">分段保存</button>
            </div>
            """ + (f'<div class="compare-mode-indicator">📊 平台对比模式：{", ".join([platforms.get(pid, {}).get("name", pid) for pid in compare_platforms])}</div>' if compare_mode_enabled and compare_platforms else '<div class="header-title">舆情日报 · 热点新闻分析</div>') + """
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

    # 按类别分组展示深度分析事件（类别按热度排序，只有有热点的类别才展示）
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
            
            <div class="section" id="platform-section">
                <div class="section-title">原始信息抓取（来源与链接）</div>
                <div class="platform-save-tip">💡 点击平台卡片右上角可单独保存该平台</div>
"""

    # 对比模式：生成布局包装器开头
    if compare_mode_enabled and compare_platforms and len(platforms) >= 2:
        if compare_layout == "grid":
            grid_class = "grid-3" if len(platforms) >= 3 else "grid-2"
            html += f'<div class="compare-grid {grid_class}">'
        elif compare_layout == "tabs":
            html += '<div class="compare-tabs">'
            html += '<div class="compare-tab-buttons">'
            for idx, sid in enumerate(platforms.keys()):
                platform_name = platforms[sid]["name"]
                active = "active" if idx == 0 else ""
                html += f'<button class="compare-tab-btn {active}" onclick="switchCompareTab(\'tab-{idx}\', event)">{html_escape(platform_name)}</button>'
            html += '</div>'
        else:  # side-by-side
            html += '<div class="compare-side-by-side">'

    # 按平台顺序排序
    def _platform_sort_key(sid: str) -> int:
        return platform_index.get(sid, len(platform_index))

    sorted_platforms = sorted(platforms.keys(), key=_platform_sort_key)

    # 按平台分组展示
    for idx, sid in enumerate(sorted_platforms):
        info = platforms[sid]
        platform_name = info["name"]
        items = info["items"]
        platform_id = f"platform-{idx}"
        
        # 平台内按排行排序
        items_sorted = sorted(items, key=lambda n: n.get("rank") or 9999)
        
        # Tabs 模式需要额外的 tab-content 包装器
        tab_active = "active" if compare_mode_enabled and compare_layout == "tabs" and idx == 0 else ""
        tab_wrapper_start = f'<div id="tab-{idx}" class="compare-tab-content {tab_active}">' if compare_mode_enabled and compare_layout == "tabs" else ''
        tab_wrapper_end = '</div>' if compare_mode_enabled and compare_layout == "tabs" else ''
        
        html += tab_wrapper_start + f"""
            <div class="platform-card" id="{platform_id}" data-platform="{html_escape(platform_name)}">
                <div class="platform-header">
                    <div class="platform-left">
                        <span class="platform-icon">📱</span>
                        <span class="platform-name">{html_escape(platform_name)}</span>
                    </div>
                    <div class="platform-right">
                        <span class="platform-count">{len(items_sorted)} 条</span>
                        <button class="platform-save-btn" onclick="saveSinglePlatform('{platform_id}', '{html_escape(platform_name)}')" title="保存该平台为图片">💾</button>
                    </div>
                </div>
                <div class="platform-news-list">
"""
        
        for i, n in enumerate(items_sorted, 1):
            rank = n.get("rank")
            rank_text = str(rank) if rank is not None else str(i)
            title = n.get("title") or ""
            link_url = n.get("mobile_url") or n.get("url") or ""
            hot_val = n.get("hot_value")
            
            html += f"""
                    <div class="news-item">
                        <span class="news-rank">{rank_text}</span>
                        <div class="news-content">
"""
            if link_url:
                html += f'<a href="{html_escape(link_url)}" target="_blank" class="news-link">{html_escape(title)}</a>'
            else:
                html += html_escape(title)
            
            if hot_val is not None and hot_val != 0:
                html += f'<span class="hot-badge">🔥 {hot_val}</span>'
            
            html += """
                        </div>
                    </div>
"""
        
        html += """
                </div>
            </div>
""" + tab_wrapper_end

    # 对比模式：关闭布局包装器
    if compare_mode_enabled and compare_platforms and len(platforms) >= 2:
        html += '</div>'

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
        async function saveSinglePlatform(platformId, platformName) {
            const platformCard = document.getElementById(platformId);
            if (!platformCard) return;
            
            const scale = 2;
            try {
                const originalBg = platformCard.style.background;
                platformCard.style.background = 'white';
                
                const canvas = await html2canvas(platformCard, { 
                    backgroundColor: '#ffffff', 
                    scale: scale, 
                    useCORS: true, 
                    logging: false 
                });
                
                platformCard.style.background = originalBg;
                
                const now = new Date();
                const base = 'BJTUPubClaw_' + platformName + '_' + now.getFullYear() + String(now.getMonth()+1).padStart(2,'0') + String(now.getDate()).padStart(2,'0') + '_' + String(now.getHours()).padStart(2,'0') + String(now.getMinutes()).padStart(2,'0');
                
                const a = document.createElement('a');
                a.download = base + '.png';
                a.href = canvas.toDataURL('image/png', 1.0);
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            } catch (e) {
                alert('保存失败：' + e.message);
            }
        }
        function switchCompareTab(tabId, event) {
            // 隐藏所有 tab 内容
            const contents = document.querySelectorAll('.compare-tab-content');
            contents.forEach(function(c) { c.classList.remove('active'); });
            // 取消所有按钮激活状态
            const buttons = document.querySelectorAll('.compare-tab-btn');
            buttons.forEach(function(b) { b.classList.remove('active'); });
            // 激活当前 tab
            const targetTab = document.getElementById(tabId);
            if (targetTab) targetTab.classList.add('active');
            if (event && event.target) event.target.classList.add('active');
        }
        
        function toggleTheme() {
            document.body.classList.toggle('dark-mode');
            const isDark = document.body.classList.contains('dark-mode');
            const btn = document.querySelector('.theme-toggle');
            if (btn) btn.textContent = isDark ? '☀️' : '🌙';
            localStorage.setItem('pubclaw-theme', isDark ? 'dark' : 'light');
        }
        document.addEventListener('DOMContentLoaded', function(){ 
            window.scrollTo(0, 0);
            // 恢复用户主题偏好
            const savedTheme = localStorage.getItem('pubclaw-theme');
            if (savedTheme === 'dark') {
                document.body.classList.add('dark-mode');
                const btn = document.querySelector('.theme-toggle');
                if (btn) btn.textContent = '☀️';
            }
        });
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
        classification_stats = state.get("classification_stats")

        news_list = news_data.get("news_list") or []

        # 修复：即使有错误，只要有新闻数据就生成报告
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

        html_content = render_langgraph_html_report(news_list, analysis_result, discussion, classification_stats)
        return {"html_report": html_content}


# === 构建图 ===
def build_graph():
    workflow = StateGraph(TrendState)

    # 获取配置中的平台列表
    platforms = []
    if CONFIG and "platforms" in CONFIG:
        platforms = CONFIG["platforms"]
    else:
        # 默认平台列表
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

    # 创建Fetch节点映射
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

    # 添加入口节点
    workflow.add_node("start_fetch", StartFetchNode())

    # 添加Fetch节点（并行抓取）
    fetch_node_names = []
    for platform in platforms:
        platform_id = platform["id"]
        if platform_id in fetch_node_map:
            node_name = f"fetch_{platform_id}"
            fetch_node_names.append(node_name)
            workflow.add_node(node_name, fetch_node_map[platform_id]())

    # 添加其他节点
    workflow.add_node("merge_fetch", MergeFetchNode())
    workflow.add_node("normalize", NormalizeNewsNode())
    workflow.add_node("insight", InsightNode())
    workflow.add_node("classify", ClassifyNode())
    workflow.add_node("forum", ForumNode())
    workflow.add_node("report", ReportNode())

    # 定义边：所有Fetch节点从start_fetch开始，然后合并
    if fetch_node_names:
        # 设置入口点
        workflow.set_entry_point("start_fetch")

        # 所有Fetch节点都从start_fetch开始（这样可以并行执行）
        for node_name in fetch_node_names:
            workflow.add_edge("start_fetch", node_name)

        # 所有Fetch节点都连接到merge_fetch
        for node_name in fetch_node_names:
            workflow.add_edge(node_name, "merge_fetch")

        # 合并后进入normalize
        workflow.add_edge("merge_fetch", "normalize")
    else:
        # 如果没有配置平台，使用旧的SpiderNode作为后备
        workflow.add_node("spider", SpiderNode())
        workflow.set_entry_point("spider")
        workflow.add_edge("spider", "normalize")

    # 后续流程
    workflow.add_edge("normalize", "insight")
    workflow.add_edge("insight", "classify")
    workflow.add_edge("classify", "forum")
    workflow.add_edge("forum", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


def run(config_path: Optional[str] = None) -> str:
    """bjtupubclaw 入口：复用 trend_radar_langgraph.py 的完整效果，只做必要的配置路径与品牌名适配。"""
    global CONFIG
    if config_path:
        os.environ["CONFIG_PATH"] = config_path
    CONFIG = load_config()

    print("=== BJTUPubClaw LangGraph 版启动 ===")

    app = build_graph()

    # 初始状态
    initial_state: TrendState = {
        "messages": [],
        "news_data": {},
        "raw_items": [],
        "analysis_result": {},
        "classification_stats": {},
        "forum_discussion": "",
        "html_report": "",
        "error": None,
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

        # 同时写入 index.html，便于直接打开目录时查看最新报告
        index_path = output_dir / "index.html"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(final_state["html_report"])

        report_path = str(output_path.resolve())
        print(f"\n✅ 流程执行成功！报告已保存: {report_path}")

        # 非 Docker 环境下自动在浏览器中打开报告
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


# === 主程序入口 ===
def main():
    try:
        run()
    except Exception as e:
        print(f"\n❌ 流程执行异常: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

