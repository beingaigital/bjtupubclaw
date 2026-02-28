# coding=utf-8

import json
import os
import re
import time
import requests
import yaml
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union
import pytz

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

# === 配置管理 (Simplified from main.py) ===
def load_config():
    """加载配置文件"""
    config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")

    if not Path(config_path).exists():
        # 如果找不到config/config.yaml，尝试找上一级
        if Path("../config/config.yaml").exists():
            config_path = "../config/config.yaml"
        else:
            print(f"配置文件 {config_path} 不存在，使用默认配置")
            return {"crawler": {"request_interval": 1000}, "platforms": []}

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return config_data

CONFIG = load_config()

# === 基础工具函数 ===
def get_beijing_time():
    """获取北京时间"""
    return datetime.now(pytz.timezone("Asia/Shanghai"))

def ensure_directory_exists(directory: str):
    """确保目录存在"""
    Path(directory).mkdir(parents=True, exist_ok=True)

# === LLM 客户端 (Generic) ===
class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model

    def invoke(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        if not self.api_key:
            return "Error: API Key not configured."
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }
        
        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data, timeout=60)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"LLM call failed: {e}")
            return f"Error: {str(e)}"

# === MindSpider (数据抓取与话题提取) ===
class DataFetcher:
    """数据获取器 (From main.py)"""
    def __init__(self, proxy_url: Optional[str] = None):
        self.proxy_url = proxy_url

    def fetch_data(self, id_info: Union[str, Any]) -> Optional[str]:
        # 简化版 fetch_data
        id_value = id_info if isinstance(id_info, str) else id_info[0] # assuming tuple (id, name)
        url = f"https://newsnow.busiyi.world/api/s?id={id_value}&latest"
        try:
            response = requests.get(url, timeout=10) # No proxy for simplicity in v2 unless configured
            if response.status_code == 200:
                return response.text
        except Exception as e:
            print(f"Error fetching {id_value}: {e}")
        return None

class MindSpider:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        # 默认新闻源映射 (ID -> Name)
        self.sources = {
            "weibo": "微博热搜",
            "zhihu": "知乎热榜",
            "bilibili-hot-search": "B站热搜",
            "toutiao": "今日头条",
            "douyin": "抖音热榜",
            "36kr": "36氪",
            "sspai": "少数派"
        }
        # 如果配置文件中有平台，优先使用配置文件
        if CONFIG and "platforms" in CONFIG:
            self.sources = {p['id']: p.get('name', p['id']) for p in CONFIG['platforms']}

    async def collect_and_save_news(self) -> Dict[str, Any]:
        """抓取新闻并提取"""
        print("MindSpider: 开始抓取新闻...")
        news_list = []
        for source_id, source_name in self.sources.items():
            print(f"正在抓取: {source_name} ({source_id})")
            raw_data = self.data_fetcher.fetch_data(source_id)
            if raw_data:
                try:
                    data = json.loads(raw_data)
                    items = data.get("items", [])
                    # 提取前10条
                    top_items = items[:10]
                    for item in top_items:
                        news_list.append({
                            "source": source_name,
                            "title": item.get("title"),
                            "url": item.get("url"),
                            "hot_value": item.get("hotValue", 0)
                        })
                except Exception as e:
                    print(f"解析 {source_name} 失败: {e}")
            time.sleep(1) # 避免过快请求

        return {"news_list": news_list}

# === InsightEngine (舆情分析) ===
class InsightEngine:
    def __init__(self):
        self.api_key = os.environ.get("INSIGHT_ENGINE_API_KEY", "")
        self.base_url = os.environ.get("INSIGHT_ENGINE_BASE_URL", "https://api.moonshot.cn/v1")
        self.model = os.environ.get("INSIGHT_ENGINE_MODEL_NAME", "moonshot-v1-8k")
        self.llm = LLMClient(self.api_key, self.base_url, self.model)

    def analyze_news(self, news_list: List[Dict]) -> Dict:
        """分析新闻列表，提取热点和情感"""
        if not news_list:
            return {"error": "No news to analyze"}
        
        print("InsightEngine: 正在分析舆情...")
        
        # 构造Prompt
        news_text = "\n".join([f"- [{n['source']}] {n['title']}" for n in news_list])
        prompt = f"""
        请分析以下今日热点新闻列表：
        {news_text}

        任务：
        1. 提取最热门的5个话题。
        2. 对每个话题进行简要的情感分析（正面/负面/中性）和简短点评。
        3. 总结今天的整体舆情趋势。

        请以JSON格式返回，格式如下：
        {{
            "top_topics": [
                {{"topic": "话题1", "sentiment": "负面", "comment": "点评..."}},
                ...
            ],
            "summary": "整体趋势总结..."
        }}
        """
        
        response = self.llm.invoke(prompt, system_prompt="你是一个资深的舆情分析专家。")
        try:
            # 尝试解析JSON
            # 有时候LLM会返回markdown包裹的json
            json_str = response.replace("```json", "").replace("```", "").strip()
            return json.loads(json_str)
        except Exception as e:
            print(f"InsightEngine JSON解析失败: {e}")
            return {"raw_analysis": response}

# === ReportEngine (报告生成) ===
class ReportEngine:
    def __init__(self):
        self.api_key = os.environ.get("REPORT_ENGINE_API_KEY", "")
        self.base_url = os.environ.get("REPORT_ENGINE_BASE_URL", "https://api.moonshot.cn/v1")
        self.model = os.environ.get("REPORT_ENGINE_MODEL_NAME", "moonshot-v1-8k")
        self.llm = LLMClient(self.api_key, self.base_url, self.model)

    def generate_html_report(self, analysis_result: Dict, news_data: Dict) -> str:
        """生成HTML报告"""
        print("ReportEngine: 正在生成HTML报告...")
        
        prompt = f"""
        基于以下舆情分析结果和原始新闻数据，生成一份现代化的HTML舆情日报。
        
        分析结果：
        {json.dumps(analysis_result, ensure_ascii=False, indent=2)}
        
        (部分)原始新闻：
        {json.dumps(news_data['news_list'][:20], ensure_ascii=False, indent=2)}... (更多省略)

        要求：
        1. 使用Bootstrap或Tailwind CSS（CDN引入）美化界面。
        2. 包含“今日热点概览”、“深度分析”、“情感分布”等板块。
        3. 响应式设计，适配移动端。
        4. 直接返回HTML代码，不要Markdown标记。
        """
        
        return self.llm.invoke(prompt, system_prompt="你是一个专业的前端工程师和数据分析师。")

# === ForumEngine (模拟多Agent讨论) ===
class ForumEngine:
    def __init__(self):
        self.api_key = os.environ.get("QUERY_ENGINE_API_KEY", "") # Reuse query engine key
        self.base_url = os.environ.get("QUERY_ENGINE_BASE_URL", "https://api.moonshot.cn/v1")
        self.model = os.environ.get("QUERY_ENGINE_MODEL_NAME", "moonshot-v1-8k")
        self.llm = LLMClient(self.api_key, self.base_url, self.model)

    def run_discussion(self, topic: str) -> str:
        """模拟关于特定话题的讨论"""
        print(f"ForumEngine: 正在就 '{topic}' 展开讨论...")
        prompt = f"""
        模拟一场关于“{topic}”的简短讨论。
        参与者：
        - Insight (深度分析)
        - Media (媒体传播)
        - Query (事实核查)
        - Host (主持人)

        请生成一段对话脚本，展示各方对该话题的不同视角。
        """
        return self.llm.invoke(prompt)

# === 主程序 ===
class TrendRadarV2:
    def __init__(self):
        self.mind_spider = MindSpider()
        self.insight_engine = InsightEngine()
        self.report_engine = ReportEngine()
        self.forum_engine = ForumEngine()

    async def run(self):
        print("=== TrendRadar V2 启动 ===")
        
        # 1. 抓取
        news_data = await self.mind_spider.collect_and_save_news()
        print(f"抓取完成，共 {len(news_data.get('news_list', []))} 条新闻")
        
        # 2. 分析
        analysis_result = self.insight_engine.analyze_news(news_data.get('news_list', []))
        print("分析完成")
        
        # 3. 论坛讨论 (选取Top1话题)
        discussion = ""
        if "top_topics" in analysis_result and isinstance(analysis_result["top_topics"], list):
            top_topic = analysis_result["top_topics"][0].get("topic", "今日热点")
            discussion = self.forum_engine.run_discussion(top_topic)
            analysis_result["forum_discussion"] = discussion
        
        # 4. 生成报告
        html_report = self.report_engine.generate_html_report(analysis_result, news_data)
        
        # 5. 保存报告
        output_dir = Path("output_v2")
        ensure_directory_exists(str(output_dir))
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        output_path = output_dir / filename
        
        # 清理markdown标记 (如果LLM还是返回了)
        if html_report.startswith("```html"):
            html_report = html_report.replace("```html", "").replace("```", "")
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_report)
            
        print(f"报告已生成: {output_path}")
        return str(output_path)

if __name__ == "__main__":
    import asyncio
    radar = TrendRadarV2()
    asyncio.run(radar.run())
