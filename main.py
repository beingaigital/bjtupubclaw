import argparse
import os

try:
    from .agent import run
except Exception:
    from agent import run


def main():
    parser = argparse.ArgumentParser(prog="bjtupubclaw", description="热点助手与舆情分析 - 支持平台对比模式")
    parser.add_argument("--config", dest="config", default=None, help="配置文件路径")
    parser.add_argument("--env", dest="env_file", default=None, help="环境变量文件路径")
    parser.add_argument("--compare", dest="compare", default=None, 
                        help="平台对比模式：指定 2-3 个平台 ID 进行对比，用逗号分隔 (例：weibo,zhihu,douyin)")
    parser.add_argument("--layout", dest="layout", default="side-by-side",
                        choices=["side-by-side", "grid", "tabs"],
                        help="对比布局方式 (默认：side-by-side)")
    parser.add_argument("--top-n", dest="top_n", type=int, default=20,
                        help="每个平台显示的热搜条数 (默认：20)")
    args = parser.parse_args()
    
    if args.env_file:
        os.environ["ENV_FILE"] = args.env_file
    
    # 如果指定了 --compare 参数，设置环境变量让 agent 使用对比模式
    if args.compare:
        compare_platforms = [p.strip() for p in args.compare.split(",")]
        os.environ["COMPARE_MODE"] = "true"
        os.environ["COMPARE_PLATFORMS"] = ",".join(compare_platforms)
        os.environ["COMPARE_LAYOUT"] = args.layout
        os.environ["COMPARE_TOP_N"] = str(args.top_n)
        print(f"📊 启用平台对比模式：{', '.join(compare_platforms)}")
        print(f"   布局：{args.layout}, 每个平台显示 Top {args.top_n}")
    
    run(config_path=args.config)


if __name__ == "__main__":
    main()