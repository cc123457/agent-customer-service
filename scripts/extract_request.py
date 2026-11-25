import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.factory import chat_model
from services.structured_output import CustomerRequestExtractor


async def main():
    """命令行读取用户描述，调用项目真实模型提取诉求，打印校验后的 JSON。

    仅直接运行本脚本时触发，不自动接入 /chat；模型调用可能产生费用。
    """
    parser = argparse.ArgumentParser(description="客服诉求 JSON 输出实验")
    parser.add_argument("text", help="用户的客服诉求")
    args = parser.parse_args()
    result = await CustomerRequestExtractor(chat_model).extract(args.text)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
