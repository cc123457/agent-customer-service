"""仅初始化历史/长期记忆数据库，不执行旧演示库的DROP/CREATE脚本。"""
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    """读取显式环境配置，增建初始记忆表并打印后端类型；已有表/原文不删除，不输出连接密码。"""
    parser = argparse.ArgumentParser(description="增建记忆数据库表，不清理数据")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("环境配置文件不存在")
        load_dotenv(args.env_file, override=True)
    from db.memory_factory import get_memory_repository
    repository = get_memory_repository()
    print(f"Memory schema ready: {repository.engine.dialect.name}; existing data preserved")


if __name__ == "__main__":
    main()
