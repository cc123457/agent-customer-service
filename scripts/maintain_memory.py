"""记忆维护命令；默认只预览，--apply才执行已验证归档和版本清理，适合外部调度器周期运行。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    """从服务端配置读取期限与目录，不输出密码；每次只处理有限批次，失败可重试。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=True)
    from db.memory_factory import get_memory_repository
    from memory.archive import MemoryArchive
    from utils.config_handler import memory_conf
    from utils.path_tool import get_abs_path
    repo = get_memory_repository()
    policy = memory_conf["retention"]
    archive = MemoryArchive(repo, get_abs_path(policy["archive_directory"]))
    now = repo.clock()
    result = {"archive": archive.archive(now - policy["archive_after_days"] * 86400, limit=args.limit, apply=args.apply),
              "cold_expiry": archive.retire_archives(now - policy["cold_archive_days"] * 86400, args.apply, args.limit),
              "versions": archive.prune_versions(policy["summary_versions_to_keep"], args.apply),
              "checkpoints": archive.prune_checkpoints(now - policy["completed_checkpoint_days"] * 86400,
                                                      now - policy["failed_checkpoint_days"] * 86400, args.apply, args.limit)}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
