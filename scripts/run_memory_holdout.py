"""冻结并执行记忆问答新题；整轮固定实现、数据、模型和预算，失败不自动换模型重跑。"""
import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    """对文件原始字节取指纹；不读取或记录密钥和运行数据库。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_files():
    """覆盖生产代码、提示词、配置及本轮生成/执行/评分工具，排除环境密钥与原始业务数据。"""
    paths = []
    for directory in ("agent", "api", "db", "memory", "model", "services", "utils", "config", "prompts"):
        paths.extend(path for path in (ROOT / directory).rglob("*") if path.is_file() and path.suffix in (".py", ".yml", ".yaml", ".txt"))
    paths.extend(ROOT / name for name in ("scripts/evaluate_history.py", "scripts/run_memory_holdout.py",
        "scripts/review_memory_holdout.py", "evals/history_metrics.py", "evals/build_memory_qa_holdout.py"))
    return sorted(set(paths))


def freeze(dataset, target):
    """首次调用前固化80题、十类配比及评分约定，禁止覆盖旧冻结文件。"""
    from collections import Counter
    from evals.history_metrics import validate_dataset
    cases = validate_dataset(json.loads(dataset.read_text(encoding="utf-8")))
    if len(cases) != 80 or any(count != 8 for count in Counter(c["category"] for c in cases).values()):
        raise ValueError("本版预注册为十类各8题")
    if target.exists():
        raise FileExistsError("冻结文件不可覆盖")
    for previous in ("history_cases.json", "refinement_cases.json", "temporal_boundary_cases.json"):
        old = json.loads((ROOT / "evals" / previous).read_text(encoding="utf-8"))["cases"]
        for field in ("id", "user_id", "question"):
            if {case[field] for case in old} & {case[field] for case in cases}:
                raise ValueError("新旧题目、用户或编号重叠：" + field)
    record = {"version": "memory-qa-holdout-v1", "frozen_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset.resolve()), "dataset_sha256": digest(dataset), "case_ids": [c["id"] for c in cases],
        "categories": dict(Counter(c["category"] for c in cases)), "answerable": sum(not c["should_abstain"] for c in cases),
        "unanswerable": sum(c["should_abstain"] for c in cases),
        "implementation_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in protected_files()},
        "model": "qwen3-omni-flash", "embedding_model": "text-embedding-v4",
        "parameters": {"temperature": 0, "max_tokens": 512, "enable_thinking": False, "max_retries": 1,
                       "concurrency": 1, "min_call_interval": 1.5},
        "arms": ["recent", "summary_recent", "memory", "full_history"],
        "primary_metric": "逐题AI语义复核的完整任务通过数/80：答案完整正确、拒答标记正确、不泄露遗忘/外人值；运行失败也保留在分母。",
        "secondary_metrics": ["固定规则通过率", "有证据题内容正确率", "无证据/遗忘/不确定题拒答表现", "检索Recall@3", "95% Wilson区间", "配对准确率差异", "真实usage与参考费用"],
        "review_protocol": "当前助手逐条检查全部320个方案输出及相应证据；不把规则分数复制为AI评分；实际人工评分留空。评审者参与数据构建，不称外部独立盲评。",
        "failure_policy": "不自动切模型、不挑最佳重跑、不删除超时/格式/语义错误；首轮固定实现报告。修复需新一版验收。",
        "scope": "内部新题冻结验收，不是外部基准或线上准确率。历史事实规则提取，真实摘要/Embedding/最终回答；不运行JWT/产品知识库/业务工具Agent。"}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def verify(path):
    """执行前后验证原文、题目顺序与所有受保护源码没有变化。"""
    record = json.loads(path.read_text(encoding="utf-8"))
    if digest(Path(record["dataset"])) != record["dataset_sha256"]:
        raise ValueError("验收题目发生变化")
    for name, expected in record["implementation_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("冻结后源码变化：" + name)
    return record


async def execute(frozen, output, env_file):
    """复用既有真实四方案评测器，模型和节流设置只来自已冻结协议。"""
    from scripts.evaluate_history import run
    record = verify(frozen)
    if output.exists():
        raise FileExistsError("正式运行输出目录必须不存在，避免拼接/覆盖")
    output.mkdir(parents=True)
    (output / "holdout_protocol.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    params = record["parameters"]
    args = SimpleNamespace(dataset=Path(record["dataset"]), split="test", limit=None, live=True, embedding=False,
        env_file=env_file, chat_model=record["model"], concurrency=params["concurrency"],
        min_call_interval=params["min_call_interval"], output=output)
    await run(args)
    verify(frozen)
    (output / "freeze_verified.json").write_text(json.dumps({"verified": True, "freeze_sha256": digest(frozen),
        "dataset_sha256": record["dataset_sha256"], "cases": len(record["case_ids"]),
        "verified_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    """freeze只写协议，run才调用模型；所有写入都使用新的路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run", "verify"))
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evals/memory_qa_holdout.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()
    if args.mode == "freeze":
        record = freeze(args.dataset, args.freeze)
        print(json.dumps({"cases": len(record["case_ids"]), "model": record["model"], "dataset_sha256": record["dataset_sha256"]}, ensure_ascii=False))
    elif args.mode == "verify":
        verify(args.freeze)
        print("冻结指纹验证通过")
    else:
        if args.output is None:
            parser.error("run需要--output")
        asyncio.run(execute(args.freeze, args.output, args.env_file))


if __name__ == "__main__":
    main()
