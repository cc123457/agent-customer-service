"""抽取质量与摘要事实保留评测：默认规则抽取，--live运行真实模型兜底及真实摘要。"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

from evals.quality_metrics import candidate_dict, extraction_aggregate, extraction_score, fact_present, summary_aggregate, summary_score
from scripts.evaluate_history import CountedModel, RequestGate, fingerprint, usage_total


class ObservedModel(CountedModel):
    """保存模型原始提取输出供错误分析，不让评分器参与生产候选校验。"""
    async def ainvoke(self, messages):
        """真实调用与usage沿用计量包装，另存原答，便于区分模型错误和应用过滤。"""
        response = await super().ainvoke(messages)
        self.events[-1]["raw"] = response.content
        return response


def validate_quality(dataset):
    """冻结前检查别名原文依据、拆分、事实引用，标签不得依赖模型生成内容。"""
    for suite in ("extraction", "summary"):
        cases = dataset[suite]
        if len({case["id"] for case in cases}) != len(cases):
            raise ValueError("案例ID重复")
        for case in cases:
            if case["split"] not in ("calibration", "test"):
                raise ValueError("无效split")
            if suite == "extraction":
                for fact in case["expected"]:
                    if not fact["values"] or not all(value in case["text"] for value in fact["values"]):
                        raise ValueError("抽取标签值无原文依据：" + case["id"])
                    if fact["valid_at"] and fact["valid_at"] not in case["text"]:
                        raise ValueError("有效时间无原文依据")
            else:
                source = case["old_summary"] + "\n" + "\n".join(row["content"] for row in case["messages"])
                if not all(fact_present(source, fact["groups"]) for fact in case["facts"]):
                    raise ValueError("摘要标签无原文依据：" + case["id"])
    return dataset


async def extract_case(case, model, gate):
    """比较现有规则与规则优先/模型补充，均调用生产MemoryRequestAnalyzer，不预先写死预测。"""
    from memory.request_analyzer import MemoryRequestAnalyzer
    result = {"id": case["id"], "category": case["category"], "suite": "extraction", "modes": {}}
    for mode in (("rules", "rules_model") if model else ("rules",)):
        events = []
        observed = ObservedModel(model, events, "extraction", gate) if mode == "rules_model" else None
        started = time.perf_counter()
        error = None
        try:
            predictions = [candidate_dict(candidate) for candidate in await MemoryRequestAnalyzer(observed).extract(case["text"])]
            if any("error" in event for event in events):
                error = "model_call_failed_fallback"
        except Exception as exception:
            predictions, error = [], type(exception).__name__
        result["modes"][mode] = {"predictions": predictions, "score": extraction_score(case, predictions),
                                "seconds": time.perf_counter() - started, "calls": events, "error": error}
    return result


async def summarize_case(case, model, gate, options):
    """直接运行生产分批摘要器；只给原文和旧摘要，不向模型泄漏关键事实标签。"""
    from memory.batch_summary import BatchSummarizer
    events, started = [], time.perf_counter()
    source = case["old_summary"] + "\n" + "\n".join(row["content"] for row in case["messages"])
    result = {"id": case["id"], "category": case["category"], "suite": "summary", "calls": events,
              "source_bytes": len(source.encode("utf-8")), "summary": "", "batches": 0}
    try:
        summarizer = BatchSummarizer(ObservedModel(model, events, "summary", gate),
                    options["summary_input_budget"], options["summary_budget"], options["summary_timeout"], options["max_summary_batches"])
        result["summary"], result["batches"] = await asyncio.wait_for(
            summarizer.summarize(case["old_summary"], case["messages"]), options["compression_timeout"])
    except Exception as error:
        result["error"] = type(error).__name__
    result.update(score=summary_score(case, result["summary"]), seconds=time.perf_counter() - started,
                  summary_bytes=len(result["summary"].encode("utf-8")))
    return result


def aggregate_quality(rows):
    """分开报告候选抽取与摘要保留，不把结构合法率冒充语义质量。"""
    extraction = [row for row in rows if row["suite"] == "extraction"]
    summaries = [row for row in rows if row["suite"] == "summary"]
    modes = sorted({mode for row in extraction for mode in row["modes"]})
    events = [event for row in summaries for event in row["calls"]]
    events += [event for row in extraction for mode in row["modes"].values() for event in mode["calls"]]
    return {"extraction": {mode: {**extraction_aggregate([row["modes"][mode] for row in extraction]),
                                  "actual_model_cases": sum(bool(row["modes"][mode]["calls"]) for row in extraction)} for mode in modes},
            "summary": summary_aggregate(summaries) if summaries else None, "actual_model_usage": usage_total(events)}


async def run(args):
    """先保存标签/实现指纹再执行，失败保留逐题记录；不写业务库、不改提取或摘要实现。"""
    import yaml
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=True)
    os.environ["LANGFUSE_PUBLIC_KEY"] = os.environ["LANGFUSE_SECRET_KEY"] = ""
    data = validate_quality(json.loads(args.dataset.read_text(encoding="utf-8")))
    suites = ("extraction", "summary") if args.suite == "all" else (args.suite,)
    if "summary" in suites and not args.live:
        raise ValueError("摘要质量必须使用--live，不能用写死摘要冒充质量结果")
    config = yaml.safe_load((ROOT / "config/memory.yml").read_text(encoding="utf-8"))
    cases = [dict(case, suite=suite) for suite in suites for case in data[suite] if case["split"] == args.split]
    if args.limit:
        cases = cases[:args.limit]
    paths = [ROOT / path for path in ("memory/request_analyzer.py", "memory/long_term.py", "memory/batch_summary.py",
              "memory/fact_rules.py", "services/json_payload.py", "model/factory.py", "model/usage.py", "config/memory.yml", "config/rag.yml")]
    manifest = {"mode": "real_model" if args.live else "rules_no_model", "split": args.split, "suite": args.suite,
                "case_ids": [case["suite"] + ":" + case["id"] for case in cases], "scope": data["scope"],
                "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                "implementation_sha256": fingerprint(paths), "min_call_interval": args.min_call_interval,
                "evaluator_sha256": fingerprint([Path(__file__), ROOT / "evals/quality_metrics.py", ROOT / "scripts/evaluate_history.py"])}
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("请选择新的输出目录，避免覆盖已有评测")
    model = None
    if args.live:
        from model.factory import chat_model
        chosen = args.chat_model or chat_model.model_name
        configured = chat_model.model_copy(update={"model_name": chosen, "streaming": True, "stream_usage": True, "temperature": 0, "max_tokens": 512, "max_retries": 1})
        model = configured.bind(extra_body={"enable_thinking": False}) if chosen.startswith("qwen") else configured
        manifest["chat_model"] = chosen
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rows, gate = [], RequestGate(args.min_call_interval)
    for case in cases:
        if gate.blocked_status:
            if case["suite"] == "extraction":
                row = await extract_case(case, None, gate)
                row["modes"]["rules_model"] = {"predictions": [], "score": extraction_score(case, []),
                    "calls": [], "error": "not_run_provider_access_" + str(gate.blocked_status)}
            else:
                source = case["old_summary"] + "\n" + "\n".join(message["content"] for message in case["messages"])
                row = {"id": case["id"], "category": case["category"], "suite": "summary", "calls": [], "summary": "", "batches": 0,
                       "source_bytes": len(source.encode()), "summary_bytes": 0, "score": summary_score(case, ""),
                       "error": "not_run_provider_access_" + str(gate.blocked_status)}
            row["status"] = "not_run_model"
        else:
            row = None
        if row is None:
            row = (await extract_case(case, model, gate) if case["suite"] == "extraction" else
                   await summarize_case(case, model, gate, config["service"]))
        rows.append(row)
        with (args.output / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        detail = (" ".join(f"{mode}:TP={result['score']['tp']},FP={result['score']['fp']},FN={result['score']['fn']}" for mode, result in row["modes"].items())
                  if case["suite"] == "extraction" else f"facts={row['score']['retained']}/{row['score']['required']} batches={row['batches']} error={row.get('error')}")
        print(f"[{len(rows)}/{len(cases)}] {case['suite']} {case['id']} {detail}", flush=True)
    if fingerprint(paths) != manifest["implementation_sha256"]:
        raise RuntimeError("执行期间生产实现变化，结果不能作为固定实现比较")
    result = {"manifest": manifest, "provider_access_blocked": gate.blocked_status, "overall": aggregate_quality(rows),
              "categories": {category: aggregate_quality([row for row in rows if row["category"] == category])
                             for category in sorted({row["category"] for row in rows})}}
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))


def main():
    """显式选择真实模型；默认离线规则抽取，模型参数只作用于评测进程。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evals/quality_cases.json")
    parser.add_argument("--suite", choices=("extraction", "summary", "all"), default="extraction")
    parser.add_argument("--split", choices=("calibration", "test"), default="test")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--chat-model", help="仅覆盖评测模型，保留项目默认配置；不能混算不同模型结果")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-call-interval", type=float, default=1.25)
    parser.add_argument("--output", type=Path, default=ROOT / "evals/runs" / ("quality-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    args = parser.parse_args()
    if args.min_call_interval < 0 or (args.limit is not None and args.limit < 1):
        parser.error("请求间隔不得为负，limit必须为正")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
