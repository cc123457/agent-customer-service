import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def percentile(values, fraction):
    """输入耗时样本和分位比例，按最近秩法返回分位值；无样本返回 None。"""
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


async def contract_cases(cases):
    """输入请求校验案例列表，向真实 FastAPI 路由发 ASGI 请求，返回逐例结果。

    替换认证与业务服务，不调用模型；验证字段规则，不代表 JWT 或回答质量。
    """
    import httpx
    from fastapi import FastAPI
    from api.chat import router
    from api.deps import get_chat_service, get_limited_user

    class NoModelService:
        async def open_turn(self, *args):
            """校验套件的禁止调用入口：非法请求若进入业务层，立即报错暴露问题。"""
            raise AssertionError("非法请求不应进入业务服务")

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_chat_service] = lambda: NoModelService()
    app.dependency_overrides[get_limited_user] = lambda: "1001"
    results = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://eval") as client:
        for case in cases:
            started = time.perf_counter()
            response = await client.post("/chat", json=case["request"])
            results.append({
                "id": case["id"], "suite": "contract", "passed": response.status_code == case["expected_status"],
                "status_code": response.status_code, "latency_seconds": time.perf_counter() - started,
                "calls": 0, "note": "真实 ASGI 请求校验；认证依赖被替换，不调用模型。",
            })
    return results


async def agent_cases(cases, agent, concurrency):
    """通过当前ChatService运行案例；每例使用临时SQL、独立检查点和个人索引，不读写真实聊天档案。

    agent为空时为每例组装真实Agent；测试可注入可控Agent，过程结束不等于答案正确。
    """
    import tempfile
    from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler
    from db.memory_repository import MemoryRepository
    from memory.session_lease import SessionLeaseStore
    from services.chat_service import ChatService
    from utils.config_handler import memory_conf

    semaphore = asyncio.Semaphore(concurrency)

    class Calls(BaseCallbackHandler):
        def __init__(self):
            """为单个案例初始化模型调用计数，不执行模型请求。"""
            self.count = 0

        def on_chat_model_start(self, serialized, messages, **kwargs):
            """每次模型开始回调累加一次，不冒充供应商HTTP重试次数。"""
            self.count += 1

    async def evaluate(case):
        """独立组装当前服务，执行多轮并在finally关闭索引、连接池和临时目录。"""
        async with semaphore:
            with tempfile.TemporaryDirectory(prefix="agent-eval-") as directory:
                repo = MemoryRepository("sqlite:///" + (Path(directory) / "memory.db").as_posix())
                index = None
                try:
                    repo.initialize()
                    usage, calls = UsageMetadataCallbackHandler(), Calls()
                    if agent is None:
                        from agent.react_agent import ReactAgent
                        from memory.checkpointer import SqlCheckpointSaver
                        from memory.request_analyzer import MemoryRequestAnalyzer
                        from model.factory import chat_model
                        actual_agent = await asyncio.to_thread(ReactAgent, checkpointer=SqlCheckpointSaver(repo.engine))
                        extractor = MemoryRequestAnalyzer(chat_model if memory_conf.get("analysis", {}).get("model_enabled") else None)
                        if memory_conf.get("semantic", {}).get("enabled"):
                            from memory.semantic_index import SemanticMemoryIndex
                            from model.factory import embed_model
                            index = SemanticMemoryIndex(repo, Path(directory) / "personal-index", embed_model,
                                                        min_similarity=memory_conf["semantic"].get("min_similarity", 0.3))
                    else:
                        actual_agent, extractor = agent, None

                    class ObservedAgent:
                        @property
                        def checkpointer(self):
                            """透传本例Agent检查点，服务据此传入独立执行ID。"""
                            return getattr(actual_agent, "checkpointer", None)

                        async def execute_stream(self, messages, user_id, **kwargs):
                            """保留执行ID与当前身份，额外附加模型调用/用量回调。"""
                            async for event in actual_agent.execute_stream(messages, user_id=user_id, callbacks=[usage, calls], **kwargs):
                                yield event

                    service = ChatService(ObservedAgent(), SessionLeaseStore(), repo, request_analyzer=extractor,
                                          semantic_index=index, **memory_conf["service"])
                    started = time.perf_counter()
                    answers, actions = [], []
                    error_type, first_answer_seconds = None, None
                    try:
                        for question in case["turns"]:
                            turn = await service.open_turn(case["id"], question, case["user_id"])
                            parts = []
                            async for item in service.reply_turn(turn, case["user_id"]):
                                if item["type"] == "answer":
                                    if first_answer_seconds is None:
                                        first_answer_seconds = time.perf_counter() - started
                                    parts.append(item["content"])
                                elif item["type"] == "action":
                                    actions.append(item["content"])
                            answers.append("".join(parts))
                    except Exception as error:
                        error_type = type(error).__name__
                    final = answers[-1] if answers else ""
                    terms = case.get("expected_terms", [])
                    result = {
                        "id": case["id"], "suite": "agent", "category": case["category"],
                        "completed": error_type is None and len(answers) == len(case["turns"]),
                        "error_type": error_type, "latency_seconds": time.perf_counter() - started,
                        "first_answer_event_seconds": first_answer_seconds, "calls": calls.count,
                        "usage_by_model": usage.usage_metadata, "tool_actions": actions, "answers": answers,
                        "term_check": all(term.lower() in final.lower() for term in terms) if terms else None,
                        "rubric": case["rubric"], "human_correct": None, "human_notes": "",
                    }
                    print(f"{case['id']}: completed={result['completed']} calls={calls.count} seconds={result['latency_seconds']:.2f}", file=sys.stderr, flush=True)
                    return result
                finally:
                    if index:
                        index.close()
                    repo.engine.dispose()

    return await asyncio.gather(*(evaluate(case) for case in cases))


def main():
    """命令行入口：选择案例/环境配置、运行评测，将结果和汇总保存到输出目录。

    默认只执行无模型的 contract 套件；agent/all 会产生真实 API 调用费用。
    输出包含本地日志和回答，不打印密钥，质量准确率留待人工标注。
    """
    parser = argparse.ArgumentParser(description="当前聊天服务固定评测；agent 模式会调用真实模型并产生费用")
    parser.add_argument("--suite", choices=["contract", "agent", "all"], default="contract")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-file", type=Path, help="显式选择模型配置，仅在本进程覆盖环境，不打印密钥")
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        if not args.env_file.is_file():
            parser.error("指定的环境配置文件不存在")
        load_dotenv(args.env_file, override=True)
    os.environ["LANGFUSE_PUBLIC_KEY"] = ""
    os.environ["LANGFUSE_SECRET_KEY"] = ""
    if args.concurrency < 1 or (args.limit is not None and args.limit < 1):
        parser.error("concurrency 和 limit 必须大于 0")
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "evals/agent_cases.json").read_text(encoding="utf-8"))
    selected = [case for case in cases if args.suite == "all" or case["suite"] == args.suite]
    if args.limit:
        selected = selected[:args.limit]
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    output = args.output or root / "evals/runs" / stamp
    output.mkdir(parents=True, exist_ok=True)
    with (output / "trace.log").open("w", encoding="utf-8") as trace, contextlib.redirect_stdout(trace):
        results = asyncio.run(contract_cases([case for case in selected if case["suite"] == "contract"]))
        live = [case for case in selected if case["suite"] == "agent"]
        if live:
            logging.getLogger("agent").setLevel(logging.WARNING)
            results.extend(asyncio.run(agent_cases(live, None, args.concurrency)))
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    live_results = [result for result in results if result["suite"] == "agent"]
    contract_results = [result for result in results if result["suite"] == "contract"]
    measured_usage = [usage for result in live_results for usage in result["usage_by_model"].values()]
    from utils.config_handler import memory_conf, rag_conf
    summary = {
        "timestamp": stamp, "dataset_total": len(cases), "executed": len(results),
        "dataset_sha256": hashlib.sha256((root / "evals/agent_cases.json").read_bytes()).hexdigest(),
        "configured_model": rag_conf["chat_model_name"], "memory_config": memory_conf,
        "concurrency": args.concurrency,
        "contract_count": len(contract_results), "contract_passed": sum(result["passed"] for result in contract_results),
        "agent_count": len(live_results), "agent_completed": sum(result["completed"] for result in live_results),
        "agent_model_calls": sum(result["calls"] for result in live_results),
        "case_latency_p50_seconds": percentile([result["latency_seconds"] for result in live_results], 0.5),
        "case_latency_p95_seconds": percentile([result["latency_seconds"] for result in live_results], 0.95),
        "reported_input_tokens": sum(usage["input_tokens"] for usage in measured_usage) if measured_usage else None,
        "reported_output_tokens": sum(usage["output_tokens"] for usage in measured_usage) if measured_usage else None,
        "quality_accuracy": None,
        "limitations": [
            "completed 表示流程结束，不代表回答正确；关键词命中只供筛查，需按 rubric 人工复核。",
            "多轮案例耗时包含所有轮次，first_answer_event_seconds 不是逐 Token 的 TTFT。",
            "Token 来自模型回调报告，不含 Embedding；未报告用量的调用可能不在汇总中。",
            "agent 测试直连业务服务，使用独立临时SQL档案、检查点和个人索引；不替代 HTTP/JWT/Redis 集成测试。",
            "这是改动后的基线，不是旧版本对照，不可由此宣称效果提升比例。",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output)


if __name__ == "__main__":
    main()
