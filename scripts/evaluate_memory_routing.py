"""冻结后分进程比较旧快照与当前记忆：固定问答模型/Embedding，不混用向量空间。"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LIVE = Path(__file__).resolve().parents[1]
ARMS = ("recent", "memory_v1", "memory_v2", "full_history")


def sha(path):
    """只哈希代码/合成题/结果，不读取环境密钥。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(root):
    """包含所有生产输入、提示词与这次执行/评分代码。"""
    files=[]
    for folder in ("agent","api","db","memory","model","services","utils","config","prompts","rag"):
        files.extend(p for p in (root/folder).rglob("*") if p.is_file() and p.suffix in (".py",".yaml",".yml",".txt"))
    files.extend(p for p in (root/"scripts").glob("*.py"))
    files.extend(p for p in (root/"evals").glob("*.py"))
    return {str(p.resolve()):sha(p) for p in sorted(set(files))}


def verify(protocol):
    """每个工作进程运行前后核对同一协议，冻结后源码变化直接失败。"""
    record=json.loads(protocol.read_text(encoding="utf-8"))
    if sha(Path(record["dataset"]))!=record["dataset_sha256"]:
        raise ValueError("冻结后题目发生变化")
    for name,digest in record["sources"].items():
        if sha(Path(name))!=digest:
            raise ValueError("冻结后源码变化："+name)
    return record


def recent_context(case, options):
    """近期基线同样携带轮次ID，确保compose_context按完整问答处理，而不借用SQL/记忆。"""
    from evals.history_metrics import visible_history
    from memory.context_builder import compose_context
    current=[r for r in visible_history(case,own_only=True,completed_only=True) if r["session_id"]==case["session_id"]]
    rows=[]
    for index,row in enumerate(current):
        rows.extend([dict(id=index*2+1,turn_id=row["id"],role="user",content=row["question"]),
                     dict(id=index*2+2,turn_id=row["id"],role="assistant",content=row["answer"])])
    rows.append(dict(id=len(rows)+1,turn_id="current",role="user",content=case["question"]))
    return compose_context(rows[-options["history_window"]:],"",[],[],options["context_budget"],
                           as_of=datetime.fromisoformat(case["question_time"]).timestamp())


def freeze(args):
    """新建协议，明确模型、预算、评分和范围；禁止覆盖任何已有验收。"""
    sys.path.insert(0,str(LIVE))
    import yaml
    from evals.history_metrics import validate_dataset
    dataset=json.loads(args.dataset.read_text(encoding="utf-8"))
    cases=validate_dataset(dataset)
    record=dict(dataset=str(args.dataset.resolve()),dataset_sha256=sha(args.dataset),case_ids=[c["id"] for c in cases],
        frozen_at=datetime.now(timezone.utc).isoformat(),model="qwen3-omni-flash",embedding="text-embedding-v3",dimensions=1024,
        parameters=dict(temperature=0,max_tokens=512,max_retries=0,min_call_interval=1.5),
        service=yaml.safe_load((LIVE/"config/memory.yml").read_text(encoding="utf-8")),
        prices=yaml.safe_load((LIVE/"config/model_prices.yml").read_text(encoding="utf-8")),
        sources={**sources(LIVE),**sources(args.v1_source)},v1_source=str(args.v1_source.resolve()),v2_source=str(LIVE),arms=ARMS,
        primary_metric="全部固定题分母，严格结构/答案完整性/拒答标记/禁止旧值的联合规则通过率；原答留待语义复核，不称人工准确率。",
        limits=["内部合成题，作者参与实现，非外部独立基准。",
          "各方案分进程顺序运行，同题同一模型参数，但提供方后端版本/时间漂移无法完全控制。",
          "历史准备运行各自生产事实分析与SQL事务；不重放过去业务工具和过去回答模型。",
          "提问运行各自ChatService上下文准备；普通答案用同一结构化评测提示，不是完整JWT/业务工具Agent验收。",
          "旧快照与新实现都固定v3重新建各自冷索引；不能把原v4首测与这轮直接归因于代码。",
          "原始历史参考包含撤回指令与旧原文，由回答模型执行遗忘要求。",
          "成本包括本方案历史分析、当前分析、摘要、索引建设、查询和回答；实际账单未知。",
          "SDK内部未暴露的HTTP重试不另行猜测；401/403停止后续批量。"])
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"protocol.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"cases":len(cases),"dataset_sha256":record["dataset_sha256"],"model":record["model"],"embedding":record["embedding"]}))


async def worker(args):
    """一个实现一个进程，防止Python模块缓存把旧类与新类混在同次比较。"""
    record=verify(args.protocol)
    source=Path(record["v1_source"] if args.arm=="memory_v1" else record["v2_source"])
    sys.path.insert(0,str(source))
    from dotenv import load_dotenv
    load_dotenv(LIVE/".env",override=True)
    os.environ["LANGFUSE_PUBLIC_KEY"]=os.environ["LANGFUSE_SECRET_KEY"]=""
    from langchain_openai import ChatOpenAI,OpenAIEmbeddings
    from model.usage import UsageLedger,usage_scope,MeteredEmbeddingClient,AsyncMeteredEmbeddingClient
    from scripts.evaluate_history import CountedModel,CountedEmbeddings,RequestGate,answer_one,full_context
    from evals.history_metrics import visible_history,answer_score
    from memory.context_builder import compose_context
    from db.memory_repository import MemoryRepository
    from services.chat_service import ChatService
    from memory.session_lease import SessionLeaseStore
    from memory.history_retrieval import HistoryRetriever
    from memory.history_index import HistorySemanticIndex
    from memory.semantic_index import SemanticMemoryIndex
    model=ChatOpenAI(model=record["model"],api_key=os.getenv("DASHSCOPE_API_KEY"),base_url=os.getenv("DASHSCOPE_BASE_URL"),
        streaming=True,stream_usage=True,temperature=0,max_tokens=512,max_retries=0).bind(extra_body={"enable_thinking":False})
    embedding=OpenAIEmbeddings(model=record["embedding"],dimensions=record["dimensions"],api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL"),check_embedding_ctx_length=False,chunk_size=10,max_retries=0)
    embedding.client=MeteredEmbeddingClient(embedding.client,embedding.model)
    embedding.async_client=AsyncMeteredEmbeddingClient(embedding.async_client,embedding.model)
    gate=RequestGate(record["parameters"]["min_call_interval"])
    target=args.protocol.parent/args.arm
    target.mkdir(exist_ok=False)
    cases=json.loads(Path(record["dataset"]).read_text(encoding="utf-8"))["cases"]
    for ordinal,case in enumerate(cases):
        events=[]
        ledger=UsageLedger(record["prices"])
        row=dict(id=case["id"],category=case["category"],arm=args.arm,calls=events,human_correct=None)
        repository=history_index=fact_index=service=turn=None
        started=time.perf_counter()
        try:
            if gate.blocked_status:
                raise RuntimeError("not_run_provider_access_"+str(gate.blocked_status))
            with usage_scope(ledger,args.arm):
                if args.arm in ("recent","full_history"):
                    if args.arm=="full_history":
                        context=full_context(case)
                    else:
                        context=recent_context(case,record["service"]["service"])
                else:
                    directory=target/case["id"]
                    directory.mkdir()
                    repository=MemoryRepository("sqlite:///"+(directory/"memory.db").as_posix())
                    repository.initialize()
                    analysis_model=CountedModel(model,events,args.arm+"/analysis",gate)
                    if args.arm=="memory_v1":
                        from memory.fact_extractor import FactExtractor
                        analyzer=FactExtractor(analysis_model)
                    else:
                        from memory.request_analyzer import MemoryRequestAnalyzer
                        analyzer=MemoryRequestAnalyzer(analysis_model,timeout=record["service"]["analysis"]["timeout"])
                    preparation=[]
                    identities={}
                    for event in visible_history(case):
                        user=event["user_id"]
                        repository.clock=lambda event=event:datetime.fromisoformat(event["created_at"]).timestamp()
                        scope=ChatService.session_scope(user,event["session_id"])
                        assert repository.claim(scope,user,event["session_id"],"seed",3600)
                        if args.arm=="memory_v1":
                            operations=await analyzer.extract(event["question"])
                        else:
                            planned=await analyzer.analyze(event["question"],memories=repository.deletion_targets(user))
                            operations=planned.operations
                            preparation.append(dict(model_called=planned.model_called,degraded=planned.degraded))
                        identity,_=repository.start_request(scope,user,"seed",event["id"],event["question"],operations)
                        identities[identity]=event["id"]
                        if event["status"]=="completed":
                            repository.finish_request(scope,"seed",identity,event["answer"])
                        repository.release(scope,"seed")
                    repository.clock=lambda:datetime.fromisoformat(case["question_time"]).timestamp()
                    counted=CountedEmbeddings(embedding)
                    options=record["service"]
                    sem={k:v for k,v in options["history_retrieval"]["semantic"].items() if k not in ("enabled","directory")}
                    history_index=HistorySemanticIndex(repository,directory/"history",counted,**sem)
                    fact_index=SemanticMemoryIndex(repository,directory/"facts",counted,min_similarity=options["semantic"]["min_similarity"])
                    kwargs={"fact_extractor" if args.arm=="memory_v1" else "request_analyzer":analyzer}
                    class ObservedHistory(HistoryRetriever):
                        """记录真实服务是否触发检索及返回来源，不额外预热索引或借用标签召回。"""
                        def search(self, *values, **filters):
                            """原样委托生产检索器，仅在完成后记录结果ID。"""
                            row["history_search_called"]=True
                            found=super().search(*values,**filters)
                            row["history_returned_ids"]=[identities.get(item["turn_id"],"UNMAPPED") for item in found]
                            return found
                    service=ChatService(None,SessionLeaseStore(),repository,
                        summarizer=CountedModel(model,events,args.arm+"/summary",gate),semantic_index=fact_index,
                        history_retriever=ObservedHistory(repository,semantic_index=history_index),**options["service"],**kwargs)
                    with usage_scope(stage=args.arm+"/context"):
                        turn=await service.open_turn(case["session_id"],case["question"],case["user_id"])
                    context=turn.messages
                    row.update(compression=str(turn.compression),analysis=getattr(turn,"memory_analysis",None),historical_analysis=preparation)
                row["context_setup_seconds"]=time.perf_counter()-started
                row["context"]=context
                if turn and getattr(turn,"direct_answer",""):
                    answer=dict(answer=turn.direct_answer,abstain=True,structure_valid=True,calls=[],seconds=0,
                        deterministic_service_reply=True,**answer_score(case,turn.direct_answer,True))
                else:
                    answer=await answer_one(CountedModel(model,events,args.arm+"/answer",gate),case,context,events)
                row["answer"]=answer
                if turn:
                    repository.finish_request(turn.scope,turn.owner,turn.turn_id,answer.get("raw",answer.get("answer","")))
        except Exception as error:
            row["error"]=type(error).__name__+":"+str(error)[:200]
        finally:
            if service and turn:
                await service.close_turn(turn)
            if history_index:
                history_index.close()
            if fact_index:
                fact_index.close()
            if repository:
                repository.engine.dispose()
        row.update(wall_seconds=time.perf_counter()-started,usage=ledger.report())
        with (target/"results.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
        print(f"{args.arm} [{ordinal+1}/{len(cases)}] {case['id']} passed={row.get('answer',{}).get('passed',False)} error={row.get('error','')}",flush=True)
    verify(args.protocol)
    (target/"verified.json").write_text(json.dumps(dict(verified=True,protocol_sha256=sha(args.protocol)),indent=2),encoding="utf-8")


def report(args):
    """汇总首轮全部结果；缺失项不从分母中移除，费用不足则明确不完整。"""
    record=verify(args.protocol)
    output={"protocol_sha256":sha(args.protocol),"cases":len(record["case_ids"]),"arms":{}}
    cases=json.loads(Path(record["dataset"]).read_text(encoding="utf-8"))["cases"]
    for arm in ARMS:
        path=args.protocol.parent/arm/"results.jsonl"
        rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        output["arms"][arm]=dict(recorded=len(rows),passed=sum(r.get("answer",{}).get("passed",False) for r in rows),
            rule_pass_rate=sum(r.get("answer",{}).get("passed",False) for r in rows)/len(record["case_ids"]),
            errors=sum("error" in r for r in rows),forbidden=sum(bool(r.get("answer",{}).get("forbidden_matches")) for r in rows),
            input_tokens=sum(e["input_tokens"] or 0 for r in rows for e in r["usage"]["events"]),
            output_tokens=sum(e["output_tokens"] or 0 for r in rows for e in r["usage"]["events"]),
            reference_cost=sum(r["usage"]["reference_cost_known_subtotal"] for r in rows),
            missing_usage_or_price=sum(r["usage"]["unpriced_or_missing_calls"] for r in rows),
            actual_paid=None)
        by_id={r["id"]:r for r in rows}
        needed=[c for c in cases if c["category"] in ("imperative","multi_session","forget","no_evidence","note_forget")]
        optional=[c for c in cases if c not in needed]
        if arm.startswith("memory_"):
            output["arms"][arm]["history_routing"]=dict(required_cases=len(needed),
                recall=sum(by_id.get(c["id"],{}).get("history_search_called",False) for c in needed)/len(needed),
                optional_trigger_rate=sum(by_id.get(c["id"],{}).get("history_search_called",False) for c in optional)/len(optional),
                scope="本40题预定义事件/回查类别需要历史；其他题事实读取足够，额外历史读取不必然是答案错误。")
            queries=[r for r in rows if r.get("analysis")]
            history_analysis=[a for r in rows for a in r.get("historical_analysis",[])]
            output["arms"][arm]["analyzer"]=dict(current_model_calls=sum(r["analysis"]["model_called"] for r in queries),
                current_degraded=sum(r["analysis"]["degraded"] for r in queries),historical_model_calls=sum(a["model_called"] for a in history_analysis),
                historical_degraded=sum(a["degraded"] for a in history_analysis))
    (args.protocol.parent/"summary.json").write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("freeze","worker","report"))
    parser.add_argument("--dataset",type=Path)
    parser.add_argument("--v1-source",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--protocol",type=Path)
    parser.add_argument("--arm",choices=ARMS)
    args=parser.parse_args()
    if args.mode=="freeze":
        freeze(args)
    elif args.mode=="worker":
        asyncio.run(worker(args))
    else:
        report(args)
