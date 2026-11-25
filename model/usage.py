"""捕获真实接口usage并按阶段估算参考原价；不记录正文/密钥，不猜测免费额度或实际账单。"""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import Lock

_scope = ContextVar("memory_usage_scope", default=(None, "unscoped"))


def current_usage_stage():
    """供Embedding包装器细分建索引/查询，不暴露其他任务的账本。"""
    return _scope.get()[1]


@contextmanager
def usage_scope(ledger=None, stage=None):
    """为异步任务/传入to_thread的上下文绑定账本；嵌套阶段不串到其他用户或评测案例。"""
    previous, phase = _scope.get()
    token = _scope.set((ledger if ledger is not None else previous, stage or phase))
    try:
        yield
    finally:
        _scope.reset(token)


def record_usage(model, kind, usage=None, *, stage=None, error=None):
    """只收实际报告的非负整数；缺失usage或错误留空，绝不按字节数或调用次数猜Token。"""
    ledger, current_stage = _scope.get()
    if ledger is None:
        return
    value = usage or {}
    if not isinstance(value, dict):
        value = value.model_dump()
    choices = (value.get("total_tokens"), value.get("input_tokens"), value.get("prompt_tokens")) if kind == "embedding" else (value.get("input_tokens"), value.get("prompt_tokens"))
    input_count = next((count for count in choices if type(count) is int and count >= 0), None)
    output_count = 0 if kind == "embedding" and input_count is not None else value.get("output_tokens", value.get("completion_tokens"))
    if type(input_count) is not int or input_count < 0:
        input_count = None
    if type(output_count) is not int or output_count < 0:
        output_count = None
    ledger.add({"model": model, "kind": kind, "stage": stage or current_stage,
                "input_tokens": input_count, "output_tokens": output_count, "error": error})


class UsageLedger:
    """单个案例或请求的线程安全账本；计价是参考原价，实际扣款始终单独留空。"""
    def __init__(self, prices=None):
        """可注入固定单价测试公式；默认读取已记录日期/来源的官方参考价。"""
        if prices is None:
            import yaml
            prices = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/model_prices.yml").read_text(encoding="utf-8"))
        self.prices, self.events, self.lock = prices, [], Lock()

    def add(self, event):
        """只追加Token计量，不保存用户文本、向量或接口地址。"""
        with self.lock:
            self.events.append(dict(event))

    def report(self):
        """按阶段返回已知原价小计和缺失项；免费额度未读取，不能将实际支付金额伪填为0。"""
        with self.lock:
            events = [dict(event) for event in self.events]
        groups = {}
        total, missing = 0.0, 0
        for event in events:
            rate = self.prices["models"].get(event["model"])
            priced = rate and event["input_tokens"] is not None and event["output_tokens"] is not None and not event["error"]
            cost = ((event["input_tokens"] * rate["input_per_million"] + event["output_tokens"] * rate["output_per_million"]) / 1_000_000 if priced else None)
            event["reference_cost"] = cost
            group = groups.setdefault(event["stage"], {"calls": 0, "input_tokens_reported": 0, "output_tokens_reported": 0,
                                      "reference_cost_known_subtotal": 0.0, "unpriced_or_missing_calls": 0})
            group["calls"] += 1
            group["input_tokens_reported"] += event["input_tokens"] or 0
            group["output_tokens_reported"] += event["output_tokens"] or 0
            group["reference_cost_known_subtotal"] += cost or 0
            group["unpriced_or_missing_calls"] += cost is None
            total += cost or 0
            missing += cost is None
        return {"currency": self.prices["currency"], "pricing_mode": self.prices.get("mode", "injected"),
                "prices_verified_on": self.prices.get("verified_on"), "pricing_sources": self.prices.get("sources", []),
                "events": events, "stages": groups, "reference_cost_known_subtotal": total,
                "unpriced_or_missing_calls": missing, "reference_cost_complete": bool(events) and missing == 0,
                "actual_paid": None, "free_quota_applied": None,
                "scope": "接口报告Token的参考原价，不是账单；SDK内部重试未单独暴露时不假装统计全部HTTP尝试。"}


def combine_reports(reports):
    """合并同一运行阶段，诊断检索费用独立，不把其他对照方案调用混入当前记忆成本。"""
    reports = list(reports)
    stages = {}
    for report in reports:
        for stage, values in report["stages"].items():
            group = stages.setdefault(stage, {key: 0 for key in values})
            for key, value in values.items():
                group[key] += value
    current_stages = ("memory", "summary", "memory_context/documents", "memory_context/query")
    memory_cost = sum(stages.get(stage, {}).get("reference_cost_known_subtotal", 0) for stage in current_stages)
    full_cost = stages.get("full_history", {}).get("reference_cost_known_subtotal", 0)
    missing = sum(stages.get(stage, {}).get("unpriced_or_missing_calls", 0) for stage in current_stages + ("full_history",))
    complete = (bool(reports) and missing == 0 and full_cost > 0
                and stages.get("memory", {}).get("calls", 0) >= len(reports)
                and stages.get("full_history", {}).get("calls", 0) >= len(reports))
    return {"cases": len(reports), "stages": stages, "currency": "CNY", "actual_paid": None,
            "pricing_mode": "official_mainland_text_reference", "prices_verified_on": reports[0].get("prices_verified_on") if reports else None,
            "embedding_input_tokens_reported": sum(event["input_tokens"] or 0 for report in reports for event in report["events"] if event["kind"] == "embedding"),
            "embedding_missing_usage_calls": sum(event["input_tokens"] is None for report in reports for event in report["events"] if event["kind"] == "embedding"),
            "memory_cold_reference_cost": memory_cost, "full_history_reference_cost": full_cost,
            "comparison_usage_complete": complete,
            "cold_reference_cost_reduction": 1 - memory_cost / full_cost if complete else None,
            "scope": "当前方案含其实际建索引/查询、摘要及回答；独立诊断检索不计入，事实提取为规则，本次无模型提取费用。每案例冷启动一次，不等于长期摊销费用。"}


class MeteredEmbeddingClient:
    """在LangChain丢弃Embedding响应usage之前采集，每个实际SDK分批分别记账。"""
    def __init__(self, client, model):
        """包装SDK末端客户端，不修改向量、分批设置或重试策略。"""
        self.client, self.model = client, model

    def create(self, **kwargs):
        """同步路径原样返回响应，只提取usage，失败向上传播。"""
        try:
            response = self.client.create(**kwargs)
            value = response if isinstance(response, dict) else response.model_dump()
            record_usage(self.model, "embedding", value.get("usage"))
            return response
        except BaseException as error:
            record_usage(self.model, "embedding", error=type(error).__name__)
            raise


class AsyncMeteredEmbeddingClient(MeteredEmbeddingClient):
    async def create(self, **kwargs):
        """异步路径与同步路径使用同一格式，不把缺失Token当成零。"""
        try:
            response = await self.client.create(**kwargs)
            value = response if isinstance(response, dict) else response.model_dump()
            record_usage(self.model, "embedding", value.get("usage"))
            return response
        except BaseException as error:
            record_usage(self.model, "embedding", error=type(error).__name__)
            raise
