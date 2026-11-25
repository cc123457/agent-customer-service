"""真实临时SQLite与独立Redis的可复现小规模基线；不调用模型，不清空Redis。"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import event
from db.memory_repository import MemoryRepository
from memory.fact_cache import RedisFactCache
from memory.long_term import extract_memories
from utils.config_handler import memory_conf
from utils.redis_client import get_fact_cache_clients


def save(repository, user, text):
    """通过真实仓库事务保存合成原话/事实；每次创建独立会话，不触及正式数据库。"""
    session, owner = uuid4().hex, uuid4().hex
    scope = user + ":" + session
    assert repository.claim(scope, user, session, owner, 120)
    try:
        turn, _ = repository.start_request(scope, user, owner, uuid4().hex, text, extract_memories(text))
        repository.finish_request(scope, owner, turn, "已收到")
    finally:
        repository.release(scope, owner)


def counters(cache, users):
    """聚合本次合成用户的进程内计数；不将缓存加载器次数当作全部SQL次数。"""
    names = ("hits", "misses", "bypasses", "sql_loads", "errors", "reclaimed_keys")
    return {name: sum(cache.stats(user)[name] for user in users) for name in names}


def memory_usage(client, prefix):
    """只扫描本次随机前缀，统计MEMORY USAGE；不包含整个Redis的运行开销。"""
    keys = list(client.scan_iter(match=prefix + ":*"))
    return dict(keys=len(keys), bytes=sum(client.memory_usage(key) or 0 for key in keys))


def run(output):
    """固定种子执行冷/热/混合访问及一次更正，记录查询、TTL与隔离后的实际占用。"""
    cache_client, control = get_fact_cache_clients()
    if cache_client is None or control is None:
        raise SystemExit("需要两个独立且配置正确的测试Redis；请设置REDIS_URL与FACT_CACHE_REDIS_URL")
    prefix = "memory:verification:" + uuid4().hex
    options = dict(memory_conf["fact_cache"])
    options.pop("enabled", None)
    cache = RedisFactCache(cache_client, directory_client=control, key_prefix=prefix, **options)
    users = ["synthetic-" + str(index) for index in range(96)]
    selects = []

    def record_sql(connection, cursor, statement, parameters, context, executemany):
        """只记录SELECT语句，参数与用户正文不进入报告。"""
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)

    try:
        with TemporaryDirectory(prefix="agent-cache-verification-") as temporary:
            repository = MemoryRepository("sqlite:///" + (Path(temporary) / "memory.db").as_posix(), fact_cache=cache)
            repository.initialize()
            try:
                event.listen(repository.engine, "before_cursor_execute", record_sql)
                for user in users[:48]:
                    save(repository, user, "我用的是X10，我家是木地板，养了一只猫，回答请简洁一点。")
                phases = []

                def phase(name, queries):
                    """每阶段独立计算有效命中率与真实SQL计数，同时保留访问延迟口径。"""
                    before = counters(cache, users)
                    selects.clear()
                    durations = []
                    for user, fields in queries:
                        started = time.perf_counter()
                        repository.search_memories(user, keys=fields)
                        durations.append((time.perf_counter() - started) * 1000)
                    after = counters(cache, users)
                    delta = {key: after[key] - before[key] for key in before}
                    denominator = delta["hits"] + delta["misses"]
                    phases.append(dict(name=name, queries=len(queries), **delta,
                        hit_rate=delta["hits"] / denominator if denominator else None,
                        all_selects=len(selects),
                        fact_body_selects=sum("FROM user_memories" in sql for sql in selects),
                        latency_ms=dict(median=statistics.median(durations),
                                        p95=sorted(durations)[int(len(durations) * .95) - 1])))

                fixed = [(user, ["device_model"]) for user in users[:64]]
                phase("cold_64_distinct_users", fixed)
                phase("same_64_queries_repeated", fixed)
                rng = random.Random(20261125)
                shapes = [["device_model"], ["floor_type"], ["pet_status"], ["device_model", "floor_type"]]
                mixed = [(rng.choices(users, weights=[8] * 16 + [1] * 80)[0], rng.choice(shapes)) for _ in range(512)]
                phase("mixed_512_skewed_users_and_four_shapes", mixed)

                old_version = repository.memory_version(users[0])
                old_key = cache.cache_key(repository._cache_partition, users[0], old_version, ["device_model"], 20)
                repository.search_memories(users[0], keys=["device_model"])
                assert cache_client.exists(old_key)
                save(repository, users[0], "实际型号是X20")
                changed = repository.search_memories(users[0], keys=["device_model"])
                assert changed[0]["value"] == "X20" and not cache_client.exists(old_key)
                directories = list(control.scan_iter(match=prefix + ":directory:*"))
                nonempty_key = cache.cache_key(repository._cache_partition, users[0], repository.memory_version(users[0]), ["device_model"], 20)
                empty_key = cache.cache_key(repository._cache_partition, users[63], 0, ["device_model"], 20)
                observed = dict(cache=memory_usage(cache_client, prefix), directory=memory_usage(control, prefix),
                    max_directory_entries=max((control.hlen(key) - 1 for key in directories), default=0),
                    nonempty_ttl_seconds=cache_client.ttl(nonempty_key), empty_ttl_seconds=cache_client.ttl(empty_key),
                    correction_old_key_removed=not bool(cache_client.exists(old_key)), correction_value=changed[0]["value"])
            finally:
                event.remove(repository.engine, "before_cursor_execute", record_sql)
                repository.engine.dispose()
        sources = ["memory/fact_cache.py", "memory/fact_cache_directory.py", "utils/redis_client.py",
                   "db/memory_factory.py", "db/memory_repository.py", "db/memory_schedule.py", "config/memory.yml",
                   "scripts/verify_fact_cache.py"]
        report = dict(schema_version=1, created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seed=20261125,
            scope="synthetic_local_cache_baseline_no_models_no_production_capacity_claim",
            dataset=dict(total_users=96, users_with_facts=48, cold_queries=64, repeated_queries=64, mixed_queries=512),
            redis={name: {key: client.info("memory")[key] for key in ("maxmemory", "maxmemory_policy")}
                   for name, client in (("control", control), ("cache", cache_client))},
            config=options, phases=phases, observed=observed,
            source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
            limitations=["受控合成访问，不代表线上命中率或并发容量", "SQL版本与遗忘策略仍需查询；sql_loads仅统计事实加载器",
                         "MEMORY USAGE仅含本次键，不等于进程RSS或maxmemory", "真实TTL秒数未加速；续期边界另由真实Redis测试用PEXPIRE验证",
                         "未配置MySQL，未调用聊天/Embedding，未比较旧实现的性能"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(dict(output=str(output), phases=phases, observed=observed), ensure_ascii=False, indent=2))
    finally:
        # 只删除本次随机命名空间，异常路径同样不留下合成事实副本。
        for client in (cache_client, control):
            for key in client.scan_iter(match=prefix + ":*"):
                client.delete(key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evals/runs/fact-cache-verification/report.json")
    args = parser.parse_args()
    run(args.output)
