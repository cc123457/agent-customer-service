"""有界事实缓存：SQL版本授权、滑动TTL、控制Redis目录与缓存Redis内容分离。"""
import hashlib
import json
import logging
import math
import secrets
import time
from collections import OrderedDict
from threading import Lock

from redis.exceptions import RedisError
from memory.fact_cache_directory import FactCacheDirectory

logger = logging.getLogger(__name__)
FACT_FIELDS = frozenset({"device_model", "floor_type", "pet_status", "reply_style"})
_READ_LUA = """
local value = redis.call('GET', KEYS[1])
if not value then return {} end
return {value, redis.call('PTTL', KEYS[1])}
"""
_PUBLISH_LUA = """
local stamp = redis.call('TIME')
local now = tonumber(stamp[1]) * 1000 + math.floor(tonumber(stamp[2]) / 1000)
if now >= tonumber(ARGV[4]) then return 0 end
local ttl = tonumber(ARGV[3])
local expires = tonumber(ARGV[5])
if expires > 0 then ttl = math.min(ttl, expires - now) end
if ttl < 1 then return 0 end
if ARGV[1] == 'touch' then
    if redis.call('GET', KEYS[1]) ~= ARGV[2] then return 0 end
    local remaining = redis.call('PTTL', KEYS[1])
    if remaining > tonumber(ARGV[6]) then return 2 end
    return redis.call('PEXPIRE', KEYS[1], ttl)
end
redis.call('SET', KEYS[1], ARGV[2], 'PX', ttl)
return 1
"""


class RedisFactCache:
    """只缓存本人结构化事实；生产装配验证两个Redis实例，显式测试注入可共享客户端。"""

    def __init__(self, redis_client=None, ttl_seconds=7200, ttl_jitter_seconds=600,
                 empty_ttl_seconds=300, refresh_threshold_seconds=3600, max_keys_per_user=32,
                 max_payload_bytes=65536, retry_cooldown=5, clock=time.monotonic,
                 key_prefix="memory:facts:v2", directory_client=None, client_provider=None,
                 wall_clock=time.time, publication_lease_seconds=15):
        """注入独立缓存/目录客户端；构造不推理，SQL加载器由每次调用提供。"""
        if (ttl_seconds <= 0 or ttl_jitter_seconds < 0 or empty_ttl_seconds <= 0
                or refresh_threshold_seconds < 0 or max_keys_per_user < 1 or max_payload_bytes <= 0
                or retry_cooldown < 0 or publication_lease_seconds <= 0):
            raise ValueError("长期事实缓存配置无效")
        self._redis = redis_client
        self._directory_client = directory_client if directory_client is not None else redis_client
        self._client_provider = client_provider
        self._key_prefix = key_prefix
        self.ttl_seconds, self.ttl_jitter_seconds = ttl_seconds, ttl_jitter_seconds
        self.empty_ttl_seconds = empty_ttl_seconds
        self.refresh_threshold_seconds = min(refresh_threshold_seconds, ttl_seconds)
        self.max_keys_per_user = max_keys_per_user
        self.max_payload_bytes, self.retry_cooldown = max_payload_bytes, retry_cooldown
        self.publication_lease_seconds = publication_lease_seconds
        self._clock, self._wall_clock = clock, wall_clock
        self._retry_after = 0
        self._state_lock = Lock()
        self._stats = OrderedDict()
        self._fill_locks = tuple(Lock() for _ in range(64))
        self._make_directory()

    def _make_directory(self):
        """目录TTL大于所有正文TTL与发布租约，防止正文存在而定位目录提前过期。"""
        retention = max(self.ttl_seconds + self.ttl_jitter_seconds, self.empty_ttl_seconds)
        self._directory = FactCacheDirectory(self._directory_client, self._key_prefix, self.max_keys_per_user,
                                             retention + self.publication_lease_seconds + 60,
                                             self.publication_lease_seconds)

    def cache_key(self, partition, user_id, version, keys, limit):
        """数据库、认证用户、字段组合、数量及SQL版本共同隔离，哈希中不包含正文。"""
        identity = json.dumps([partition, user_id, sorted(set(keys)), limit], ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(identity.encode()).hexdigest()
        return f"{self._key_prefix}:data:{digest}:r{version}"

    def _record(self, user_id, **changes):
        """累计本进程每用户计数；只保留最近1024个用户统计，不保存正文。"""
        with self._state_lock:
            counters = self._stats.pop(user_id, {})
            for name, count in changes.items():
                counters[name] = counters.get(name, 0) + count
            self._stats[user_id] = counters
            if len(self._stats) > 1024:
                self._stats.popitem(last=False)

    def _available(self):
        """只读当前连接/冷却状态，统计接口不会为了显示状态发起网络连接。"""
        with self._state_lock:
            return self._redis is not None and self._directory_client is not None and self._clock() >= self._retry_after

    def _ready(self):
        """启动时不可用的独立缓存可在冷却后重新验证连接，禁止退回混用控制实例。"""
        if self._available():
            return True
        with self._state_lock:
            if self._clock() < self._retry_after or self._client_provider is None:
                return False
        cache, directory = self._client_provider()
        with self._state_lock:
            self._redis, self._directory_client = cache, directory
            if cache is None or directory is None:
                self._retry_after = self._clock() + self.retry_cooldown
                return False
            self._make_directory()
            return True

    def _failed(self, user_id):
        """缓存或目录故障进入短暂冷却；SQL错误不在此处被吞掉。"""
        self._record(user_id, errors=1)
        with self._state_lock:
            first = self._clock() >= self._retry_after
            self._retry_after = self._clock() + self.retry_cooldown
        if first:
            logger.warning("事实缓存或键目录暂不可用，回退SQL并进入短暂冷却")

    def _valid_rows(self, rows):
        """预留明确expires_at的读取边界；未来有效期接入后缓存也不会续命已到期事实。"""
        result = []
        for row in rows:
            expires = row.get("expires_at")
            if expires is not None:
                if type(expires) not in (int, float) or not math.isfinite(expires):
                    raise ValueError("缓存事实到期时间无效")
                if expires <= self._wall_clock():
                    continue
            result.append(row)
        return result

    def _decode(self, payload, user_id, version, keys, limit):
        """核验结构、身份、版本和来源；问答正文/未知字段不属于此缓存。"""
        if len(payload) > self.max_payload_bytes:
            raise ValueError("缓存正文超限")
        entry = json.loads(payload)
        if not isinstance(entry, dict) or entry.get("schema") != 2:
            raise ValueError("缓存格式不兼容")
        if entry.get("user_id") != user_id or entry.get("version") != version:
            raise ValueError("缓存身份或版本不匹配")
        rows = entry.get("rows")
        if not isinstance(rows, list) or len(rows) > limit:
            raise ValueError("缓存条目数量无效")
        for row in rows:
            if not isinstance(row, dict) or row.get("user_id") != user_id or row.get("status") != "active":
                raise ValueError("缓存事实归属或状态无效")
            if row.get("attribute", row.get("memory_key")) not in keys:
                raise ValueError("缓存事实不属于本次字段查询")
            if not all(isinstance(row.get(name), str) for name in ("id", "memory_key", "value")):
                raise ValueError("缓存事实字段类型无效")
            if type(row.get("source_message_id")) is not int or row["source_message_id"] < 1:
                raise ValueError("缓存事实来源无效")
        if len(self._valid_rows(rows)) != len(rows):
            raise ValueError("缓存含已到期事实")
        return rows

    def _ttl(self, rows):
        """空结果短缓存；非空结果加抖动，并返回最早业务失效时间供Redis原子限定。"""
        seconds = self.empty_ttl_seconds if not rows else self.ttl_seconds + secrets.randbelow(self.ttl_jitter_seconds + 1)
        expiry = min((row["expires_at"] for row in rows if row.get("expires_at") is not None), default=0)
        return seconds * 1000, int(expiry * 1000)

    def _clear(self, partition, user_id, victims):
        """先清正文，再按目录旧值CAS确认；Redis失败时留下目录供以后重试。"""
        if victims:
            self._redis.delete(*(item["key"] for item in victims))
            removed = self._directory.acknowledge(partition, user_id, victims)
            self._record(user_id, reclaimed_keys=removed)

    def _publish(self, partition, user_id, version, key, payload, rows, mode="set"):
        """先占目录名额，再限时提交正文，最后复核版本；旧写入不能复活新版本缓存。"""
        ttl_ms, expiry_ms = self._ttl(rows)
        reservation = None
        for _ in range(2):
            reservation = self._directory.reserve(partition, user_id, version, key, ttl_ms)
            if reservation.get("retry"):
                self._clear(partition, user_id, reservation["victims"])
                continue
            break
        if not reservation or not reservation.get("ok"):
            self._record(user_id, publication_bypasses=1)
            return
        changed = self._redis.eval(_PUBLISH_LUA, 1, key, mode, payload, ttl_ms, reservation["lease_end"],
                                   expiry_ms, self.refresh_threshold_seconds * 1000)
        confirmed = self._directory.confirm(partition, user_id, version, key, reservation["token"])
        if not confirmed.get("ok"):
            self._clear(partition, user_id, [dict(key=key, record=confirmed.get("record", reservation["record"]),
                                                 ended=True, token=reservation["token"])])
            self._record(user_id, publication_bypasses=1)
        elif mode == "touch" and changed == 1:
            self._record(user_id, refreshes=1)

    def get_or_load(self, partition, user_id, version, keys, limit, loader):
        """SQL授权版本下读缓存；过期/损坏/故障回源，命中只在阈值内滑动续期。"""
        def load():
            """SQL错误直接传播；仅过滤明确到期行，不推测事实或生成空答案。"""
            return self._valid_rows(loader())
        if not self._ready():
            self._record(user_id, bypasses=1, sql_loads=1)
            return load()
        key = self.cache_key(partition, user_id, version, keys, limit)
        slot = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % len(self._fill_locks)
        with self._fill_locks[slot]:
            if not self._ready():
                self._record(user_id, bypasses=1, sql_loads=1)
                return load()
            try:
                observed = self._directory.observe(partition, user_id, version)
                if observed.get("stale"):
                    self._record(user_id, stale_bypasses=1, bypasses=1, sql_loads=1)
                    return load()
                self._clear(partition, user_id, observed["victims"])
                cached = self._redis.eval(_READ_LUA, 1, key)
            except RedisError:
                self._failed(user_id)
                self._record(user_id, bypasses=1, sql_loads=1)
                return load()
            if cached:
                payload, remaining_ms = cached
                try:
                    rows = self._decode(payload, user_id, version, keys, limit)
                except (ValueError, TypeError, UnicodeError, RecursionError):
                    self._record(user_id, invalid_entries=1)
                else:
                    self._record(user_id, hits=1)
                    # 空结果不滑动续期；并发续期仍由Redis比较原值与TTL，避免每轮写TTL。
                    if rows and remaining_ms <= self.refresh_threshold_seconds * 1000:
                        try:
                            self._publish(partition, user_id, version, key, payload, rows, "touch")
                        except RedisError:
                            self._failed(user_id)
                    return rows
            self._record(user_id, misses=1, sql_loads=1)
            rows = load()
            payload = json.dumps(dict(schema=2, user_id=user_id, version=version, rows=rows),
                                 ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
            if len(payload) > self.max_payload_bytes:
                self._record(user_id, oversized=1)
                return rows
            try:
                self._publish(partition, user_id, version, key, payload, rows)
            except RedisError:
                self._failed(user_id)
            return rows

    def note_invalidation(self, user_id, *, partition=None, version=None):
        """SQL提交后逻辑失效并尽力回收旧版本；失败不回滚事实，后续请求继续清理目录。"""
        self._record(user_id, invalidations=1)
        if partition is None or version is None or not self._ready():
            return
        try:
            observed = self._directory.observe(partition, user_id, version)
            self._clear(partition, user_id, observed.get("victims", []))
        except RedisError:
            self._failed(user_id)

    def stats(self, user_id):
        """本人本进程统计；SQL版本读取和缓存正文命中是不同指标，不报告线上容量结论。"""
        with self._state_lock:
            counters = dict(self._stats.get(user_id, {}))
        names = ("hits", "misses", "bypasses", "sql_loads", "errors", "invalid_entries", "oversized", "invalidations",
                 "refreshes", "reclaimed_keys", "publication_bypasses", "stale_bypasses")
        result = {name: counters.get(name, 0) for name in names}
        count = result["hits"] + result["misses"]
        return dict(enabled=True, available=self._available(), scope="current_process", **result,
                    hit_rate=result["hits"] / count if count else None)
