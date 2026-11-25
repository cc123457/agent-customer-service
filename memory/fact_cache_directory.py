"""控制Redis中的有界缓存键目录；不保存事实正文，不依赖缓存实例中的索引存活。"""
import hashlib
import json
import secrets

from redis.exceptions import RedisError


_DIRECTORY_LUA = r"""
local request = cjson.decode(ARGV[1])
local stamp = redis.call('TIME')
local now = tonumber(stamp[1]) * 1000 + math.floor(tonumber(stamp[2]) / 1000)
local function result(value) return cjson.encode(value) end
local function retain() redis.call('EXPIRE', KEYS[1], request.directory_ttl) end
if request.op == 'ack' then
    local removed = 0
    for _, item in ipairs(request.items) do
        local raw = redis.call('HGET', KEYS[1], item.key)
        if raw and raw == item.record then
            local entry = cjson.decode(raw)
            if entry.writing_until <= now or (item.ended and entry.token == item.token) then
                removed = removed + redis.call('HDEL', KEYS[1], item.key)
            end
        end
    end
    return result({removed=removed})
end
if request.op == 'confirm' then
    local raw = redis.call('HGET', KEYS[1], request.key)
    if not raw then return result({ok=false}) end
    local entry = cjson.decode(raw)
    local version = tonumber(redis.call('HGET', KEYS[1], '_version') or '-1')
    if entry.token ~= request.token or version ~= request.version or entry.writing_until < now then
        return result({ok=false, record=raw})
    end
    entry.writing_until = 0
    redis.call('HSET', KEYS[1], request.key, cjson.encode(entry))
    retain()
    return result({ok=true})
end
local version = tonumber(redis.call('HGET', KEYS[1], '_version') or '-1')
if request.version < version then return result({stale=true, victims={}}) end
redis.call('HSET', KEYS[1], '_version', request.version)
local members = redis.call('HGETALL', KEYS[1])
local victims, oldest, count = {}, nil, 0
for index=1,#members,2 do
    local key, raw = members[index], members[index+1]
    if key ~= '_version' then
        local entry = cjson.decode(raw)
        if entry.expires_at <= now and entry.writing_until <= now then
            redis.call('HDEL', KEYS[1], key)
        else
            count = count + 1
            if entry.writing_until <= now then
                local item = {key=key, record=raw}
                if entry.version < request.version then table.insert(victims, item) end
                if key ~= request.key and (not oldest or entry.touched < oldest.touched) then
                    oldest = {key=key, record=raw, touched=entry.touched}
                end
            end
        end
    end
end
retain()
if request.op == 'observe' then return result({stale=false, victims=victims}) end
if #victims > 0 then return result({retry=true, victims=victims}) end
local raw = redis.call('HGET', KEYS[1], request.key)
if raw and cjson.decode(raw).writing_until > now then return result({busy=true, victims={}}) end
if not raw and count >= request.max_keys then
    if oldest then return result({retry=true, victims={oldest}}) end
    return result({busy=true, victims={}})
end
local lease_end = now + request.lease_ms
local entry = {version=request.version, token=request.token, writing_until=lease_end,
               touched=now, expires_at=now + request.ttl_ms + request.lease_ms + 1000}
local encoded = cjson.encode(entry)
redis.call('HSET', KEYS[1], request.key, encoded)
retain()
return result({ok=true, token=request.token, record=encoded, lease_end=lease_end})
"""


class FactCacheDirectory:
    """每数据库/用户最多max_keys个目标；在控制实例上串行登记版本和短写入租约。"""

    def __init__(self, client, prefix, max_keys=32, ttl_seconds=8000, publication_lease_seconds=15):
        """构造不访问网络；目录保留时间须长于缓存TTL及正在提交的写入。"""
        self.client, self.prefix = client, prefix
        self.max_keys, self.ttl_seconds = max_keys, ttl_seconds
        self.lease_ms = publication_lease_seconds * 1000

    def key(self, partition, user_id):
        """固定数据库和认证用户范围，不使用客户端提交的任意Redis键。"""
        digest = hashlib.sha256(json.dumps([partition, user_id], ensure_ascii=False).encode()).hexdigest()
        return self.prefix + ":directory:" + digest

    def _call(self, partition, user_id, **request):
        """一次Lua原子更新目录；只回传定位元数据，协议异常按Redis故障处理。"""
        request.update(directory_ttl=self.ttl_seconds)
        raw = self.client.eval(_DIRECTORY_LUA, 1, self.key(partition, user_id), json.dumps(request))
        try:
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("目录响应不是对象")
            # Lua的空table可能编码成{}；在Python侧统一为列表。
            if "victims" in value:
                value["victims"] = value["victims"] or []
            return value
        except (TypeError, ValueError) as error:
            raise RedisError("缓存目录响应无效") from error

    def observe(self, partition, user_id, version):
        """记录SQL最新版本并返回可回收旧键；过期写入不能重新登记更低版本。"""
        return self._call(partition, user_id, op="observe", version=version)

    def reserve(self, partition, user_id, version, key, ttl_ms):
        """先登记目标再允许写缓存；满额返回可淘汰项，不突破每用户键数上限。"""
        return self._call(partition, user_id, op="reserve", version=version, key=key, ttl_ms=ttl_ms,
                          max_keys=self.max_keys, lease_ms=self.lease_ms, token=secrets.token_hex(16))

    def confirm(self, partition, user_id, version, key, token):
        """缓存提交后再次检查目录版本；并发更正已发生时拒绝确认旧写入。"""
        return self._call(partition, user_id, op="confirm", version=version, key=key, token=token)

    def acknowledge(self, partition, user_id, victims):
        """外部删除成功后按原目录值CAS移除；新发布者改变过的目录不会被旧任务删掉。"""
        if victims:
            return self._call(partition, user_id, op="ack", items=victims)["removed"]
        return 0
