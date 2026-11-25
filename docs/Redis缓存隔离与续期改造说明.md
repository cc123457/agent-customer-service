# Redis缓存隔离与续期：第1～2阶段交付

代码已写回 `D:\RAG\agent`。本次只实施生命周期方案的第1～2阶段：Redis资源隔离、缓存续期、旧版本回收和有限键数量。SQL仍保存原文、摘要、长期事实与Checkpoint；本次没有启用7天归档、365天正文删除、统一会话删除或普通事实到期任务。

## 1. 为什么先改这一块

原来事实缓存、会话快速锁和限流共用REDIS_URL对应实例。直接对它开启allkeys-lfu，缓存挤满时可能把锁和限流记录一起淘汰。因此先隔离实例，再延长缓存驻留时间。

另一问题是固定TTL：旧版非空、空结果都从写入开始计时30～33分钟，命中不续期；更正只换SQL版本，旧缓存主要等待TTL到期。延长TTL时必须同时限制空结果、旧版本与每用户键数，不能只把1800改成7200。

## 2. 文件、改前、改后与原因

| 文件/入口 | 原来 | 现在 | 为什么放在这里 |
|---|---|---|---|
| docker-compose.yml | 一个Redis，未显式设置maxmemory | 控制redis默认128MiB/noeviction；redis-cache默认256MiB/allkeys-lfu，关闭缓存持久化 | 部署层负责物理资源与淘汰隔离；不同DB编号共享内存，不能替代实例隔离 |
| .env.example | 只有REDIS_URL | 新增FACT_CACHE_REDIS_URL、两个内存预算、缓存映射端口和REDIS_CONTROL_REQUIRED | 本机/容器连接及容量属于部署配置，不把端口写死在业务流程中 |
| utils/redis_client.py：get_fact_cache_clients | 缓存复用get_redis | 复用连接池，核验run_id不同、两个实例都有预算、控制noeviction与缓存LFU/LRU | 连接组装时阻止误把同一实例的不同DB当隔离；配置不满足则事实查询走SQL |
| api/deps.py：get_limited_user | 控制启动不可用时可使用本进程内存 | REQUIRED=1时不可用返回503；运行中的限流Redis异常也转503 | 请求入口必须明确控制服务故障，不能把局部内存限流当作全局限流 |
| db/memory_factory.py：_build_repository | 给事实缓存注入共用Redis | 注入缓存正文客户端、控制目录客户端及恢复连接的provider | 仓库装配负责依赖，聊天服务无需到处解析Redis配置 |
| config/memory.yml：fact_cache | 1800秒加0～180秒抖动 | 非空7200+0～600秒、阈值3600秒、空结果300秒、32键上限 | 统一管理缓存策略，避免不同入口采用不同TTL |
| memory/fact_cache.py：get_or_load、_publish | GET/SET固定TTL，旧键等过期 | Lua读正文及PTTL、按阈值续期；先登记再发布、发布后复核；SQL旁路与计数 | 这些都是同一次缓存访问的职责，替换原实现，没有另建并行V2类 |
| memory/fact_cache_directory.py | 没有受保护的用户键目录 | 每数据库/用户一个HASH，记录版本与最多32个正文键；15秒发布租约和CAS回收确认 | 定位信息不能和缓存正文一起被LFU淘汰；单独文件承载目录Lua，便于阅读 |
| db/memory_repository.py：_advance_memory_version、start_request、delete_memory | 同事务推进版本，提交后只记失效计数 | 返回已提交新版本，并通知缓存目录定向回收旧键 | SQL事务是事实改变的权威边界，不能在事务提交前清理并宣称修改已生效 |
| db/memory_schedule.py：activate_due | 未来事实激活后推进版本 | 提交后同样携新版本通知回收 | 到期激活也是事实改变，不能漏掉这一条写路径 |
| tests/test_fact_cache.py、test_cache_isolation.py | 旧GET/SET协议替身和真实Redis重复运行部分断言 | 正常Lua路径使用真实两个实例；只有故障/连接配置分支注入异常或Mock | Python字典不能验证Redis Lua原子行为；保留真实SQL校验和事务失败测试 |
| .github/workflows/ci.yml、docker-compose.redis-test.yml | CI一个Redis；本机测试可能复用6379 | CI两实例；本机专用6391/6392，明确预算，强制依赖可用 | 不让关键集成用例因缺服务悄悄跳过，也不对现有6379施压 |

新增和改动的核心函数都有职责注释。目录只存键名、版本、短租约令牌和时间，不复制事实正文。

## 3. 一次事实读取现在怎么走

1. `db/memory_repository.py`读取SQL用户记忆版本；如果未来计划到了生效点，先事务激活。遗忘策略和其他SQL检查仍保留。
2. `memory/fact_cache.py`用数据库分区、认证用户、字段组合、数量上限和SQL版本构造键。当前默认前缀是`memory:facts:v2`。
3. `FactCacheDirectory.observe()`在控制实例登记所见版本，返回可清理的旧键；已落后于目录版本的读者直接回SQL，不重新发布旧版本。
4. 缓存实例Lua同时读正文和剩余TTL。应用校验JSON、身份、版本、active状态、字段、来源与明确业务失效时间。
5. 有效命中返回事实；非空且剩余不超过3600秒时才尝试续期，空结果不续期。
6. 未命中或损坏时由原SQL加载器查询事实，回填前先在目录预约名额，再限时发布正文，最后复核版本。目录/缓存故障走SQL，SQL错误继续上抛。

缓存只省去事实正文查询。本轮实测热访问每次仍有3条SELECT，包括版本与访问规则；不能写成“命中后不查数据库”。原来的“命中保留一条版本SELECT”指版本检查本身，不等于全链路只有一条SQL。

### 续期的具体例子

非空条目写入后TTL是7200～7800秒。10分钟后读取，剩余时间仍大于3600秒，直接返回，不写TTL。约一个多小时后再读，若剩余已不超过3600秒，则续回7200～7800秒。

这是近似空闲窗口，不是每次命中都精确延后两小时；阈值续期减少频繁写入。LFU仍可能在TTL前淘汰缓存，之后按需从SQL恢复。空结果固定300秒；用户新增事实时SQL换版本，无需等待这5分钟。

`expires_at`目前仅接入缓存边界：如果加载器明确提供该时间，读取排除已到期事实，续期也不得超过其失效点。普通事实的SQL有效期表、自然语言期限提取和跨历史过滤尚在后续阶段，不能称为已经完成完整记忆到期机制。

## 4. 用户更正后，为什么旧回填不能复活

假设SQL版本7的型号是X10，用户改成X20。

旧版：SQL事务更新为版本8；新读者访问r8键，r7继续等待TTL。旧请求晚到时仍可能生成一个r7缓存键，但不能覆盖r8。

现在：SQL仍先提交版本8；随后`note_invalidation(partition, user, version=8)`推进控制目录版本，清理已完成发布的r7正文，删除成功后才CAS移除目录记录。

如果旧请求已登记、正在向另一个Redis发布r7，目录暂不删除其活跃发布名额。它发布后必须`confirm()`：发现目录已经是8，就撤回自己的r7键。写入Lua还检查15秒发布期限，已超过期限的发布者不能再写正文。

跨SQL和两个Redis没有统一原子事务。旧进程若在发布后崩溃，或缓存连接中断，物理回收可能延迟到后续访问/TTL；SQL版本检查仍使旧键不能被新版本请求当成有效事实。更新前已经取得旧快照的请求可能完成旧快照读取，本次没有承诺撤回已经发出的内容。

目录放在noeviction控制实例，避免缓存LFU连定位信息一起淘汰。目录与正文均有TTL；每用户最多32条定位记录，另有一个_version字段。名额满时回收最早登记/续期的非发布项，这不是逐次命中更新的严格LRU。它也不是后续“用户删除任务已经全部物理完成”的证明：持久清理任务属于第5阶段。

## 5. 真实验证结果

测试服务是用户启动的独立项目`agent-cache-test`，Redis版本7.4.12：控制6391为16MiB/noeviction，缓存6392为8MiB/allkeys-lfu。run_id核验两者确实不同。现有6379未修改配置、未清空、未压力测试。

新增专项测试17项全部通过，覆盖：

- 非空阈值续期、空结果不续期、明确业务失效时间限制。
- 每用户形状上限，删除旧形状后重新从SQL加载。
- SET后/confirm前并发更正，旧键撤回、新键保留；超过发布期限不允许写入。
- 目录故障时不信任已缓存正文，启动时不可用后恢复；SQL错误不伪装成“无记忆”。
- 同实例不同DB/不正确内存策略拒用；连接池复用；REQUIRED控制失败返回503。
- 向8MiB专用缓存写入16KiB条目直至evicted_keys增长，检查另一实例的锁owner、锁TTL和限流拒绝状态仍正确。

TTL续期边界使用真实Redis的PEXPIRE缩短剩余时间后验证，不是假装已经等待两小时。压力用例只对显式TEST_FACT_CACHE_DEDICATED=1的专用实例运行，不修改其全局配置。

全量回归：214 passed、96 skipped、1个既有langchain-community弃用警告。96项均因本机未配置测试MySQL而跳过；不表示项目更换了数据库，也不代表MySQL已经验收。原先正常路径替身的重复参数化已移除，增加真实Lua/目录用例，因此总数不与旧209/107按净新增直接比较。CI已配置MySQL与两Redis，但本次没有执行远端GitHub Actions。

### 小规模访问基线

`scripts/verify_fact_cache.py`使用临时SQLite、96个合成用户（48个有事实）、随机种子20261125和真实两Redis。不访问正式记忆库，不调用模型/Embedding。每阶段重新统计，混合阶段承接前面已建立的缓存。

| 阶段 | 查询 | 有效命中 | 事实SQL加载 | 命中率 | 全部SELECT |
|---|---:|---:|---:|---:|---:|
| 首次查询64个不同用户 | 64 | 0 | 64 | 0% | 256 |
| 原样重复上述查询 | 64 | 64 | 0 | 100% | 192 |
| 512次偏斜用户访问、4种字段组合 | 512 | 337 | 175 | 65.82% | 1711 |

这是受控合成分布，不能称为线上命中率或高并发容量。没有测旧实现延迟，因此不归因宣称此次改造提升了某个性能百分比。

一次更正后的采样：缓存正文236键、MEMORY USAGE合计115488字节；控制目录95个HASH、合计115216字节，每HASH最多4条正文定位记录。目录开销在这个小条目场景接近正文开销，部署预算必须把它算进去；上述数值都不含进程RSS、全局哈希表及连接缓冲。实测非空TTL7439秒，空结果298秒；更正后旧键已删除，新值X20。抖动和运行时间会使复跑TTL/内存有小幅变化。

原始聚合与源文件指纹：`evals/baselines/redis-cache-isolation-20261125.json`。程序回归JUnit原始记录：`evals/runs/redis-cache-isolation-20261125/pytest.xml`，运行产物目录不提交Git。

## 6. 怎么复跑

在`D:\RAG\agent`的PowerShell执行；测试使用新的临时SQLite和本例随机Redis前缀，结束只删除这些前缀。

```powershell
docker compose -p agent-cache-test -f docker-compose.redis-test.yml up -d
$env:TEST_REDIS_URL='redis://127.0.0.1:6391/15'
$env:TEST_FACT_CACHE_REDIS_URL='redis://127.0.0.1:6392/15'
$env:REDIS_URL=$env:TEST_REDIS_URL
$env:FACT_CACHE_REDIS_URL=$env:TEST_FACT_CACHE_REDIS_URL
$env:REQUIRE_REDIS='1'
$env:REQUIRE_FACT_CACHE_REDIS='1'
$env:TEST_FACT_CACHE_DEDICATED='1'
$env:REDIS_CONTROL_REQUIRED='1'
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe scripts/verify_fact_cache.py --output evals/runs/fact-cache-verification/report.json
```

## 7. 正式启用与当前边界

本次修改代码与部署文件，未改含密钥的实际.env，也未重启现有业务后端/6379。6391/6392只用于验收。旧.env如果没有FACT_CACHE_REDIS_URL，新程序会旁路事实缓存；不能宣称当前运行的业务服务已经启用双Redis。

启用时先检查现有Redis由谁管理，再按docker-compose.yml准备控制实例（默认6379）和缓存实例（默认6380），设置各自预算与策略；本机后端填写.env中的FACT_CACHE_REDIS_URL，生产/多worker设置REDIS_CONTROL_REQUIRED=1并重启后端。容器后端已配置服务名地址及REQUIRED=1。控制noeviction内存耗尽会拒绝写入，需监控与扩容；128/256MiB只是可配置起点，不是承诺容纳多少用户。

新默认缓存前缀v2，不读取或续期旧v1键；旧键等原TTL自然清理，不对旧实例做全库扫描删除。控制目录丢失、控制实例恢复旧快照、SQL版本表重置或数据库还原，应协调切换缓存命名空间/重建缓存，不能把仍存在的旧正文视为可完整定位。后续统一删除需要持久清理任务，尚未在本批交付。

缓存连接池每条命令/连接2秒超时，关闭SDK额外自动重试；多个命令串行执行不等于整次请求总耗时保证2秒。事实缓存失败短暂冷却后可恢复；控制服务启动时未连接的进程在REQUIRED=1下返回503，需要服务恢复后重启重新装配。

同进程64把条带锁继续合并相同键冷读，未实现跨进程分布式回填合并。统计仍只属于当前进程、最多1024用户，不是集群监控。本批没有进行万用户并发、SQL连接池或磁盘容量测试；后续按既定方案进入会话生命周期与热正文计数。
