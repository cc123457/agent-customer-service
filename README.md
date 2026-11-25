# 智能客服 Agent（RAG + ReAct，前后端分离）

基于 **LangChain 1.0 + LangGraph** 的企业级智能客服系统，当前以扫地/扫拖机器人场景为例。
后端为 **FastAPI 微服务**（SSE 异步流式），前端为 **Streamlit**（纯客户端），两者通过 HTTP 完全解耦。

> 知识库与业务数据均为示例，可替换为任意行业的知识文档与数据库。

## 技术栈

| 层 | 技术 |
|---|---|
| 智能体框架 | LangChain 1.0 `create_agent`（ReAct）+ LangGraph 运行时 |
| RAG | Chroma 向量库 + DashScope `text-embedding-v3`（可配置、向量空间隔离）+ 递归分块 |
| 大模型 | Qwen（OpenAI 兼容协议接入） |
| 后端 | FastAPI + SSE 流式输出（异步链路） |
| 前端 | Streamlit（纯客户端，通过 HTTP 调后端 API） |
| 数据 | SQLite（用户 / 记录）+ SQLite 或 MySQL（原文 / 长期记忆）+ Chroma（知识向量） |
| 认证 | JWT（登录签发 / 接口鉴权 / 用户身份注入 Agent） |
| 限流 | 控制Redis ZSET滑动窗口（Lua原子）；开发模式可使用单进程内存兜底 |
| 可观测性 | Langfuse（链路追踪 / token 成本 / 按用户筛选） |
| 外部工具 | MCP 协议接入（天气服务） |
| 会话记忆 | SQL 原文档案 + 分批摘要水位 + 有界上下文 + 跨会话事实检索；Redis 事实缓存与 Redis / 内存快速租约 |
| 配置 / 日志 | YAML 配置 + 分级日志 |

## 核心特性

- **RAG 知识问答**：PDF / TXT → 分块 → 向量化 → 召回 → 总结
- **多工具 ReAct**：内置工具 + MCP 外部工具（RAG 检索、天气、定位、用户 ID、月份、外部数据、报告上下文注入）
- **中间件体系**：工具调用监控（异步）、模型日志、动态提示词场景切换（问答 vs 报告生成）
- **工具调用轨迹可见化**：结构化流式输出（`action` / `answer` 分离），前端展示 Agent 每一步调用了什么工具
- **MCP 外部工具接入**：天气服务通过 MCP 协议接入，客户端零改动即可切换服务商
- **JWT 认证**：登录签发 token、接口鉴权、用户身份经 runtime context 注入工具（`get_user_id` 返回真实登录用户）
- **接口限流**：按用户每分钟限流，Redis ZSET 滑动窗口 + Lua 脚本原子化（消除并发竞态），超限返回 429
- **链路可观测**：Langfuse 自动追踪模型调用、工具调用、RAG 内部链路的耗时与 token 成本
- **原文档案与工作上下文分离**：SQL 保存原文和请求状态，模型只接收预算内摘要、相关记忆和近期问答；摘要成功也不删除原文。
- **分批摘要**：动态扣除指令和阶段摘要预算，长消息可拆分；全部成功才事务提交摘要/水位，失败保留旧版本并冷却。
- **跨会话长期记忆**：统一读写分析，规则优先、模型至多一次补充；支持对象/冲突、更正/遗忘屏障与来源追溯；SQL字段/关键词与独立Chroma语义召回结合。
- **结构化事实缓存**：独立Redis实例缓存型号、地面、养宠和回复风格查询；SQL事务版本使旧缓存失效，并按控制实例中的目录定向回收。支持2小时空闲续期、5分钟空结果、每用户最多32个键和本人当前进程统计。
- **历史问答混合检索**：先按用户、会话和时间过滤，再用BM25Plus＋独立Chroma语义召回、RRF融合，SQL复核后补齐双方原文；支持冷原文恢复和预算内问答节选。
- **会话并发保护**：控制Redis/开发模式内存快速租约配合SQL权威租约；同会话互斥，可选request_id持久去重。Docker设置REDIS_CONTROL_REQUIRED=1，控制服务不可用返回503。
- **前后端分离**：FastAPI 后端 + Streamlit 前端，通过 REST / SSE 通信
- **真实数据**：用户与使用记录存于 SQLite，替换原 mock 数据
- **知识库增量入库**：文件 MD5 去重，避免重复向量化
- **混合检索**：Chroma 语义召回 + BM25 关键词召回（jieba 中文分词），通过 RRF 融合排名；空库安全返回，当前实例成功入库后自动清除 BM25 缓存，下次查询重建。

### 混合检索验证与边界

执行 `.venv\Scripts\python.exe -m pytest tests/test_hybrid_retriever.py -q` 可验证空库、空白文档、查询参数、融合去重、中文关键词召回、自定义返回数量，以及入库后的索引刷新。

其中效果对比采用受控场景：模拟向量召回漏掉“滤网”资料，检查混合召回是否补回该资料。它验证关键词分支能补漏，不代表真实知识库上的整体质量提升；正式评估还需固定问题集与相关文档标注，比较 Recall@k、MRR 和延迟。

BM25 在当前进程内懒加载并缓存。缓存刷新覆盖当前实例的 `load_document()` 入库；其他进程或脚本直接修改 Chroma 后，已有服务实例需要重启才能重建 BM25。当前实现每路召回相同数量的候选，再融合返回指定数量。

### 当前记忆主链路

2026-11-25升级增加统一记忆读写分析、按需检索和持久遗忘屏障，删除同时约束近期消息、摘要、历史回查、缓存与旧检查点。逐文件前后对照见[记忆路由与遗忘改造说明](docs/记忆路由与遗忘改造说明.md)。Embedding配置已生效，切换模型使用独立集合重建，旧向量不混查。 最终回归209项通过、107项MySQL未配置跳过；40题定向冻结首测旧记忆16/40、新记忆36/40，最终已知题回归仍36/40，另48例抽取F1为98.46%。评测器修正、剩余4题问题和费用口径见[记忆路由与遗忘验收报告](docs/记忆路由与遗忘验收报告.md)，均不是线上准确率。

阅读 [记忆模块说明](docs/记忆模块说明.md) 了解当前代码；[代码清理说明](docs/代码清理说明.md) 对照合并和删除内容。聊天业务只保留 services/chat_service.py 的 ChatService，原文/摘要/事实/检查点统一由关系库保存，Redis额外缓存结构化事实查询，Redis/内存租约组件负责快速互斥。

缓存隔离与续期已实施，逐文件改前/改后、部署切换及真实Redis验证见[Redis缓存隔离与续期改造说明](docs/Redis缓存隔离与续期改造说明.md)。这是生命周期方案的第1～2阶段；7天归档、统一会话删除及普通事实过期尚待后续实施。

默认记忆数据库为data/memory.db，可用MEMORY_DATABASE_URL选择MySQL。消息归档、分批摘要、统一读写分析、对象/冲突处理、独立Chroma语义检索和持久图恢复均由这一套服务组织，不再保留旧版Redis消息/摘要流程。

上下文由摘要、近期完整问答和相关记忆按预算组成；摘要失败保留原文，暂未选中的原文不会全部进入本轮输入。预算是UTF-8字节加固定开销的估算，不等于精确Token数，也未完整限制图内工具结果。批次上限导致摘要任务反复重试的边界尚未在本次结构清理中改变。

接口均按JWT认证身份隔离：

- GET /conversations/{session_id}/messages：原文分页，冷档案按索引恢复。
- GET /memories、DELETE /memories/{memory_id}：查看和软删除本人事实。
- GET /memories/cache-stats：查看本人当前后端进程的事实缓存命中、未命中、旁路和错误计数；重启后重置，多worker不聚合。
- GET /history/search：搜索本人历史问答，参数query、可选session_id、after、before；支持JWT、按用户限流和时间范围，返回双方原文与来源。
- GET /conversations/{session_id}/requests/{request_id}/checkpoints：查看本人请求执行概况。
- POST /conversations/{session_id}/requests/{request_id}/resume：继续最后一条失败请求，不重复归档用户输入。

软删除事实不等于抹除原始档案；检查点恢复也不会自动撤销业务数据库写入或代码变化。维护脚本默认只预览，--apply才执行归档/保留策略，本地冷文件需独立备份。

程序回归使用python -m pytest tests/ -q；CI强制真实MySQL/Redis可用。记忆开发集30例、Agent/接口开发集60例均已迁移至当前模块；另有60例合成历史标注（12例校准、48例验收），支持真实模型的四种上下文对照。运行和统计范围见 [评测说明](evals/README.md)。模型替身用于可控错误路径，合成集通过不代表生产质量。

质量评测另覆盖候选抽取Precision/Recall/F1、历史检索Precision及拒答、摘要关键事实保留，并提供方案盲评网页和人工结果导入。具体结果与未覆盖范围见[记忆质量补充评测](docs/记忆质量补充评测.md)；人工评审结果需真人填写，不能以规则分数替代。

记忆更正现使用有限预算属性键，明确日期的未来事实到期后才激活，并同步推进缓存版本；摘要补保编号、日期和时刻原文。实现前后对照见[记忆修复实现说明](docs/记忆修复实现说明.md)，真实模型补测、Embedding用量与完整冷启动参考费用见[记忆优化实测报告](docs/记忆优化实测报告.md)。评测可通过`--chat-model`单独选择有额度的模型，不改项目默认模型；不同模型结果分别记录。

后续另编80题，冻结实现后使用同一qwen3-omni-flash完成四方案首轮对照和320条AI语义复核：当前记忆严格任务通过率82.50%，仅近期33.75%，摘要加近期46.25%，完整历史83.75%；当前正文内容正确率另为88.75%。当时新题暴露遗忘表达、旧值回查、陈述式检索路由和拒答字段问题，原始失败与首测分数保留；不是线上准确率或外部盲测。最新口径与失败详情见[记忆问答新题验收](docs/记忆问答新题验收报告.md)，可提交聚合与原文引用位于evals/baselines/memory-qa-holdout*v1-20261125.json。

## 目录结构

```
agent/
├── main.py              # FastAPI 入口（装配 + 挂载路由）
├── app.py               # Streamlit 前端（登录 + 调用后端 API）
├── api/                 # 接口层
│   ├── auth.py          #   登录接口 /login
│   ├── chat.py          #   对话接口 /chat、/health
│   ├── memory.py        #   原文分页 / 本人记忆查看与软删除
│   └── deps.py          #   依赖组装 + 鉴权 + 限流
├── services/            # 业务层
│   ├── chat_service.py  # 唯一聊天主链路：归档、压缩、召回、执行与恢复
│   └── chat_types.py    # ChatTurn、压缩状态与异常
├── agent/               # 智能体核心
│   ├── react_agent.py   #   ReAct 智能体封装
│   └── tools/           #   工具 + 中间件
├── mcp_servers/         # MCP 服务（自建天气服务的示例）
│   └── weather_server.py
├── rag/                 # RAG 服务（检索 + 向量库）
├── model/               # 模型工厂（LLM / Embedding）
├── memory/              # 分批摘要、上下文选择、长期记忆策略、Redis/内存租约
├── db/                  # 数据访问层
│   ├── repository.py    #   演示用户/记录 SQLite 读写
│   └── memory_*.py      #   SQLAlchemy 历史/记忆模型、仓库与数据库组装
├── scripts/             # 一次性脚本
│   └── init_db.py       #   建库 + 导入示例数据
├── config/              # YAML 配置
├── prompts/             # 提示词模板
├── utils/               # 工具函数（安全 / 限流 / Redis / 可观测性 / 配置 / 日志等）
└── data/                # 知识库、SQLite 库、外部数据
```

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
# Windows 激活虚拟环境
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置密钥

复制 `.env.example` 并重命名为 `.env`，填入：

```
DASHSCOPE_API_KEY=你的模型key
DASHSCOPE_BASE_URL=你的模型服务地址
SECRET_KEY=你的JWT签名密钥            # python -c "import secrets; print(secrets.token_hex(32))"
LANGFUSE_PUBLIC_KEY=pk-lf-...         # 可选：留空则关闭链路追踪
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
WEATHER_MCP_URL=你的天气MCP服务地址    # 可选：留空则天气工具不可用
REDIS_URL=redis://127.0.0.1:6379/0    # 控制实例：租约、限流、缓存键目录
FACT_CACHE_REDIS_URL=redis://127.0.0.1:6380/0  # 另一物理实例：事实缓存
REDIS_CONTROL_REQUIRED=0            # 单进程开发可为0，多进程部署用1
MEMORY_DATABASE_URL=                # 留空用 data/memory.db，可填 MySQL 专用库 URL
```

### 3. 初始化（首次运行，在项目根目录执行）

```bash
# 建 SQLite 库，导入示例数据（含预设登录账号）
python -m scripts.init_db

# 加载知识库到 Chroma 向量库
python -m rag.vector_store

# 增建独立历史/记忆表，不清空现有数据；服务首次使用时也会初始化
python scripts/init_memory_db.py --env-file .env
```

记忆库可使用 MySQL：先创建专用数据库和账号，设置
`MEMORY_DATABASE_URL=mysql+pymysql://agent:URL_ENCODED_PASSWORD@127.0.0.1:3306/agent_memory?charset=utf8mb4`。
可选 Docker 叠加文件为 `docker-compose.mysql.yml`，密码和容器内 URL 见 `.env.example`。
切换 URL 不会自动迁移旧 SQLite/Redis 数据，也不改变原演示业务库。

### 4. 启动独立的控制Redis与缓存Redis

```bash
docker compose up -d redis redis-cache
```

控制实例默认128MiB、noeviction；缓存实例默认256MiB、allkeys-lfu，预算可在.env覆盖。同一实例换DB编号无法隔离淘汰，程序会检查实例run_id及内存策略，不满足条件时仅旁路事实缓存。已有6379服务请先核对其配置/容器归属，再执行切换；不能直接清空旧实例。

不启动Redis时，REDIS_CONTROL_REQUIRED=0的单进程开发模式可用内存快速租约/限流，事实查询直接走SQL。生产或多worker应设1；启动时控制连接不可用或限流命令失败明确返回503。启动时未连接到控制实例的进程需在服务恢复后重启，以重新组装租约和限流组件。

事实缓存策略位于config/memory.yml的fact_cache；命中仍核对SQL版本和遗忘规则。测试使用另建的6391/6392实例，运行方式见改造说明，不把测试服务当正式缓存部署。

### 5. 启动后端

```bash
uvicorn main:app --reload
```

### 6. 启动前端（另开一个终端）

```bash
streamlit run app.py
```

浏览器打开 Streamlit 页面（默认 http://localhost:8501），**用测试账号 `1001` / 密码 `123456` 登录**后即可对话。

## 架构

```
Streamlit 前端 (app.py)  ── 登录 → 持有 token
      │  HTTP / SSE（带 Authorization: Bearer <token>）
      ▼
FastAPI 后端 (main.py + api/)
      ├─ 鉴权（JWT 验签 → user_id）
      ├─ 限流（Redis ZSET 滑动窗口 + Lua，按用户）
      ▼
ChatService (services/)
      ├─ SQL 原文/摘要水位/长期记忆 → SQLite 或 MySQL
      ├─ 预算内上下文选择 ← 相关事实、历史片段与近期问答
      └─ Redis/内存快速租约 + SQL 权威租约
      ▼
ReactAgent (agent/)  ── 工具 (agent/tools/)
      │                    ├─ RAG 检索 (rag/) → Chroma
      │                    ├─ 业务数据 (db/) → SQLite
      │                    └─ 天气 (MCP 协议) → 外部 MCP 服务
      ├─ 中间件：工具监控（异步）/ 模型日志 / 动态提示词切换
      └─ Langfuse：链路追踪（耗时 / token / 成本，按用户筛选）
```

## 客服诉求 JSON 输出实验

运行 `python scripts/extract_request.py "我的 X10 扫地机器人无法充电"`，使用现有模型配置提取
`intent`、`product_model`、`summary`。未提供型号时返回 `null`，不猜测型号。

该实验独立于现有聊天链路：模型端使用 `response_format={"type":"json_object"}`（JSON Mode），
提示词提供 JSON Schema，应用端使用 Pydantic 严格校验类型、必填字段及枚举，并禁止额外字段。
校验失败最多纠错一次，仍失败则抛出明确异常；接口调用错误直接向上传播。
这不是原生 Structured Outputs，JSON Mode 不保证符合 Schema；当前配置模型的接口兼容性需实测。
结构校验也不能证明型号、意图等语义正确，更不能代替身份认证及工具授权。

离线回归：`python -m pytest tests/test_structured_output.py -q`。
测试使用模拟模型，覆盖合法输出、非法 JSON、错误类型、枚举、缺失及额外字段、重试上限和空输入，
不代表线上模型准确率或接口可用性。真实调用会使用 `.env` 中的模型 API 并产生调用费用。

## License

仅供学习参考。
