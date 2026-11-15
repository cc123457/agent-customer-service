# 智能客服 Agent（RAG + ReAct，前后端分离）

基于 **LangChain 1.0 + LangGraph** 的企业级智能客服系统，当前以扫地/扫拖机器人场景为例。
后端为 **FastAPI 微服务**（SSE 异步流式），前端为 **Streamlit**（纯客户端），两者通过 HTTP 完全解耦。

> 知识库与业务数据均为示例，可替换为任意行业的知识文档与数据库。

## 技术栈

| 层 | 技术 |
|---|---|
| 智能体框架 | LangChain 1.0 `create_agent`（ReAct）+ LangGraph 运行时 |
| RAG | Chroma 向量库 + DashScope `text-embedding-v4` + 递归分块 |
| 大模型 | Qwen（OpenAI 兼容协议接入） |
| 后端 | FastAPI + SSE 流式输出（异步链路） |
| 前端 | Streamlit（纯客户端，通过 HTTP 调后端 API） |
| 数据 | SQLite（用户 / 记录）+ Chroma（向量）+ Redis（会话 / 限流） |
| 认证 | JWT（登录签发 / 接口鉴权 / 用户身份注入 Agent） |
| 限流 | Redis ZSET 滑动窗口（Lua 原子）+ 内存降级 |
| 可观测性 | Langfuse（链路追踪 / token 成本 / 按用户筛选） |
| 外部工具 | MCP 协议接入（天气服务） |
| 会话记忆 | Redis List + 内存降级 |
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
- **会话记忆持久化**：Redis List 存储多轮对话（重启不丢、多实例共享），Redis 不可用时降级内存
- **前后端分离**：FastAPI 后端 + Streamlit 前端，通过 REST / SSE 通信
- **真实数据**：用户与使用记录存于 SQLite，替换原 mock 数据
- **知识库增量入库**：文件 MD5 去重，避免重复向量化

## 目录结构

```
agent/
├── main.py              # FastAPI 入口（装配 + 挂载路由）
├── app.py               # Streamlit 前端（登录 + 调用后端 API）
├── api/                 # 接口层
│   ├── auth.py          #   登录接口 /login
│   ├── chat.py          #   对话接口 /chat、/health
│   └── deps.py          #   依赖组装 + 鉴权 + 限流
├── services/            # 业务层
│   └── chat_service.py  #   会话管理 + 驱动 Agent
├── agent/               # 智能体核心
│   ├── react_agent.py   #   ReAct 智能体封装
│   └── tools/           #   工具 + 中间件
├── mcp_servers/         # MCP 服务（自建天气服务的示例）
│   └── weather_server.py
├── rag/                 # RAG 服务（检索 + 向量库）
├── model/               # 模型工厂（LLM / Embedding）
├── memory/              # 会话记忆存储（Redis / 内存）
├── db/                  # 数据访问层
│   └── repository.py    #   SQLite 读写
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
REDIS_URL=redis://127.0.0.1:6379/0    # 可选：连不上自动降级内存
```

### 3. 初始化（首次运行，在项目根目录执行）

```bash
# 建 SQLite 库，导入示例数据（含预设登录账号）
python -m scripts.init_db

# 加载知识库到 Chroma 向量库
python -m rag.vector_store
```

### 4. 启动 Redis（可选，用于会话记忆 + 限流持久化）

```bash
docker run -d --name my-redis -p 6379:6379 redis:latest
```

不启动 Redis 也能跑（会话与限流自动降级为内存实现）。

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
ChatService (services/) ── 会话记忆 (memory/ → Redis，降级内存)
      ▼
ReactAgent (agent/)  ── 工具 (agent/tools/)
      │                    ├─ RAG 检索 (rag/) → Chroma
      │                    ├─ 业务数据 (db/) → SQLite
      │                    └─ 天气 (MCP 协议) → 外部 MCP 服务
      ├─ 中间件：工具监控（异步）/ 模型日志 / 动态提示词切换
      └─ Langfuse：链路追踪（耗时 / token / 成本，按用户筛选）
```

## License

仅供学习参考。
