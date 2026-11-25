# 当前评测入口

两个固定开发集测试不同能力：memory_cases.json有30个记忆抽取/召回/冲突案例；agent_cases.json有48个Agent案例和12个接口校验案例。它们不等于独立质量测试集。

## 冻结实现后的80题记忆问答新验收

最新结果见[记忆问答新题验收报告](../docs/记忆问答新题验收报告.md)。`memory_qa_holdout.json`另编80题（10类各8题），原文、编号、问题与旧开发题分开；55题可回答、25题应拒答/澄清。运行前保存生产实现、提示词、配置、评测器及数据SHA256，首轮结束前后均核对，未用新结果修改实现或标签。

四方案使用同一qwen3-omni-flash，正式400次聊天/摘要调用均取得usage；全部320条输出由当前助手逐条复核，不将规则分数复制为语义准确率。当前记忆正文正确71/80＝88.75%；按冻结的正文、拒答字段和受限值三项严格口径为66/80＝82.50%，仅近期27/80＝33.75%，摘要加近期37/80＝46.25%，完整历史67/80＝83.75%。遗忘及检索路由失败保留，另有1题失败轮次事实有效性的产品语义争议，主成绩不改标签。人工评分仍空，不称外部独立盲评。

```powershell
# 已冻结文件不要覆盖；复现结果必须另起目录并全部保留
python scripts/run_memory_holdout.py verify --freeze evals/freezes/memory-qa-holdout-v1-20261125.json
python scripts/run_memory_holdout.py run --freeze evals/freezes/memory-qa-holdout-v1-20261125.json --output evals/runs/NEW-holdout --env-file .env
python scripts/review_memory_holdout.py prepare --source evals/runs/NEW-holdout --output evals/runs/NEW-holdout-review
# assessments需要实际逐条判断；工具只校验来源和汇总，不自动代填
python scripts/review_memory_holdout.py aggregate --manifest evals/runs/NEW-holdout-review/manifest.json --assessments YOUR_ASSESSMENTS.jsonl --output evals/runs/NEW-holdout-review/summary.json
```

历史事实提取使用规则，真实摘要/Embedding/回答使用临时SQLite/Chroma；不包含JWT、产品知识库或业务工具Agent，不代表整个客服系统准确率。可提交的冻结/聚合及完整AI判定分别为`baselines/memory-qa-holdout-v1-20261125.json`和`baselines/memory-qa-holdout-review-v1-20261125.json`；原始回答、用量与证据包保留在runs目录。当前方案冷启动参考费用仍增加，不能以回答输入较短宣传降本。

## 记忆修复与费用补测

`refinement_cases.json`另编24例预算更正、生效计划、运行时间及遗忘题；`temporal_boundary_cases.json`补8条跨月、跨年与生效零点。原48题用于已知问题回归，新题也只有有限模板多样性；一旦据其失败继续修改实现，就应视作开发回归，不能再称独立盲测。运行结果、失败保留和简历建议见[记忆优化实测报告](../docs/记忆优化实测报告.md)，逐文件前后对照见[实现说明](../docs/记忆修复实现说明.md)。可提交汇总为`baselines/memory-refinement-20261125.json`，阶段结果、超时与403原始运行均保留。

```powershell
# --chat-model仅作用于本次评测；不写回config/rag.yml
python scripts/evaluate_history.py --live --chat-model qwen3-omni-flash --split test --env-file .env --min-call-interval 1.5 --output evals/runs/NEW-refinement-regression
python scripts/evaluate_history.py --live --chat-model qwen3-omni-flash --dataset evals/refinement_cases.json --split test --env-file .env --min-call-interval 1.5 --output evals/runs/NEW-refinement-acceptance
python scripts/evaluate_memory_quality.py --live --chat-model qwen3-omni-flash --suite all --split test --env-file .env --min-call-interval 1.5 --output evals/runs/NEW-refinement-quality
python scripts/report_memory_cost.py --source evals/runs/NEW-refinement-regression --free-quota-reported --output evals/runs/NEW-reference-cost.json
```

费用读取API实际usage，按案例与阶段区分摘要、各方案回答、Embedding索引与查询、独立检索诊断。当前冷启动费用只计当前方案实际承担部分；免费额度由用户说明，未读账单时实际支付保留为空。`--free-quota-reported`不会将估算原价改成0元。缺usage或请求失败时不报告有效总成本降幅；旧运行未记录Embedding末端usage，不能按字节数追补Token。官方参考价见config/model_prices.yml。

真实模型抽取校准遇到外层JSON代码围栏，现仅清除外围后继续严格Pydantic与原文依据校验，raw保留。401/403会阻断同一运行后续模型调用并记录未执行项；不会自动轮换模型。不同模型、实现指纹和中断记录单独保存，不拼接成一次成功运行。

## 抽取、摘要和人工评审补充

新增`quality_cases.json`：抽取12例校准＋48例验收；摘要6例校准＋24例验收。通过生产FactExtractor/BatchSummarizer运行真实模型，不把金标准送入模型。`quality_metrics.py`分别计算严格候选Precision/Recall/F1、负例误写、关键事实保留和有限已知错误检查。历史评测同时输出固定Precision@3、按实际返回的精确率和拒答混淆矩阵。完整口径、结果及已知评分误差见[记忆质量补充评测](../docs/记忆质量补充评测.md)。

```powershell
# 默认只测离线规则；--live才调用真实模型
python scripts/evaluate_memory_quality.py --output evals/runs/NEW-rules
python scripts/evaluate_memory_quality.py --live --suite all --split calibration --env-file .env --output evals/runs/NEW-quality-calibration
python scripts/evaluate_memory_quality.py --live --suite all --split test --env-file .env --output evals/runs/NEW-quality-test
# 既有真实回答可直接离线重评分，不必重复花费模型调用
python scripts/rescore_history.py --source evals/runs/history-test-paced-20261124 --output evals/runs/NEW-rescore-history.json
python scripts/rescore_quality.py --source evals/runs/quality-test-20261124 --output evals/runs/NEW-rescore-quality.json
# 导出方案盲评网页；导入真实人工填写的JSON，未评条目保持空白
python scripts/review_memory.py export --source evals/runs/history-test-paced-20261124 --output evals/runs/NEW-review
python scripts/review_memory.py import --key evals/runs/NEW-review/blind_key.json --reviews YOUR_EXPORTED_JSON --output evals/runs/NEW-human-scores.json
```

人工评审页左侧显示按时间排列的全部可用原文，默认先看标注证据；不是检索排名，不能按“第一条”机械选择。右侧每次评一个回答，内容、拒答标记和禁止值泄露分开判断。只有真人填写并导入后才有人工准确率，程序/模型打分不填human_correct。目前用户要求人工环节暂停。`evals/baselines/quality-coverage-20261124.json`保存可提交聚合结果，运行明细仍被Git忽略。

用户后续授权由当前助手完成AI语义复核：192条已全量审阅，分别评价完整历史任务结果和基于实际输入的回答行为，每条含真实引用与理由；人工评分继续保持空白。报告见[记忆回答AI复核](../docs/记忆回答AI复核报告.md)，聚合基线为`evals/baselines/ai-review-20261124.json`，完整带理由记录为`ai-review-20261124-records.json`。`scripts/review_memory_ai.py`提供证据包整理、判定完整性/引用校验和只读页面生成，不能把规则分数复制成AI复核。该评审者参与过数据实现，不称独立盲测。

## 历史检索标注集与记忆价值对照

`history_cases.json`是另外编写的60例合成数据：12例校准、48例验收，用户和原始会话互不共享。每例包含问题、提问时间、原始双方对话、正确证据轮次、答案别名和禁止泄漏值。包含同会话早期信息、跨会话、助手处理结论、换说法、多来源、时间、更正、遗忘、缺少证据、用户隔离和失败轮次。

```powershell
# 默认仅离线BM25，不调用模型
python scripts/evaluate_history.py --output evals/runs/history-offline
# 真实Embedding：BM25 / 语义 / 混合三路比较
python scripts/evaluate_history.py --embedding --env-file .env --output evals/runs/history-retrieval
# 校准后再验收；输出目录必须是新目录，防止混用记录
python scripts/evaluate_history.py --live --split calibration --env-file .env --output evals/runs/history-calibration
python scripts/evaluate_history.py --live --split test --env-file .env --output evals/runs/history-test
```

`--live`在临时SQLite里通过现有租约/事务写入原文，使用当前`ChatService.open_turn()`、真实摘要、长期事实索引和历史索引组装输入，再由同一个真实模型回答。事实提取仅采用规则，未运行模型提取器，也未执行产品知识库或业务工具Agent，不能将此项结果当作完整客服端到端准确率。基准不连接正式数据库，不测Redis缓存命中率。

四种输入分别为：仅当前会话最近20条消息；当前会话摘要加压缩后近期消息；当前记忆方案（含相关长期事实和历史片段）；本人全部已完成历史作为参考。完整历史有独立32768估算单位上限，超出明确报错。未来轮次模拟尚未发生，不写入临时库；这不能证明库里已存在未来行时的过滤行为。外人和失败轮次实际入库，用于检查SQL隔离。

- 检索主指标是Hit@3、宏平均Recall@3、MRR@3及全部证据覆盖率。多来源问题仅命中一份也算Hit，但不算证据找全。没有正确证据的负例单列非空返回数，不混入召回分母；返回片段不等于答案泄漏。
- 回答以Pydantic严格校验结构，再检查短答案的全部必要内容、拒答标记和禁止值；这是规则评分，`human_correct`仍为空，人工复核需另做。格式修正最多一次，失败留在分母，不用金标准修补事实。
- 实际模型Token由提供方usage记录；估算上下文单位另列。摘要一次生成供两个方案复用，实际账单只算一次，摊销比较时应分别算入各自方案。Embedding只记SDK操作数/输入字节，一次SDK操作可能有多批HTTP请求及重试，精确Token未知。
- 索引建设时间与热索引查询时间分开。检索和回答顺序轮换、并行度默认2；延迟不代表生产容量或严格隔离的性能基准。
- 真实模型请求默认至少间隔1.25秒，可用`--min-call-interval`调整；人为限速排队时间单列，模型耗时不包含这段等待。首次验收曾出现429，原结果保留，随后只调整评测节奏并全量重跑，未修改检索实现、提示词或标签。

输出包含`manifest.json`（数据、评测器及实现指纹）、逐题`results.jsonl`（原答、上下文、证据、耗时与用量）、总表和分类`summary.json`。合成集同类题有模板相似性，未经人工二次审核，不能称为外部盲测或线上准确率。校准发现的模型错误仍保留；验收后不按失败项改数据或调检索参数。本轮结果与限制见[历史检索评测报告](../docs/历史检索评测报告.md)。

## 记忆开发集

~~~powershell
python scripts/evaluate_memory.py
python scripts/evaluate_memory.py --semantic --env-file .env --output evals/runs/memory-semantic.json
~~~

默认不调用模型，使用规则和SQL词法召回；--semantic使用真实Embedding和临时Chroma。检查跨会话输入事实命中、旧值错误召回和应澄清标记，不生成最终回答，所以结果不是问答准确率。普通回归使用可控概念向量，只证明索引/程序路径。

## 接口与Agent开发集

~~~powershell
python scripts/evaluate_agent.py --suite contract
python scripts/evaluate_agent.py --suite agent --limit 1 --env-file .env
python scripts/evaluate_agent.py --suite all --env-file .env --concurrency 2
~~~

contract向真实FastAPI路由发送ASGI请求，替换认证和业务依赖，检查字段规则，不调用模型，不能据此证明JWT质量。

agent与all会调用真实模型和工具；每例组装当前唯一ChatService，使用独立临时SQLite档案、SQL检查点和个人记忆索引，同例多轮共享数据。配置启用模型提取/语义检索时也会产生相应调用。产品知识库与演示业务数据仍是项目现有内容，天气取决于外部MCP。

显式--env-file只在评测进程覆盖配置并关闭Langfuse远程追踪，不打印密钥。默认输出evals/runs下的results.json、summary.json、trace.log，已被Git忽略；输出可能包含业务内容，不应直接公开。

- completed表示流程正常结束，不代表答案正确。
- term_check是字面关键词筛查；human_correct保留为人工复核字段。
- case_latency包含整个案例所有轮次；首个answer事件不是逐Token TTFT。
- calls和usage_by_model来自附加在Agent链路上的回调，不保证覆盖独立摘要/提取及Embedding，也不能把缺失用量当作0。

## 功能冒烟与真实数据库回归

~~~powershell
python scripts/verify_memory_flow.py
python scripts/verify_memory_flow.py --live --env-file .env --output evals/runs/memory-flow-live.json

$env:TEST_MEMORY_DATABASE_URL='mysql+pymysql://tester:PASSWORD@127.0.0.1:3306/agent_memory_test?charset=utf8mb4'
$env:TEST_REDIS_URL='redis://127.0.0.1:6379/15'
$env:REQUIRE_MYSQL='1'
$env:REQUIRE_REDIS='1'
python -m pytest tests/ -q -p no:cacheprovider
~~~

MySQL测试库名必须以agent_memory_test开头，测试结束会清理其中的测试表，不要指向业务库。Redis仅清理测试命名空间，不执行FLUSHDB。独立pytest覆盖JWT、租约、真实SQL事务、检查点恢复、配额与归档；评测脚本不能替代这些验证。


## 2026-11-25：路由与遗忘升级对照

实现和阅读顺序见[改造说明](../docs/记忆路由与遗忘改造说明.md)。`memory_routing_holdout.json`为升级后另编的40题，四组是近期、旧记忆快照、新记忆和完整历史。`scripts/evaluate_memory_routing.py`在首次模型调用前冻结源码、数据、相同问答模型和相同Embedding；旧新版本分进程载入，避免Python模块混用。旧版80题和82.50%首测保留，不能用修复后的重跑覆盖。

新对照对旧新两组都使用qwen3-omni-flash和text-embedding-v3/1024维，各自重新建冷索引。历史准备调用各自的生产分析器及SQL事务；当前请求调用各自的ChatService上下文准备。服务直接澄清的回答单独标记，其他回答用相同结构化评测提示。费用包含实际分析、摘要、Embedding建设/查询和回答；实际账单未知。不是完整JWT/业务工具Agent验收，也不是外部独立盲测。

执行顺序为freeze、各arm的worker、report；正式目录禁止覆盖。原始结果、上下文、usage和源指纹位于evals/runs/memory-routing-holdout-v2-20261125。验证通过后公开聚合放入evals/baselines，说明见后续验收报告。


当前交付聚合为`baselines/memory-routing-release-v2_2-20261125.json`，160份AI复核为`baselines/memory-routing-ai-review-v2-20261125.json`。首轮近期基线曾漏传轮次ID而发生40个构造错误，原记录保留并明确无对比意义；仅修正基线构造后在独立协议补跑。原记忆两组首测及完整历史原答未挑选或替换。原源码包保存于首轮目录frozen_sources.zip，当前活动源码已升级，复现旧协议应使用冻结包。

最终v2.2另跑同40题作为**已知题开发回归**，不能改称新题验收；48例抽取回归也复用既有开发集。结果、真实usage、参考费用和失败详情见[验收报告](../docs/记忆路由与遗忘验收报告.md)。
