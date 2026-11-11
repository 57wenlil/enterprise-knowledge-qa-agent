# 企业知识问答 Agent｜校招笔试项目 3

> 目的：帮助员工找到制度 / FAQ 问题的可靠答案。**用工具获得证据，再由模型动态决定下一步**；输出文档 ID、版本日期与逐字原文引用，对资料冲突、资料缺失及工具异常不擅自编造答案。
>
> **注意：`data/` 全部是为笔试构造的合成制度，不是任何真实公司的正式规定。**

## 0. 快速验收（无 API Key、无 Milvus、无 LangGraph 也能运行）

要求 Python 3.11+。在项目根目录执行：

```bash
python -m app.cli --mode demo --ask "年假有几天？"
python -m app.cli --mode demo --ask "差旅报销期限是多少天？" --json
python -m app.cli --mode demo --ask "年假有几天？" --followup "那要提前多久申请？"
python -m app.cli --mode demo --ask "年假有几天？" --fail-tool search_documents --json
python -m pytest -q
python -m scripts.generate_examples
```

如缺 `pytest`：`python -m pip install pytest`。纯 Python 的离线演示模式不需要任何其他第三方依赖。

**重要模式声明：** `--mode demo` 用明确标注的离线规则策略模拟 `model.decide()`，仅用于验证工程闭环、证据与边界，不代表实测 LLM 的自主决策正确率。真实模型决策请按下文设置 `--mode llm`。两种模式共用相同 `StateGraph` / 决策工具节点、引用校验和 Trace 数据结构。

## 1. 真实 LLM 驱动（提交场景推荐）

安装可选运行库：

```bash
python -m pip install -r requirements.txt
# macOS/Linux:
export LLM_API_KEY="你的API密钥"
export LLM_MODEL="qwen-plus"
export LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
python -m app.cli --mode llm --ask "差旅报销期限是多少天？" --json
```

支持其他兼容 OpenAI `/chat/completions` 且返回 JSON 对象的服务，只需更换 `LLM_BASE_URL`、`LLM_MODEL` 和 `LLM_API_KEY`，无需重写工具。Windows PowerShell 使用 `$env:LLM_API_KEY="..."`。密钥不得提交到仓库，`.env.example` 仅是说明模板。

模型每轮必须输出严格 JSON 的 `action`，可在 `search_documents`、`read_document`、`ask_user`、`finish` 之间动态选择；工具执行结果会作为下一轮决策输入。未配置 API Key 将明确返回 `error`，**不会偷偷切换为规则模拟**。本次交付环境没有可用的外部模型密钥，因此没有宣称完成真实模型端到端准确率测试；需在本人环境运行并保存真实 LLM Trace。

## 2. 可复用来源（两份上传的代码）

| 现有项目 | 复用点 | 本项目的位置 / 改动 |
|---|---|---|
| `customer_service_agent.zip` | `app/graph/main_graph.py` 的 LangGraph StateGraph、条件路由、执行流 | `app/graph/main_graph.py`：重写为单 Agent `model_decision → execute_tool → model_decision` 闭环；去除物业工单和不必要业务 |
| `customer_service_agent.zip` | `app/graph/state.py` 的 TypedDict AgentState | `app/graph/state.py`：精简为问题、历史、模型动作、工具观察、Trace、状态 |
| `customer_service_agent.zip` | `app/llm/client.py` 的 OpenAI-compatible HTTP JSON 调用 | `app/llm/client.py`：保留接口思路，新增专用于逐步决策的提示词、错误隔离 |
| `customer_service_agent.zip` | `app/rag/knowledge_base.py` 的中文词面检索基础 | `app/retrieval/search.py`：修复弱匹配，引入中文 2/3-gram、IDF、同主题冲突候选召回 |
| `高校智慧问答RAG (2).zip` | `core/document_processor.py` 的 Parent–Child 文档切块与来源元数据 | `app/retrieval/chunker.py`：只用标准库重写，稳定 chunk ID、原文偏移、父子关联 |
| `高校智慧问答RAG (2).zip` | `core/vector_store.py` 的文档检索 / 重排序分层思想 | `app/retrieval/search.py` / `app/tools.py`：预留同名检索工具替换点；**未接入** Milvus/BGE-M3（不误报已运行） |

旧项目依赖重、模型权重及 Milvus 数据不随压缩包提供，所以本次为可复现笔试交付采用本地索引。要切换为高校项目原 BGE-M3 + Milvus，只替换 `LocalRetriever.search` 的实现并保持输出字段及 `KnowledgeTools` 接口即可。**重用的是框架、状态、客户端、分块算法和检索设计，并非把两个原项目原样复制成一个大工程。**

## 3. 工程结构

```text
enterprise_knowledge_agent_submission/
├── README.md
├── app/
│   ├── cli.py                    # CLI / 多轮对话 / JSON 导出
│   ├── service.py                # Session history / runtime assembly
│   ├── graph/
│   │   ├── main_graph.py         # LangGraph 决策/执行闭环 + 同节点离线驱动
│   │   └── state.py              # AgentState
│   ├── llm/
│   │   ├── client.py             # 真实 OpenAI-compatible LLM JSON 控制器
│   │   ├── decision.py           # Action JSON 白名单与字段校验
│   │   └── demo.py               # 显式离线模拟，非真实模型
│   ├── retrieval/
│   │   ├── chunker.py            # Parent-Child 切块与稳定 offset/ID
│   │   ├── repository.py         # 文档 ID / 版本 / 原文管理
│   │   └── search.py             # 本地轻量中文检索
│   ├── evidence.py               # 逐字引用、数字、同范围制度冲突校验
│   └── tools.py                  # search_documents / read_document
├── data/
│   ├── catalog.json              # 合成 FAQ/制度元数据
│   └── docs/D01.md ... D06.md    # 六份合成原文
├── tests/                       # 六类验收 + 防护 / 模型接口测试
├── examples/                    # 实际离线运行输出与完整 Trace
├── scripts/generate_examples.py # 复现实验样本
└── docs/                         # 题目要求的五篇交付文档
```

## 4. Agent 决策协议

工具 1：`search_documents({"query":"...","top_k":5})`，返回 `doc_id`、`title`、`version`、`version_date`、`scope`、`chunk_id`、`snippet`、`score` 等。

工具 2：`read_document({"doc_id":"D03"})`，返回上述元数据与对应 Markdown **完整原文**。

控制器行动：`ask_user`、`finish` 并非伪造工具：它们是模型可以选取的终止动作。`finish.status` 可为 `completed`、`insufficient_evidence` 或 `error`。检测到无法确定优先级的冲突，推荐模型选 `ask_user`。

数据流：

```text
用户问题 / 对话历史
    ↓
model_decision --模型JSON动作--> search_documents / read_document
    ↑                                 │
    └─────────── Tool observation ────┘
    ↓
引用与冲突校验；成功：完成/追问；失败：反馈模型重选动作
    ↓
结构化输出 status, answer, citations, missing_information, trace
```

**约束与失败策略：**

- 只有已通过 `read_document` 读取过的文档可作为 `completed` 答案的引用。引用 `quote` 必须是原文的**逐字子串**；版本日期等信息由资料仓库填充，不信任模型自填。
- 答案的明确数量必须出现在所引原句中。该检查只是明显幻觉拦截，不声称解决全部语义幻觉。
- 对同一主题、相同适用范围、相互矛盾且没有制度优先级的候选，强制禁止输出单一肯定结论；必须读两份原文并说明冲突/追问，不能简单认定新版本一定优先。
- 搜索 `top_k=1` 时仍补充同主题冲突候选，不容靠狭窄召回规避审查。
- 任一工具失败会保留失败观察；超过最大模型决策步数时返回 `incomplete`。默认上限 8，可用 `--max-steps` 修改（2–25）。
- 基础模型动态选择工具，业务校验器**只否决不安全结论并提示修正**，不替模型写死唯一的决策顺序。

## 5. 真实交付样例（离线可复现）

**路径 A，明确证据：** `年假有几天？` → 搜索 → 读取 D03 → 引用 D03“5天”原句 → `completed`。

**路径 B，冲突核查：** `报销期限多少天？` → 搜索返回 D01/D02 → 逐份读取 → 发现“30个自然日”与“15个自然日”同一适用范围、无优先级规则 → `needs_clarification` 并引用两份原文。

**路径 C，资料外：** `明天股票会涨吗？` → 搜索无命中 → `insufficient_evidence`，不补造答案。

**路径 D，上下文追问：** `年假有几天？` 后问 `那要提前多久申请？` → 结合对话历史定位 D03 → 引用“提前3个工作日”原句。

**路径 E，故障：** `--fail-tool read_document` → 记录工具异常 → `error`，无伪造引用。

可查 `examples/acceptance_cases.json`（全部 6 个指定验收输入/实际输出/Trace）、`examples/trace_clear.json`、`examples/trace_conflict.json`。

## 6. 验证与可解释性

```bash
python -m pytest -q
python -m scripts.generate_examples
python -m app.cli --mode demo --ask "差旅报销期限多少天" --json \
  --trace-out examples/conflict_local.json
```

`trace` 的每个步骤记录：`stage`（decision/tool/evidence_guard）、`step`、模型选定 `action`、调用 `arguments`、工具 `success`、`result_summary` 和简要 `reason_summary`。不记录模型内部隐式推理、API Key；没有把推断性“思维链”写成模型输出。

**已验证范围**：本交付环境已运行 Python 单元/集成测试，全部通过；完整测试清单见 [`docs/03-evaluation.md`](docs/03-evaluation.md)。未验证范围：真实 LLM 网络调用的正确率、BGE/Milvus 部署质量、LangGraph 可选依赖在无网络环境中的安装与实际执行。笔试评估真实模型能力时应运行 `--mode llm`，不能把 demo 测试成绩当成模型基准表现。

## 7. 其他文档

- [`docs/01-requirements.md`](docs/01-requirements.md) — 用户、范围和成功标准
- [`docs/02-solution.md`](docs/02-solution.md) — 决策协议、工具接口、工程取舍
- [`docs/03-evaluation.md`](docs/03-evaluation.md) — 六类验收、故障、已知不足
- [`docs/04-ai-usage.md`](docs/04-ai-usage.md) — AI 辅助范围、人工核验待办与材料说明
- [`docs/05-retrospective.md`](docs/05-retrospective.md) — 最难问题、学习收获和下一步

## 8. 提交提醒

提交**该项目本身的私有 GitHub 仓库 URL + 最终 commit SHA + 题号 3**。将本文件夹内容上传为同一仓库；不要把上传的两个原项目及真实账号密钥一起提交。`examples/` 中的 Trace 为公开可分享的合成测试结果。外部模型实测需自行配置密钥并将**脱敏后的**结果纳入新 commit。
