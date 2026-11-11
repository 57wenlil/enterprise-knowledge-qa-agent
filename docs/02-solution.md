# 02｜方案设计、动态决策与技术取舍

## 技术与复用

选择 Python 3.11+、标准库 Markdown 文档仓库与本地中文 token IDF 检索，避免在笔试环境部署 Milvus/BGE 依赖；可选 LangGraph `StateGraph` 驱动两节点，OpenAI-compatible `chat/completions` 负责动作 JSON。`app/graph/main_graph.py` / `state.py` 从上传的物业客服 Agent 框架改造；`app/llm/client.py` 保留其兼容模型适配设计；`app/retrieval/chunker.py` 基于高校 RAG 的 Parent–Child 子块思想重写，稳定 `chunk_id`+`offset`；`app/retrieval/search.py` 改造物业项目的 lexical scoring，借鉴高校项目的检索分层接口。详细文件对照见 README。

实际工程不加载两份原项目的所有依赖与业务模块，避免引入对题目无关的物业接口、OCR 和 Milvus 服务。高校项目的 BGE-M3 / Milvus 混合检索思路是明确的升级路线，但在这个包内并未实际执行。

## 数据模型与工具协议

元数据维护在 `data/catalog.json`，原文维护在 `data/docs/Dxx.md`，每个条目包含稳定 `doc_id`、标题、版本、`version_date`、生效时间、适用范围、来源、主题、可选规范性主张键和值，以及有无已知主管部门。仓库先校验路径与生效日期，原文由 `read_document(doc_id)` 返回，避免模型在搜索摘要上“脑补”。

`search_documents(query,top_k)` 负责返回候选 `doc_id`、原文片段、Parent–Child `chunk_id`、版本、适用范围、相关性等。小规模数据可用本地 2/3 字 n-gram + IDF 检索；工具结果字段不依赖检索后端，未来可将检索器替换成原高校项目的混合检索+重排序。搜索 `top_k` 默认 5（演示策略使用 6），限制最大 8；即使 `top_k=1` 也携带命中主题的冲突制度候选。

## 模型循环

真实模型模式下，每轮 `LLMDecisionModel.decide(view)` 都通过 API 请求 JSON。`view` 包含问题、最近多轮聊天历史、先前工具结果、检验证据失败反馈以及剩余动作预算。模型四类动作：`search_documents`（补充/改写检索）、`read_document`（取原文）、`ask_user`（需要用户补充信息）、`finish`（含答案或拒答状态）。

模型决定后，`execute_tool` 对参数做边界验证、调用工具、把成功或失败的**可观察结果**带回模型。`finish` / `ask_user` 会被确定性证据校验器验证；失败时由校验器提供可公开的错误摘要，**下一步仍交由模型重选动作**。模型输出 `reason_summary` 用于解释当前动作依据，不记录模型内部思维链。

若系统没有安装 LangGraph，运行**完全相同的节点函数**但用受限 Python 循环驱动，Trace 会显示同样的动作。这仅是零依赖演示兜底，不应作为“已运行 LangGraph”的证明；安装后即切换为编译后的图执行。

## 冲突规则

查证 D01/D02 同一个 `claim_key=travel_reimbursement_deadline`、同适用范围，而 `claim_value` 分别为 30 天、15 天；两者都生效，现有合成资料缺失优先级规则。即便 D02 的版本日期更晚，也不能自动推断“新文档取代旧文档”。模型应读取两个全文并对照，再选择 `ask_user` 或带证据的冲突说明；校验器阻止以 `completed` 输出单一断言，也阻止直接将已发现冲突笼统标记成无资料。若用户明确要求“仅根据 D01”，系统允许**限定文档范围**的回答，并声明不代表 D01 优先于其他规定。

该冲突识别是小样本元数据驱动的工程性检查，不是完整法律规范推理：复杂的范围重叠、上下位法、多规则例外需扩充 metadata 与规则引擎。

## 引用与数量校验

`EvidenceChecker` 把已读文档的 `content` 作为事实源，只允许 `quote` 是读到的 Markdown 原文逐字子串；引用 ID、版本和来源由仓库添加，禁止 LLM 伪造元数据。完成的回答至少一条引用，并先执行检索+读取。答案中“3工作日、5天、30自然日”等明显数量也必须在所引文本出现；数量单位检查防止简单编造，但不等于全面事实一致性证明。

## 异常和运行边界

结构化动作仅支持四种类型；工具仅允许两个只读入口，限制 query 长度、ID 形态、top_k。模型失败返回 `error`；工具失败被写入 observations 和 Trace，由模型决定降级（离线演示会明确停止）；最大决策步骤默认 8（可设 2–25），达到上限返回 `incomplete`。返回结构一致，便于 CLI 和未来 API 复用。

## 保密边界

人工合成资料，没有真实凭证；不存储 API Key，不上传私钥、不记录模型内部推理。生产阶段需补充企业文档 ACL、租户隔离、Prompt Injection 测试、可追溯审计，以及制度变更缓存失效机制。
