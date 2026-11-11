# 03｜评估：用例、执行路径、结果与限制

## 可复现实验设置

- Python 3.11+；以 `data/catalog.json` 的 6 份 Markdown 合成制度为唯一知识源，测试固定 `as_of=2026-11-11`。
- `python -m pytest -q` 运行自动化测试；`python -m scripts.generate_examples` 生成独立可审查的实际输入/结果与 Trace。
- 使用 `mode=demo` 的**离线动作测试替身**，不调用远程 LLM。两条不同路径、工具观察、冲突保护和引用格式均为真实代码执行结果，但不是实际模型识别质量评估。
- 当前环境不能联网安装 LangGraph，执行的是与 StateGraph 定义相同的 Python 节点函数；实际图运行待安装依赖后验证。

## 指定六类用例

| 编号 | 输入 | 预期状态 | 已观察结果（离线） | 检验要点 |
|---|---|---|---|---|
| 1 明确答案 | 年假有几天？ | `completed` | 通过 | D03 原文“5天”，搜索→读取→完成 |
| 2 制度冲突 | 差旅报销期限是多少天？ | `needs_clarification` | 通过 | 检索 D01/D02→读取双方→引用 30/15 天→确认优先级 |
| 3 资料外 | 明天哪只股票涨得最多？ | `insufficient_evidence` | 通过 | 搜索无命中，引用空，不胡编 |
| 4 相似改写 | 员工入职一年可以休多少天带薪假？ | `completed` | 通过 | 同义词“带薪假/年假”召回 D03 |
| 5 上下文追问 | 年假有几天？→那要提前多久申请？ | `completed` | 通过 | 继承年假主题、引用 D03“提前3个工作日” |
| 6 工具失败 | 故障注入 search_documents | `error` | 通过 | 工具失败 Trace + 明确错误、无伪造引用 |

## 附加边界验证

- `read_document` 工具失败不编造；指定 D01 可作有边界限定的答复，不宣称 D01 具有最高优先级。
- 模型在尚未读取原文时试图完成，引用校验阻断并允许下一步再读取；篡改引文、编造 99 天、在 D01/D02 冲突时直接选单一答案都被拦截。
- 模型反复搜索，超过设置上限后 `incomplete`；未知动作/非法参数返回明确错误；文档 ID 路径校验；稳定切块偏移与可复现 chunk ID。
- 模型只要求 `top_k=1` 也会携带同主题的冲突候选；无法将已检出的冲突无引证地伪装成“无资料”。
- 真实模型接口以 Stub Client 验证“模型返回哪种 JSON 动作，执行器就选哪种动作”，但没有远程质量评估。

## 实际执行记录

命令：

```bash
python -m pytest -q
# 21 passed
python -m scripts.generate_examples
```

可审查的完整 JSON 在：

- `examples/acceptance_cases.json` 六种用例全部输入/输出/trace
- `examples/trace_clear.json` 普通答案完整路径
- `examples/trace_conflict.json` 冲突核查完整路径
- `examples/trace_context.json` 多轮追问路径
- `examples/trace_tool_failure.json` 工具失败路径

普通答案动作顺序：`search_documents → read_document → finish`。

冲突核查动作顺序：`search_documents → read_document(D01) → read_document(D02) → ask_user`。

每次工具调用都附 `arguments`、`success`、`result_summary`；模型动作附观察驱动的 `reason_summary`。完整回传结果与测试状态采用脚本从实际执行生成，不是手工编造的 Trace。

## 局限

1. 检索样本规模小且是合成语料；中文词面检索无法保证真实大规模语义检索的召回、精度；BGE-M3/Milvus 模块为明确替换点，未部署。
2. `--mode llm` 的提示词、模型偏好、接口 JSON 兼容性和模型工具选择质量必须通过真实凭证进一步测试。**离线模拟的 21 项测试通过不是 LLM 准确率 100%。**
3. 文档冲突判断基于同一主张键、同范围和不同值，复杂上下位制度、范围部分覆盖、明确废止的情况需进一步建模。
4. 引文逐字校验与数字守卫不等于所有语义断言均被证明；上线前需要人工评审和更强的事实一致性检查。
5. 尚未针对真实企业权限、Prompt Injection、并发、大量文档或成本进行性能/安全测试。
