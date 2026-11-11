"""Single-agent cyclic graph, adapted from the supplied property LangGraph.

LangGraph is used if installed; fallback runs the *same nodes* in a tiny local
loop for offline CI with no network. This fallback is not advertised as LangGraph.
"""
from __future__ import annotations
from typing import Any

from app.graph.state import AgentState
from app.llm.decision import Decision, InvalidDecision
from app.evidence import EvidenceChecker
from app.tools import KnowledgeTools


class AgentRuntime:
    def __init__(self, model, tools: KnowledgeTools, max_steps: int = 8,
                 prefer_langgraph: bool = True):
        if not 2 <= max_steps <= 25:
            raise ValueError("max_steps must be 2-25")
        self.model = model
        self.tools = tools
        self.checker = EvidenceChecker(tools.repo)
        self.max_steps = max_steps
        self.graph = self._create_graph() if prefer_langgraph else None

    def _model_view(self, state: AgentState) -> dict:
        return {
            "question": state["question"], "history": state.get("history", [])[-8:],
            "observations": state.get("observations", [])[-16:],
            "feedback": state.get("feedback", [])[-4:],
            "steps_left": self.max_steps - state.get("decision_steps", 0),
            "tool_calls": state.get("tool_calls", 0),
        }

    def decision_node(self, state: AgentState) -> dict:
        step = state.get("decision_steps", 0) + 1
        trace = list(state.get("trace", []))
        if step > self.max_steps:
            trace.append({"step": step, "stage": "guard", "reason": "maximum decision steps reached"})
            return {"decision_steps": step, "trace": trace, "phase": "end",
                    "status": "incomplete", "answer": "达到最大执行步数，尚未形成可核验答案；本轮未完成。",
                    "missing_information": ["需要补充证据或增加允许的执行步数"], "citations": []}
        try:
            decision = Decision.from_data(self.model.decide(self._model_view(state)))
        except Exception as exc:
            # Deliberately no raw request content / API key leakage in errors.
            # The model client deliberately sanitizes transport errors; no URL/key in Trace.
            from app.llm.client import ModelCallError
            detail = str(exc)[:250] if isinstance(exc, (InvalidDecision, ModelCallError)) else "模型决策调用失败"
            trace.append({"step": step, "stage": "model_error", "error_type": type(exc).__name__,
                          "result_summary": detail})
            return {"decision_steps": step, "trace": trace, "phase": "end", "status": "error",
                    "answer": "模型决策调用失败或返回非法动作，本轮终止且不编造答案。",
                    "missing_information": [type(exc).__name__], "citations": []}
        trace.append({"step": step, "stage": "decision", "action": decision.action,
                      "arguments": decision.arguments, "reason_summary": decision.reason_summary})
        updated = {"decision_steps": step, "trace": trace, "last_decision": decision.to_dict()}
        if decision.action in ("finish", "ask_user"):
            valid, feedback = self.checker.check(decision, state.get("observations", []), state["question"])
            if not valid:
                trace.append({"step": step, "stage": "evidence_guard", "result_summary": feedback})
                return {**updated, "trace": trace, "phase": "retry",
                        "feedback": list(state.get("feedback", [])) + [feedback]}
            status = "needs_clarification" if decision.action == "ask_user" else decision.status
            return {**updated, "phase": "end", "status": status, "answer": decision.answer,
                    "citations": self.checker.enrich_citations(decision.citations), "missing_information": decision.missing_information}
        return {**updated, "phase": "tool"}

    def execute_tool_node(self, state: AgentState) -> dict:
        decision = Decision.from_data(state["last_decision"])
        args = decision.arguments
        count = state.get("tool_calls", 0) + 1
        try:
            result = self.tools.call(decision.action, args)
            success = True
            error = None
            if decision.action == "search_documents":
                hits = result.get("hits", [])
                summary = f"命中 {len(hits)} 份文档：" + ", ".join(
                    f"{x['doc_id']}({x['score']:.2f})" for x in hits
                )
            else:
                summary = f"已读取 {result['doc_id']} 全文、版本与适用范围（{len(result['content'])} 字）"
        except Exception as exc:
            result = {}
            success = False
            error = f"{type(exc).__name__}: {str(exc)[:180]}"
            summary = "工具调用失败：" + error
        observation: dict[str, Any] = {
            "tool": decision.action, "arguments": args, "success": success, "result": result,
        }
        if error:
            observation["error"] = error
        trace = list(state.get("trace", []))
        trace.append({"step": state.get("decision_steps", 0), "stage": "tool",
                      "tool": decision.action, "arguments": args,
                      "success": success, "result_summary": summary})
        return {"phase": "retry", "observations": list(state.get("observations", [])) + [observation],
                "trace": trace, "tool_calls": count}

    @staticmethod
    def route(state: AgentState) -> str:
        if state.get("phase") == "end":
            return "end"
        if state.get("phase") == "tool":
            return "tool"
        return "retry"

    def _create_graph(self):
        try:
            from langgraph.graph import StateGraph, START, END
        except ImportError:
            return None
        builder = StateGraph(AgentState)
        builder.add_node("model_decision", self.decision_node)
        builder.add_node("execute_tool", self.execute_tool_node)
        builder.add_edge(START, "model_decision")
        builder.add_conditional_edges("model_decision", self.route,
                                      {"end": END, "tool": "execute_tool", "retry": "model_decision"})
        builder.add_edge("execute_tool", "model_decision")
        return builder.compile()

    def run(self, question: str, history: list[dict] | None = None) -> dict:
        if not isinstance(question, str) or not question.strip() or len(question) > 1000:
            raise ValueError("question must contain 1-1000 characters")
        state: AgentState = {
            "question": question, "history": list(history or []), "observations": [], "feedback": [],
            "trace": [], "decision_steps": 0, "tool_calls": 0,
            "mode": self.model.mode,
        }
        if self.graph is not None:
            result = self.graph.invoke(state, config={"recursion_limit": self.max_steps * 3 + 8})
        else:
            # Pure-Python environment fallback uses identical guard and action nodes.
            for _ in range(self.max_steps * 3 + 8):
                state.update(self.decision_node(state))
                if state["phase"] == "end":
                    break
                if state["phase"] == "tool":
                    state.update(self.execute_tool_node(state))
            else:
                raise RuntimeError("fallback runner exhausted safety bound")
            result = state
        return {k: result.get(k) for k in (
            "mode", "status", "answer", "citations", "missing_information", "trace",
            "decision_steps", "tool_calls"
        )}
