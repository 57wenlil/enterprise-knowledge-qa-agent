"""OFFLINE DEMO POLICY ONLY: deterministic test double, NOT an LLM/ReAct claim.

Used to verify tools, trace, citations, conflict guard and packaging without an
API key. For actual model-selected actions, use --mode llm.
"""
from __future__ import annotations
import re

from app.retrieval.search import normalize, tokenize


def _known_history_topic(history: list[dict]) -> str:
    """Just enough history resolution to exercise a follow-up in offline mode."""
    for message in reversed(history):
        if message.get("role") == "user":
            text = message.get("content", "")
            if any(w in text for w in ("年假", "年休", "休假", "带薪")):
                return "年假"
            if any(w in text for w in ("出差", "差旅", "报销")):
                return "差旅报销"
            if any(w in text for w in ("电脑", "设备", "报修")):
                return "设备报修"
    return ""


def _best_sentence(content: str, question: str) -> str:
    lines = [line.strip() for line in re.split(r"[\n。]+", content) if line.strip()]
    if not lines:
        return ""
    q_tokens = tokenize(question)
    is_advance = any(kw in question for kw in ("提前", "流程", "怎么申请", "如何申请"))
    is_duration = any(kw in question for kw in ("几天", "多少天", "有多久", "几日")) and not is_advance
    def score(line: str) -> float:
        overlap = len(q_tokens & tokenize(line))
        bonus = 0
        if is_advance and "提前" in line and "申请" in line:
            bonus += 12
        if is_duration and ("休假天数" in line or "5天" in line):
            bonus += 9
        if any(kw in question for kw in ("报销", "差旅", "D01", "D02")) and "提交时限" in line:
            bonus += 12
        if ("维修" in question or "报修" in question or "电脑" in question) and "工单" in line:
            bonus += 5
        if ("逾期" in question or "过期" in question) and "逾期" in line:
            bonus += 7
        if "密码" in question and "密码" in line:
            bonus += 7
        if len(line) <= 3:
            bonus -= 2
        return bonus + overlap / max(1, len(q_tokens)) - len(line) / 5000
    return max(lines, key=score)


class DemoDecisionModel:
    mode = "demo"

    def decide(self, view: dict) -> dict:
        question = view["question"]
        obs = view.get("observations", [])
        history = view.get("history", [])
        base_query = question
        # Contextual follow-up: previous user question anchors elided subject.
        if any(x in question for x in ("那", "这个", "它", "上述", "刚才", "需要提前")):
            topic = _known_history_topic(history)
            if topic and topic not in question:
                base_query = topic + " " + question
        for item in reversed(obs):
            if item.get("success") is False and item.get("tool") in ("search_documents", "read_document"):
                return {
                    "action": "finish", "status": "error", "answer": "知识库工具执行失败，无法核对原文；本轮不提供未经证实的答案。",
                    "missing_information": [item.get("error", "工具失败")],
                    "reason_summary": "观察到工具失败；停止，避免编造答案。",
                }
        searched = [o for o in obs if o.get("tool") == "search_documents" and o.get("success")]
        if not searched:
            return {"action": "search_documents", "arguments": {"query": base_query, "top_k": 6},
                    "reason_summary": "尚未获取证据，先检索候选资料。"}
        hits = []
        hit_ids = set()
        for o in searched:
            for h in o["result"].get("hits", []):
                if h["doc_id"] not in hit_ids:
                    hits.append(h)
                    hit_ids.add(h["doc_id"])
        if not hits:
            return {"action": "finish", "status": "insufficient_evidence",
                    "answer": "当前提供的合成制度和 FAQ 中未找到足以回答该问题的资料，无法确认。",
                    "missing_information": ["需要提供与问题有关的可靠制度或 FAQ"],
                    "reason_summary": "检索未命中可用文档，不能猜测。"}
        explicit_doc = re.search(r"(?<![A-Za-z0-9])(D\d{2})(?!\d)", question, re.I)
        selected_id = explicit_doc.group(1).upper() if explicit_doc else ""
        if selected_id:
            selected = next((h for h in hits if h["doc_id"] == selected_id), None)
            if not selected:
                return {"action": "finish", "status": "insufficient_evidence",
                        "answer": f"未找到你指定的 {selected_id} 的可核验资料。",
                        "reason_summary": "用户指定的文档未出现在检索命中中。"}
            hits = [selected]
        conflict_groups: dict[str, list[dict]] = {}
        for hit in hits:
            conflict_groups.setdefault(hit["claim_key"], []).append(hit)
        conflicting = [group for group in conflict_groups.values()
                       if len({h["claim_value"] for h in group}) > 1
                       and len({h["scope"] for h in group}) == 1]
        conflict = sorted(conflicting[0], key=lambda h: h["doc_id"]) if conflicting else []
        readings = {o["result"]["doc_id"]: o["result"] for o in obs
                    if o.get("tool") == "read_document" and o.get("success")}
        if conflict:
            for hit in conflict:
                if hit["doc_id"] not in readings:
                    return {"action": "read_document", "arguments": {"doc_id": hit["doc_id"]},
                            "reason_summary": "已发现不同报销期限，须逐份读取原文核查适用范围和优先级。"}
            cites = []
            for hit in conflict:
                source = readings[hit["doc_id"]]
                sentence = next((line.strip() for line in re.split(r"[。\n]+", source["content"])
                                 if "提交时限" in line and hit["claim_value"].rstrip("天") in line), "")
                if not sentence:
                    sentence = _best_sentence(source["content"], "提交时限")
                cites.append({"doc_id": hit["doc_id"], "quote": sentence})
            summary = "；".join(f"{h['doc_id']} 规定 {h['claim_value']}" for h in conflict)
            return {
                "action": "ask_user", "reason_summary": "原文确认存在同范围制度冲突，未找到可确定优先级的规则。",
                "answer": f"目前无法直接给出唯一的报销期限：{summary}。两份资料均未提供可确认的适用优先级，请确认应以哪个制度为准，或提供制度优先级/适用部门信息。",
                "citations": cites,
                "missing_information": ["适用制度或制度优先级的正式确认"],
            }
        top = hits[0]
        if top["doc_id"] not in readings:
            return {"action": "read_document", "arguments": {"doc_id": top["doc_id"]},
                    "reason_summary": f"检索命中 {top['doc_id']}，先读取原文核实答案和文档版本。"}
        source = readings[top["doc_id"]]
        quote = _best_sentence(source["content"], base_query)
        prefix = f"仅按你指定的 {selected_id}（不代表其优先于其他制度）" if selected_id else f"根据 {top['doc_id']}《{source['title']}》"
        return {
            "action": "finish", "status": "completed", "reason_summary": "已核对原文，可给出带引用的回答。",
            "answer": f"{prefix}：{quote}。",
            "citations": [{"doc_id": top["doc_id"], "quote": quote}],
        }
