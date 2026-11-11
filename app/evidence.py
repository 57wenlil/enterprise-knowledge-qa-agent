"""Deterministic citation and conflict guard; it audits, not replaces, model planning."""
from __future__ import annotations
from collections import defaultdict
import re

from app.retrieval.repository import DocumentRepository
from app.llm.decision import Decision


class EvidenceChecker:
    def __init__(self, repository: DocumentRepository):
        self.repo = repository

    @staticmethod
    def hits(observations: list[dict]) -> list[dict]:
        result: dict[str, dict] = {}
        for obs in observations:
            if obs.get("tool") == "search_documents" and obs.get("success"):
                for hit in obs.get("result", {}).get("hits", []):
                    result[hit["doc_id"]] = hit
        return list(result.values())

    @staticmethod
    def read_documents(observations: list[dict]) -> dict[str, dict]:
        return {
            obs["result"]["doc_id"]: obs["result"]
            for obs in observations
            if obs.get("tool") == "read_document" and obs.get("success") and "doc_id" in obs.get("result", {})
        }

    def unresolved_conflicts(self, observations: list[dict], question: str) -> list[dict]:
        """Conservative: no implicit 'latest wins' for conflicting active policies.

        If the user explicitly requests comparison of a single doc version (D01 or
        D02), answer must be qualified as doc-specific; we allow that use case.
        """
        if re.search(r"(?<![A-Za-z0-9])D\d{2}(?!\d)", question, re.I):
            return []  # user explicitly scoped a document ID; don't assert it is authoritative
        grouped: dict[str, list[dict]] = defaultdict(list)
        for item in self.hits(observations):
            if item.get("claim_key") and item.get("claim_value"):
                grouped[item["claim_key"]].append(item)
        conflicts = []
        for group in grouped.values():
            if len({it["claim_value"] for it in group}) < 2:
                continue
            # Same declared scope and no explicit priority. Version date alone is not priority.
            same_scope = len({it["scope"] for it in group}) == 1
            if same_scope and len({it.get("authority") for it in group}) != 1:
                # Different authorities without a known priority are ambiguous as well.
                pass
            if same_scope:
                conflicts.extend(group)
        return sorted(conflicts, key=lambda x: x["doc_id"])

    def enrich_citations(self, citations: list[dict]) -> list[dict]:
        """Metadata is taken from source of truth, not free-form LLM text."""
        enriched = []
        for item in citations:
            doc = self.repo.get(item["doc_id"])
            quote = item["quote"]
            containing = next((c for c in doc.chunks if quote in c.text), None)
            enriched.append({
                "doc_id": doc.doc_id, "title": doc.title, "version": doc.version,
                "version_date": doc.version_date, "source": doc.source,
                "chunk_id": containing.chunk_id if containing else None,
                "quote": quote,
            })
        return enriched

    def check(self, decision: Decision, observations: list[dict], question: str) -> tuple[bool, str]:
        if decision.action not in ("finish", "ask_user"):
            return True, ""
        read = self.read_documents(observations)
        citations = decision.citations
        for citation in citations:
            doc_id, quote = citation["doc_id"], citation["quote"]
            if doc_id not in read:
                return False, f"引用 {doc_id} 尚未通过 read_document 读取原文"
            if not quote.strip() or len(quote) > 600 or quote not in read[doc_id]["content"]:
                return False, f"引用 {doc_id} 的原文不匹配（必须逐字复制）"
        conflict = self.unresolved_conflicts(observations, question)
        if conflict:
            conflict_ids = {c["doc_id"] for c in conflict}
            if decision.action == "finish" and decision.status == "completed":
                return False, f"检索发现适用范围相同但规定冲突的 {sorted(conflict_ids)}，不能选择单一结论；先读原文并 ask_user"
            if decision.action == "finish" and decision.status == "insufficient_evidence":
                if not conflict_ids.issubset(read) or not conflict_ids.issubset(
                    {c["doc_id"] for c in citations}
                ) or "冲突" not in decision.answer:
                    return False, "已发现制度冲突，不能跳过原文核验与冲突引用直接笼统宣称无资料"
            if decision.action == "ask_user":
                if not conflict_ids.issubset(read):
                    return False, f"缺少冲突资料原文：{sorted(conflict_ids - read.keys())}"
                if not conflict_ids.issubset({c["doc_id"] for c in citations}):
                    return False, "冲突说明需要引用每份相互矛盾的文档"
        if decision.action == "finish" and decision.status == "completed":
            if not any(x.get("tool") == "search_documents" and x.get("success") for x in observations):
                return False, "完整答案需要先调用 search_documents"
            if not citations:
                return False, "完整答案必须包含原文引用"
            # Questions about a duration require a quoted duration, not a doc title.
            cited_text = "\n".join(c["quote"] for c in citations)
            if any(word in question for word in ("期限", "多少天", "几天", "多久", "几日")):
                if not re.search(r"\d+\s*(?:个)?(?:自然日|工作日|天|小时)", cited_text):
                    return False, "该问题询问时长，但所引原文没有时间数量依据"
            # Prevent obvious invented quantities even when a quote is otherwise valid.
            for num, unit in re.findall(r"(?<!\d)(\d+)\s*(?:个)?(自然日|工作日|天|小时|年)", decision.answer):
                variants = [f"{num}{unit}", f"{num}个{unit}"]
                if not any(x in cited_text for x in variants):
                    return False, f"答案中的数量 {num}{unit} 未被引用原文支持"
        return True, ""
