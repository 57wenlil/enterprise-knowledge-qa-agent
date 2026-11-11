"""Local hybrid-style lexical retriever with stable source IDs.

Adaptation of the tokenize/lexical_score/KnowledgeBase.search approach from
用户提供的物业 customer_service_agent `app/rag/knowledge_base.py`.
Adds Unicode-aware Chinese n-gram query scoring, IDF, snippet offsets,
original-document pointers and no out-of-scope loose fallback.

The heavy BGE-M3 + Milvus dense/sparse/rerank architecture of the supplied
高校 RAG is documented as an optional production replacement, NOT claimed to
be running in this minimal submission.
"""
from __future__ import annotations
from collections import Counter
import math
import re

from app.retrieval.repository import DocumentRepository

SYNONYMS = {
    "出差": "差旅", "报账": "报销", "费用申报": "报销", "最迟": "时限",
    "最晚": "时限", "多久": "多少天", "何时": "什么时候", "年休": "年假",
    "带薪休假": "年假", "年休假": "年假", "电脑坏了": "设备故障", "带薪假": "年假", "休年假": "年假", "休假": "年假",
    "笔记本": "电脑", "设备维修": "设备故障", "修电脑": "电脑报修", "休多少天": "几天",
    "递交": "提交", "流程": "申请流程", "审批": "批准",
}
FILLERS = ("请问", "想问一下", "麻烦", "您好", "谢谢", "一下", "员工", "公司", "规定", "请告诉我")


def normalize(text: str) -> str:
    text = text.lower()
    for a, b in sorted(SYNONYMS.items(), key=lambda x: -len(x[0])):
        text = text.replace(a, b)
    for filler in FILLERS:
        text = text.replace(filler, "")
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)


def tokenize(text: str) -> set[str]:
    """CJK bigrams/trigrams plus alphabetic words; avoids single-char false positives."""
    normalized = normalize(text)
    tokens = set(re.findall(r"[a-z_]+|\d+", normalized))
    for cn in re.findall(r"[\u4e00-\u9fff]+", normalized):
        tokens.update(cn[i:i + 2] for i in range(len(cn) - 1))
        tokens.update(cn[i:i + 3] for i in range(len(cn) - 2))
        if len(cn) == 1:
            tokens.add(cn)
    return tokens


class LocalRetriever:
    def __init__(self, repo: DocumentRepository):
        self.repo = repo
        self.chunk_bags = {}
        self.doc_freq: Counter[str] = Counter()
        for doc in repo.documents.values():
            for chunk in doc.chunks:
                bag = tokenize(chunk.text + " " + doc.title + " " + doc.topic)
                self.chunk_bags[chunk.chunk_id] = bag
                self.doc_freq.update(bag)
        self.N = max(1, len(self.chunk_bags))

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip():
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        hits = []
        for doc in self.repo.documents.values():
            for chunk in doc.chunks:
                bag = self.chunk_bags[chunk.chunk_id]
                overlap = query_tokens & bag
                if not overlap:
                    continue
                weight = lambda tok: math.log(1 + (self.N + 1) / (1 + self.doc_freq[tok]))
                numerator = sum(weight(t) for t in overlap)
                denominator = sum(weight(t) for t in query_tokens)
                score = numerator / max(denominator, 1e-8)
                # Topic/title matches protect short questions (e.g. 年假有几天).
                norm_query = normalize(query)
                norm_topic = normalize(doc.topic)
                topic_bag = tokenize(doc.title + " " + doc.topic)
                if norm_topic and norm_topic in norm_query and len(norm_topic) >= 2:
                    score += 0.30
                elif query_tokens & topic_bag:
                    score += 0.12
                # Avoid one accidental bigram causing an out-of-scope answer.
                if len(overlap) < 2 and score < 0.22:
                    continue
                if score < 0.18:
                    continue
                hits.append({
                    "doc_id": doc.doc_id, "title": doc.title, "version": doc.version,
                    "version_date": doc.version_date, "scope": doc.scope,
                    "claim_key": doc.claim_key, "claim_value": doc.claim_value,
                    "authority": doc.authority,
                    "chunk_id": chunk.chunk_id, "parent_id": chunk.parent_id,
                    "snippet": chunk.text[:260], "score": round(min(score, 1.0), 4),
                })
        # Only highest-scoring chunk per doc; deterministic tie breaking.
        best_by_doc: dict[str, dict] = {}
        for item in hits:
            old = best_by_doc.get(item["doc_id"])
            if old is None or item["score"] > old["score"]:
                best_by_doc[item["doc_id"]] = item
        ranked = sorted(best_by_doc.values(), key=lambda d: (-d["score"], d["doc_id"]))
        return ranked[:max(1, min(int(top_k), 8))]
