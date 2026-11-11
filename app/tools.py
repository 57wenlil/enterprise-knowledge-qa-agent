"""Explicit read/search tool boundary; failures are observable, not swallowed."""
from __future__ import annotations
from app.retrieval.repository import DocumentRepository
from app.retrieval.search import LocalRetriever


class KnowledgeTools:
    def __init__(self, repo: DocumentRepository, *, fail_tools: set[str] | None = None):
        self.repo = repo
        self.retriever = LocalRetriever(repo)
        self.fail_tools = fail_tools or set()

    def call(self, name: str, arguments: dict) -> dict:
        if name in self.fail_tools:
            raise RuntimeError(f"模拟 {name} 工具不可用")
        if name == "search_documents":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip() or len(query) > 400:
                raise ValueError("search_documents requires nonempty query of <=400 characters")
            k = arguments.get("top_k", 5)
            if not isinstance(k, int):
                raise ValueError("top_k must be integer")
            # Always include conflicting candidates from the same claim family,
            # even if an LLM asks for top_k=1. A narrow top_k must not hide
            # an unresolved 30-day vs 15-day policy conflict.
            ranked = self.retriever.search(query, top_k=max(k, 8))
            first = ranked[:max(1, min(k, 8))]
            selected_ids = {hit["doc_id"] for hit in first}
            family_keys = {hit["claim_key"] for hit in first if hit.get("claim_key")}
            peers = [hit for hit in ranked if hit["doc_id"] not in selected_ids
                     and hit.get("claim_key") in family_keys]
            return {"query": query, "hits": (first + peers)[:8]}
        if name == "read_document":
            doc_id = arguments.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id.startswith("D") or len(doc_id) > 32:
                raise ValueError("read_document requires valid doc_id")
            return self.repo.read(doc_id)
        raise ValueError("unsupported tool: " + str(name))
