"""Entry point for interactive chat and multi-turn follow-ups."""
from __future__ import annotations
from pathlib import Path

from app.graph.main_graph import AgentRuntime
from app.llm.client import LLMDecisionModel
from app.llm.demo import DemoDecisionModel
from app.retrieval.repository import DocumentRepository, DEFAULT_DATA_DIR
from app.tools import KnowledgeTools


class KnowledgeAgent:
    def __init__(self, *, mode: str = "demo", data_dir: Path | str = DEFAULT_DATA_DIR,
                 max_steps: int = 8, fail_tools: set[str] | None = None,
                 model=None, prefer_langgraph: bool = True, as_of: str | None = None):
        if mode not in {"demo", "llm"}:
            raise ValueError("mode must be 'demo' or 'llm'")
        self.repository = DocumentRepository(data_dir, as_of=as_of)
        self.tools = KnowledgeTools(self.repository, fail_tools=fail_tools)
        self.model = model or (DemoDecisionModel() if mode == "demo" else LLMDecisionModel())
        self.runtime = AgentRuntime(self.model, self.tools, max_steps=max_steps,
                                    prefer_langgraph=prefer_langgraph)
        self.history: list[dict[str, str]] = []

    def chat(self, question: str) -> dict:
        response = self.runtime.run(question, self.history)
        self.history.extend([{"role": "user", "content": question},
                             {"role": "assistant", "content": response["answer"] or ""}])
        self.history = self.history[-12:]
        return response
