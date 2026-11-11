"""Validated model-decided actions (not deterministic branch masquerading as ReAct)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

VALID_ACTIONS = {"search_documents", "read_document", "ask_user", "finish"}
VALID_FINISH_STATUSES = {"completed", "insufficient_evidence", "error"}


class InvalidDecision(ValueError):
    pass


@dataclass
class Decision:
    action: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason_summary: str = ""
    answer: str = ""
    citations: list[dict[str, str]] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    status: str = "completed"

    @classmethod
    def from_data(cls, data: Any) -> "Decision":
        if not isinstance(data, dict):
            raise InvalidDecision("model response must be a JSON object")
        action = data.get("action", "")
        if action not in VALID_ACTIONS:
            raise InvalidDecision(f"unknown action: {action!r}")
        args = data.get("arguments", {})
        if not isinstance(args, dict):
            raise InvalidDecision("arguments must be an object")
        citations = data.get("citations", [])
        if not isinstance(citations, list) or len(citations) > 10:
            raise InvalidDecision("citations must be a list with <= 10 elements")
        for item in citations:
            if not isinstance(item, dict) or not isinstance(item.get("doc_id"), str) or not isinstance(item.get("quote"), str):
                raise InvalidDecision("citations must contain doc_id and quote")
        missing = data.get("missing_information")

        if missing is None:
            missing = []
        elif isinstance(missing, str):
            missing = [missing] if missing.strip() else []
        elif not isinstance(missing, list) or not all(isinstance(x, str) for x in missing):
            raise InvalidDecision(
                "missing_information must be a string list"
            )
        status = data.get("status", "completed")
        if action == "finish" and status not in VALID_FINISH_STATUSES:
            raise InvalidDecision("invalid finish status")
        answer = data.get("answer", "")
        if not isinstance(answer, str) or len(answer) > 4000:
            raise InvalidDecision("answer must be text within 4000 characters")
        return cls(
            action=action, arguments=args, reason_summary=str(data.get("reason_summary", ""))[:250],
            answer=answer, citations=citations,
            missing_information=missing[:10], status=status,
        )

    def to_dict(self) -> dict:
        return {
            "action": self.action, "arguments": self.arguments, "reason_summary": self.reason_summary,
            "answer": self.answer, "citations": self.citations,
            "missing_information": self.missing_information, "status": self.status,
        }
