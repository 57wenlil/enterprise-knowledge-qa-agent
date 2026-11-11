from app.llm.client import LLMDecisionModel, ModelCallError, OpenAICompatibleLLM
from app.llm.decision import Decision
import pytest


def test_model_requires_key_without_pretending_demo(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client = OpenAICompatibleLLM(api_key="")
    with pytest.raises(ModelCallError, match="LLM_API_KEY"):
        client.json("system", "user")


class StubClient:
    def __init__(self):
        self.calls = []

    def json(self, system, user):
        self.calls.append((system, user))
        return {"action": "read_document", "arguments": {"doc_id": "D03"},
                "reason_summary": "工具已检索到 D03，需要核对原文"}


def test_real_model_interface_controls_next_action():
    client = StubClient()
    model = LLMDecisionModel(client=client)
    choice = Decision.from_data(model.decide({"question": "年假几天", "observations": []}))
    assert choice.action == "read_document"
    assert choice.arguments == {"doc_id": "D03"}
    assert "年假几天" in client.calls[0][1]
