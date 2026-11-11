import pytest
from app.llm.decision import Decision, InvalidDecision
from app.retrieval.chunker import chunk_document
from app.service import KnowledgeAgent
from app.tools import KnowledgeTools


class ScriptedModel:
    """Each decision is supplied at test time, proving the runner uses model output."""
    mode = "scripted"

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.views = []

    def decide(self, view):
        self.views.append(view)
        return self.decisions.pop(0) if self.decisions else {
            "action": "finish", "status": "insufficient_evidence", "answer": "无充足证据",
        }


def test_requires_read_before_final_and_replans_after_guard(repo):
    original = repo.get("D03").text
    quote = next(x for x in original.splitlines() if "第一条" in x).strip()
    fake = ScriptedModel([
        {"action": "search_documents", "arguments": {"query": "年假"}},
        {"action": "finish", "answer": "每年5天年假", "citations": [{"doc_id": "D03", "quote": quote}]},
        {"action": "read_document", "arguments": {"doc_id": "D03"}},
        {"action": "finish", "answer": "每年5天年假", "citations": [{"doc_id": "D03", "quote": quote}]},
    ])
    agent = KnowledgeAgent(model=fake, prefer_langgraph=False)
    result = agent.chat("年假有几天？")
    assert result["status"] == "completed"
    assert any(t["stage"] == "evidence_guard" for t in result["trace"])
    assert fake.views[2]["feedback"]  # validator feedback returned for next model decision


def test_prevents_unfounded_quantity(repo):
    quote = next(x for x in repo.get("D03").text.splitlines() if "第一条" in x)
    fake = ScriptedModel([
        {"action": "search_documents", "arguments": {"query": "年假"}},
        {"action": "read_document", "arguments": {"doc_id": "D03"}},
        {"action": "finish", "answer": "年假每年有99天", "citations": [{"doc_id": "D03", "quote": quote}]},
    ])
    r = KnowledgeAgent(model=fake, max_steps=3, prefer_langgraph=False).chat("年假有几天？")
    assert r["status"] == "incomplete"
    assert any(t["stage"] == "evidence_guard" for t in r["trace"])


def test_rejects_forged_quote_even_when_doc_id_exists():
    fake = ScriptedModel([
        {"action": "search_documents", "arguments": {"query": "年假"}},
        {"action": "read_document", "arguments": {"doc_id": "D03"}},
        {"action": "finish", "answer": "每年5天", "citations": [
            {"doc_id": "D03", "quote": "公司每年给予100天假期"}
        ]},
    ])
    r = KnowledgeAgent(model=fake, max_steps=3, prefer_langgraph=False).chat("年假有几天？")
    assert r["status"] == "incomplete"
    assert not r["citations"]


def test_conflict_guard_blocks_attempt_to_pick_single_doc():
    fake = ScriptedModel([
        {"action": "search_documents", "arguments": {"query": "差旅报销期限"}},
        {"action": "read_document", "arguments": {"doc_id": "D01"}},
        {"action": "finish", "answer": "期限是30天", "citations": [{"doc_id": "D01", "quote":
            "第二条 提交时限：员工应在差旅结束之日起30个自然日内提交报销申请，附完整发票与行程凭证"}]},
    ])
    result = KnowledgeAgent(model=fake, max_steps=3, prefer_langgraph=False).chat("报销期限是多少天？")
    assert result["status"] == "incomplete"
    assert any(t["stage"] == "evidence_guard" and "冲突" in t["result_summary"] for t in result["trace"])


def test_max_steps_guard_stops_repeated_searches():
    fake = ScriptedModel([{"action": "search_documents", "arguments": {"query": "年假"}}] * 12)
    r = KnowledgeAgent(model=fake, max_steps=3, prefer_langgraph=False).chat("年假几天？")
    assert r["status"] == "incomplete"
    assert r["tool_calls"] == 3
    assert r["decision_steps"] == 4


def test_invalid_model_action_is_error():
    fake = ScriptedModel([{"action": "exec_shell", "arguments": {"cmd": "pwd"}}])
    r = KnowledgeAgent(model=fake, prefer_langgraph=False).chat("年假几天？")
    assert r["status"] == "error"
    assert not r["citations"]


def test_document_read_rejects_unknown_ids(repo):
    tool = KnowledgeTools(repo)
    with pytest.raises(KeyError):
        tool.call("read_document", {"doc_id": "D999"})
    with pytest.raises(ValueError):
        tool.call("read_document", {"doc_id": "../../.env"})


def test_chunk_stable_offsets():
    text = ("一个句子。" * 90) + ("第二个段落。" * 80)
    chunks = chunk_document("D99", text, parent_size=300, child_size=120, overlap=15)
    assert len(chunks) > 5
    assert len({x.chunk_id for x in chunks}) == len(chunks)
    assert all(text[x.start:x.end] == x.text for x in chunks)
    assert [x.chunk_id for x in chunks] == [x.chunk_id for x in chunk_document(
        "D99", text, parent_size=300, child_size=120, overlap=15)]


def test_invalid_decision_protocol():
    with pytest.raises(InvalidDecision):
        Decision.from_data({"action": "unknown"})
    with pytest.raises(InvalidDecision):
        Decision.from_data({"action": "read_document", "arguments": []})


def test_retriever_includes_conflict_peers_even_if_top_k_one(repo):
    hits = KnowledgeTools(repo).call("search_documents", {"query": "差旅报销期限", "top_k": 1})["hits"]
    assert {x["doc_id"] for x in hits} == {"D01", "D02"}


def test_cannot_hide_known_conflict_as_generic_no_data():
    fake = ScriptedModel([
        {"action": "search_documents", "arguments": {"query": "差旅报销期限", "top_k": 1}},
        {"action": "finish", "status": "insufficient_evidence", "answer": "无资料", "citations": []},
    ])
    r = KnowledgeAgent(model=fake, max_steps=2, prefer_langgraph=False).chat("差旅报销期限？")
    assert r["status"] == "incomplete"
    assert any(t["stage"] == "evidence_guard" for t in r["trace"])
