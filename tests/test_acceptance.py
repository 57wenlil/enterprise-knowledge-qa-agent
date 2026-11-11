"""The SIX requested acceptance scenarios use the real repository/tools/guard.

Offline action strategy is a test double, not model-quality evidence.
"""
from app.service import KnowledgeAgent


def actions(result):
    return [x["action"] for x in result["trace"] if x["stage"] == "decision"]


def test_01_clear_answer(agent, repo):
    r = agent.chat("年假有几天？")
    assert r["status"] == "completed"
    assert "5天" in r["answer"]
    assert actions(r) == ["search_documents", "read_document", "finish"]
    assert r["citations"][0]["doc_id"] == "D03"
    assert r["citations"][0]["quote"] in repo.get("D03").text
    assert r["citations"][0]["version_date"] == "2026-03-05"


def test_02_conflicting_policies_requires_clarification(agent, repo):
    r = agent.chat("差旅报销期限是多少天？")
    assert r["status"] == "needs_clarification"
    assert "30天" in r["answer"] and "15天" in r["answer"]
    assert "确认" in r["answer"]
    assert actions(r) == ["search_documents", "read_document", "read_document", "ask_user"]
    assert {c["doc_id"] for c in r["citations"]} == {"D01", "D02"}
    for citation in r["citations"]:
        assert citation["quote"] in repo.get(citation["doc_id"]).text
    assert r["missing_information"]


def test_03_out_of_scope(agent):
    r = agent.chat("明天哪只股票涨得最多？")
    assert r["status"] == "insufficient_evidence"
    assert "无法确认" in r["answer"]
    assert r["citations"] == []
    assert actions(r) == ["search_documents", "finish"]


def test_04_paraphrased_question(agent):
    r = agent.chat("员工入职一年可以休多少天带薪假？")
    assert r["status"] == "completed"
    assert "5天" in r["answer"]
    assert r["citations"][0]["doc_id"] == "D03"


def test_05_context_followup(agent):
    first = agent.chat("年假有几天？")
    second = agent.chat("那要提前多久申请？")
    assert first["status"] == second["status"] == "completed"
    assert "提前3个工作日" in second["answer"]
    assert "D03" == second["citations"][0]["doc_id"]
    searched = [t for t in second["trace"] if t.get("tool") == "search_documents"]
    assert searched and "年假" in searched[0]["arguments"]["query"]


def test_06_tool_failure_returns_no_fabrication():
    agent = KnowledgeAgent(mode="demo", fail_tools={"search_documents"},
                           prefer_langgraph=False)
    r = agent.chat("年假有几天？")
    assert r["status"] == "error"
    assert r["citations"] == []
    assert "失败" in r["answer"]
    assert any(t["stage"] == "tool" and not t["success"] for t in r["trace"])


def test_additional_tool_read_failure():
    agent = KnowledgeAgent(mode="demo", fail_tools={"read_document"},
                           prefer_langgraph=False)
    r = agent.chat("年假有几天？")
    assert r["status"] == "error" and r["citations"] == []


def test_specified_doc_makes_a_qualified_answer(agent):
    r = agent.chat("根据D01，报销期限是多少天？")
    assert r["status"] == "completed"
    assert "30个自然日" in r["answer"]
    assert "不代表其优先于其他制度" in r["answer"]
    assert {c["doc_id"] for c in r["citations"]} == {"D01"}
