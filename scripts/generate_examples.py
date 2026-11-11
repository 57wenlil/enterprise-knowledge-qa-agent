"""Create actual structured outputs and trace examples from the offline test strategy.

Run: python -m scripts.generate_examples
These are NOT traces from a remote LLM and are marked accordingly.
"""
from __future__ import annotations
from pathlib import Path
from datetime import date
import json

from app.service import KnowledgeAgent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples"


def _save(name: str, payload):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run():
    cases = []
    tasks = [
        ("01_clear_answer", "年假有几天？", None),
        ("02_conflict", "差旅报销期限是多少天？", None),
        ("03_out_of_scope", "明天哪只股票涨得最多？", None),
        ("04_paraphrase", "员工入职一年可以休多少天带薪假？", None),
        ("05_context_followup", "年假有几天？", "那要提前多久申请？"),
        ("06_tool_failure", "年假有几天？", None),
    ]
    for tag, question, followup in tasks:
        agent = KnowledgeAgent(mode="demo", prefer_langgraph=False,
                               as_of="2026-11-11",
                               fail_tools={"search_documents"} if tag == "06_tool_failure" else set())
        first = agent.chat(question)
        payload = {"case": tag, "mode": "demo", "question": question,
                   "result": first, "truth": "合成制度，离线动作策略"}
        if followup:
            payload["followup_question"] = followup
            payload["followup_result"] = agent.chat(followup)
        cases.append(payload)
    _save("acceptance_cases.json", cases)
    _save("trace_clear.json", cases[0])
    _save("trace_conflict.json", cases[1])
    _save("trace_context.json", cases[4])
    _save("trace_tool_failure.json", cases[5])
    for case in cases:
        result = case.get("followup_result", case["result"])
        print(f"{case['case']}: {result['status']} | actions={result['decision_steps']} "
              f"| tools={result['tool_calls']} | cites={len(result['citations'])}")


if __name__ == "__main__":
    run()
