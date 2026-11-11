"""python -m app.cli --help"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from app.service import KnowledgeAgent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="企业知识问答 Agent（真实 LLM / 明确标注的离线演示）")
    parser.add_argument("--mode", choices=["demo", "llm"], default="demo")
    parser.add_argument("--ask", type=str, help="提问（不指定则进入交互模式）")
    parser.add_argument("--followup", type=str, help="本次提问后的同会话追问")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--fail-tool", choices=["search_documents", "read_document"], help="模拟工具失败")
    parser.add_argument("--json", action="store_true", help="打印结构化 JSON 和 Trace")
    parser.add_argument("--trace-out", type=Path, help="把 JSON/Trace 写到文件")
    parser.add_argument("--data-dir", type=Path, help="自定义 catalog.json + docs/ 目录")
    parser.add_argument("--no-langgraph", action="store_true", help="显式使用同节点纯 Python 驱动器")
    args = parser.parse_args(argv)
    kwargs = {"mode": args.mode, "max_steps": args.max_steps,
              "fail_tools": {args.fail_tool} if args.fail_tool else set(),
              "prefer_langgraph": not args.no_langgraph}
    if args.data_dir:
        kwargs["data_dir"] = args.data_dir
    agent = KnowledgeAgent(**kwargs)
    if args.mode == "demo":
        print("[离线演示模式：规则策略模拟动作，不代表真实 LLM 决策效果]", file=sys.stderr)
    answers = []

    def run_question(question: str):
        result = agent.chat(question)
        answers.append({"question": question, **result})
        if args.json:
            print(json.dumps(answers[-1], ensure_ascii=False, indent=2))
        else:
            print(f"\n状态: {result['status']} | 证据: {len(result['citations'])} | 工具调用: {result['tool_calls']}")
            print("回答:", result["answer"])
            for cite in result["citations"]:
                print(f"引用 [{cite['doc_id']}]: {cite['quote']}")
            print("决策路径:", " -> ".join(x["action"]
                                           for x in result["trace"] if x["stage"] == "decision"))

    if args.ask:
        run_question(args.ask)
        if args.followup:
            run_question(args.followup)
    else:
        print("输入问题，输入 /exit 退出")
        for line in sys.stdin:
            question = line.strip()
            if question in {"/exit", "/quit"}:
                break
            if question:
                run_question(question)
    if args.trace_out:
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        args.trace_out.write_text(json.dumps(answers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Trace saved: {args.trace_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
