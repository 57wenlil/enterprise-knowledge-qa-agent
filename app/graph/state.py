"""Adapted state from 物业客服 Agent's app/graph/state.py.

Separate observation, decisions and trace: persistent across tool cycles; each
turn is initialized independently and can consume prior chat turns as context.
"""
from __future__ import annotations
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    observations: list[dict[str, Any]]
    feedback: list[str]
    last_decision: dict[str, Any]
    trace: list[dict[str, Any]]
    decision_steps: int
    tool_calls: int
    phase: str
    status: str
    answer: str
    citations: list[dict[str, str]]
    missing_information: list[str]
    mode: str
