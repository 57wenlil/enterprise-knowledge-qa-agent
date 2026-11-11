from pathlib import Path
import pytest
from app.service import KnowledgeAgent


@pytest.fixture
def agent():
    return KnowledgeAgent(mode="demo", prefer_langgraph=False, as_of="2026-11-11")


@pytest.fixture
def repo(agent):
    return agent.repository


def path_to_data():
    return Path(__file__).resolve().parents[1] / "data"
