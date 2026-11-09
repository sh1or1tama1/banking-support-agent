"""Fast offline tests (no internet, no API key).   python -m pytest tests -q

They guard the things that would silently invalidate results:
  - no near-duplicate group crosses train/val/test; every intent present in every split
  - the knowledge base contains TRAIN rows only
  - prompts keep {{Placeholders}} and JSON braces intact
  - the agent's loops are bounded and its escalation branches fire
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LLM_PROVIDER", "mock")

from src.data.split import assign_groups, make_split  # noqa: E402
from src.llm.client import LLMResponse  # noqa: E402
from src.llm.prompts import load_prompt  # noqa: E402
from src.rag.agentic_rag import AgenticConfig, AgenticRAG  # noqa: E402
from src.rag.retriever import BM25Retriever, DenseRetriever, build_kb  # noqa: E402


@pytest.fixture(scope="module")
def standin():
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_standin_bitext

    make_standin_bitext.main(n_per_intent=30)
    from src import config as C

    df = pd.read_parquet(C.RAW / "_standin_bitext.parquet")
    df.insert(0, "row_id", [f"bx_{i:05d}" for i in range(len(df))])
    return make_split(assign_groups(df))


def test_groups_never_cross_splits(standin):
    assert (standin.groupby("group_id")["split"].nunique() == 1).all()


def test_every_intent_in_every_split(standin):
    for s in ("train", "val", "test"):
        assert set(standin.loc[standin["split"] == s, "intent"]) == set(standin["intent"])


def test_kb_is_train_only(standin):
    kb = build_kb(standin)
    assert set(kb["doc_id"]) <= set(standin.loc[standin["split"] == "train", "row_id"])


def test_prompt_render_keeps_braces():
    sys_p, user_p = load_prompt("naive_rag_v1").render(query="hi {there}", context="A: call {{Customer Support Phone Number}}")
    assert "{{Customer Support Phone Number}}" in sys_p + user_p
    assert '{"action": "answer" | "clarify" | "escalate"' in sys_p
    assert "hi {there}" in user_p


class ScriptedLLM:
    """Returns canned JSON per step, keyed by a phrase in the system prompt."""
    provider, model = "scripted", "scripted"

    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def chat(self, system, user, max_tokens=None):
        for key, reply in self.replies.items():
            if key in system:
                self.calls.append(key)
                return LLMResponse(json.dumps(reply), 10, 5, 0.0, 0.0, False, "scripted")
        raise AssertionError("unexpected prompt")


ANALYZE = "Return JSON with keys intent"
GRADE = "Return JSON with keys relevant"
REWRITE = "Return JSON with keys query"
VALIDATE = "Return JSON with keys verdict"
GENERATE = "checked for relevance"


def make_agent(standin, replies, **cfg):
    kb = build_kb(standin)
    llm = ScriptedLLM(replies)
    return AgenticRAG(llm, DenseRetriever(kb), BM25Retriever(kb), classifier=lambda q: ("block_card", 0.9),
                      cfg=AgenticConfig(**cfg)), llm


BANKING = {"intent": "block_card", "is_banking": True, "needs_clarification": False, "sub_questions": [], "search_query": "block card"}
ANSWER = {"action": "answer", "answer": "1. Open {{Banking App}} ...", "reason": "x"}


def test_out_of_scope_escalates_without_retrieval(standin):
    agent, llm = make_agent(standin, {ANALYZE: {**BANKING, "is_banking": False}})
    out = agent.answer("what's the weather")
    assert out["action"] == "escalate" and llm.calls == [ANALYZE]


def test_no_relevant_docs_rewrites_then_escalates(standin):
    agent, llm = make_agent(standin, {ANALYZE: BANKING, GRADE: {"relevant": []}, REWRITE: {"query": "x"}})
    out = agent.answer("block my card")
    assert out["action"] == "escalate"
    assert out["retrieval_rounds"] == 2 and llm.calls.count(REWRITE) == 1  # bounded loop


def test_validation_fail_revises_once_then_escalates(standin):
    agent, llm = make_agent(standin, {ANALYZE: BANKING, GRADE: {"relevant": [1]}, GENERATE: ANSWER,
                                      VALIDATE: {"verdict": "fail", "problems": ["invented fee"], "should_escalate": False}})
    out = agent.answer("block my card")
    assert out["action"] == "escalate" and out["revisions"] == 1
    assert llm.calls.count(GENERATE) == 2 and llm.calls.count(VALIDATE) == 2


def test_happy_path_answers(standin):
    agent, llm = make_agent(standin, {ANALYZE: BANKING, GRADE: {"relevant": [1, 2]}, GENERATE: ANSWER,
                                      VALIDATE: {"verdict": "pass", "problems": [], "should_escalate": False}})
    out = agent.answer("block my card")
    assert out["action"] == "answer" and out["n_llm_calls"] == 4 and len(out["contexts"]) == 2


def test_ablation_switches_skip_llm_calls(standin):
    agent, llm = make_agent(standin, {ANALYZE: BANKING, GENERATE: ANSWER}, use_grading=False, use_validation=False)
    out = agent.answer("block my card")
    assert out["action"] == "answer" and llm.calls == [ANALYZE, GENERATE]
