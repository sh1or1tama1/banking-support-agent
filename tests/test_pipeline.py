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


def test_auto_checks_ignore_customer_amounts():
    from src.eval.harness import auto_checks
    d = pd.DataFrame({"query": ["I asked for $100 but only got $20", "hi"],
                      "answer": ["Sorry you got $20 instead of $100.", "The fee is $5, call 1-800-555-1234"]})
    out = auto_checks(d)
    assert out["invented_any_specific"].tolist() == [False, True]


def test_disagreements_and_merge_final():
    from src.eval.judge import ITEMS, disagreements, merge_final
    base = {"eval_id": ["e1", "e1"], "blind_system": ["S1", "S2"], "query": ["q", "q"], "answer": ["a", "b"]}
    a = pd.DataFrame({**base, **{k: [1, 1] for k in ITEMS}})
    b = pd.DataFrame({**base, **{k: [1, 1] for k in ITEMS}})
    b.loc[1, "relevance"] = 2
    d = disagreements(a, b)
    assert len(d) == 1 and d.iloc[0]["item"] == "relevance"
    d["agreed"] = [2]
    final = merge_final(a, d)
    assert final.loc[1, "relevance"] == 2 and final.loc[0, "relevance"] == 1


def test_load_ratings_joins_sheet(tmp_path):
    from src.eval.judge import ITEMS, load_ratings
    sheet = pd.DataFrame({"eval_id": ["e1"], "blind_system": ["S1"], "query": ["q"], "expected_action": ["answer"],
                          "action": ["answer"], "answer": ["a"], **{k: [None] for k in ITEMS}, "notes": [""]})
    sheet.to_csv(tmp_path / "sheet.csv", index=False)
    pd.DataFrame({"eval_id": ["e1"], "blind_system": ["S1"], **{k: [1] for k in ITEMS}, "notes": ["x"]}).to_csv(tmp_path / "r.csv", index=False)
    r = load_ratings(tmp_path / "r.csv", tmp_path / "sheet.csv")
    assert r.loc[0, "query"] == "q" and r.loc[0, "relevance"] == 1
