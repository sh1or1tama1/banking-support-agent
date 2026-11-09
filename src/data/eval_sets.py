"""Build the frozen Task-2 evaluation table (one row = one customer query).

Columns
  eval_id, eval_set, query, gold_intent, expected_action, reference_response, flags, source_id

eval_set              source                                    expected_action
--------------------  ----------------------------------------  ---------------
bitext_core           Bitext TEST split, k per intent             answer
bitext_hard           Bitext TEST split, by hard flag (C,Z,Q,..)  answer
b77_real_inscope      BANKING77 test, mapped intents (high/med)   answer
b77_unsupported       BANKING77 test, intents Bitext cannot cover escalate
out_of_domain         hand-written (data/mappings)                escalate
ambiguous             hand-written (data/mappings)                clarify

Task-1 (classification) does NOT use this table; it uses the full Bitext test split plus
the full mapped BANKING77 test set (see classify/).
"""
from __future__ import annotations

import pandas as pd

from .. import config as C
from ..utils import parse_flags, sample_per_group

HARD_FLAGS = ["C", "Z", "Q", "W", "N", "E", "K"]


def load_b77_mapping(strict: bool = False) -> pd.DataFrame:
    m = pd.read_csv(C.B77_MAPPING_FILE)
    if strict:
        m = m[m["confidence"] == "high"]
    else:
        m = m[m["confidence"].isin(["high", "medium"])]
    return m


def banking77_mapped(b77: pd.DataFrame, strict: bool = False, split: str = "test") -> pd.DataFrame:
    """BANKING77 rows whose intent maps to a Bitext intent (real-customer external test)."""
    m = load_b77_mapping(strict)
    m = m[m["bitext_intent"] != "UNSUPPORTED"]
    d = b77[b77["b77_split"] == split].merge(m, on="b77_intent", how="inner")
    return d.rename(columns={"bitext_intent": "gold_intent"})


def banking77_unsupported(b77: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    m = load_b77_mapping(strict=False)
    m = m[m["bitext_intent"] == "UNSUPPORTED"]
    return b77[b77["b77_split"] == split].merge(m, on="b77_intent", how="inner")


def build_eval_sets(
    splits: pd.DataFrame,
    b77: pd.DataFrame,
    per_intent_core: int = 4,
    per_flag_hard: int = 10,
    max_b77_per_intent: int = 8,
    max_b77_inscope: int = 100,
    per_unsupported_intent: int = 2,
    seed: int = C.SEED,
) -> pd.DataFrame:
    test = splits[splits["split"] == "test"].copy()
    test["flagset"] = test["flags"].map(parse_flags)
    rows = []

    core = sample_per_group(test, "intent", per_intent_core, seed)
    for r in core.itertuples():
        rows.append(("bitext_core", r.instruction, r.intent, "answer", r.response, r.flags, r.row_id))

    used = set(core["row_id"])
    for flag in HARD_FLAGS:
        pool = test[test["flagset"].map(lambda s: flag in s) & ~test["row_id"].isin(used)]
        pick = pool.sample(min(per_flag_hard, len(pool)), random_state=seed)
        used |= set(pick["row_id"])
        for r in pick.itertuples():
            rows.append(("bitext_hard", r.instruction, r.intent, "answer", r.response, r.flags, r.row_id))

    inscope = sample_per_group(banking77_mapped(b77), "b77_intent", max_b77_per_intent, seed)
    inscope = inscope.sample(min(max_b77_inscope, len(inscope)), random_state=seed)
    for r in inscope.itertuples():
        rows.append(("b77_real_inscope", r.text, r.gold_intent, "answer", None, None, r.b77_id))

    unsup = sample_per_group(banking77_unsupported(b77), "b77_intent", per_unsupported_intent, seed)
    for r in unsup.itertuples():
        rows.append(("b77_unsupported", r.text, None, "escalate", None, None, r.b77_id))

    for i, r in enumerate(pd.read_csv(C.OOD_FILE).itertuples()):
        rows.append(("out_of_domain", r.query, None, "escalate", None, r.ood_type, f"ood_{i:03d}"))
    for i, r in enumerate(pd.read_csv(C.AMBIGUOUS_FILE).itertuples()):
        rows.append(("ambiguous", r.query, None, "clarify", None, r.plausible_intents, f"amb_{i:03d}"))

    ev = pd.DataFrame(
        rows, columns=["eval_set", "query", "gold_intent", "expected_action", "reference_response", "flags", "source_id"]
    )
    ev.insert(0, "eval_id", [f"ev_{i:04d}" for i in range(len(ev))])
    return ev


def human_eval_subset(ev: pd.DataFrame, n: int = 100, seed: int = C.SEED) -> pd.DataFrame:
    """Stratified subset for the two-reviewer human rubric (proportional, at least 5 per set)."""
    share = ev["eval_set"].value_counts(normalize=True)
    k = (share * n).round().clip(lower=5).astype(int)
    parts = [g.sample(min(int(k[name]), len(g)), random_state=seed) for name, g in ev.groupby("eval_set")]
    return pd.concat(parts).reset_index(drop=True)
