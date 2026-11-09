"""TEST ONLY: create a small fake dataset with the exact Bitext schema, so the pipeline can be
smoke-tested without internet. Never use its numbers in the report.

    python scripts/make_standin_bitext.py  -> data/raw/_standin_bitext.parquet
    BITEXT_LOCAL=data/raw/_standin_bitext.parquet python -m pytest tests -q
"""
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

INTENTS = {
    "ACCOUNT": {"check_recent_transactions": "see my recent transactions", "close_account": "close my account",
                "create_account": "open a new account"},
    "ATM": {"dispute_ATM_withdrawal": "dispute an ATM withdrawal", "recover_swallowed_card": "get back a card the ATM swallowed"},
    "CARD": {"activate_card": "activate my card", "activate_card_international_usage": "use my card abroad",
             "block_card": "block my card", "cancel_card": "cancel my credit card",
             "check_card_annual_fee": "check the annual fee of my card",
             "check_current_balance_on_card": "check the balance on my card"},
    "CONTACT": {"customer_service": "contact customer service", "human_agent": "speak to a human agent"},
    "FEES": {"check_fees": "check your fees"},
    "FIND": {"find_ATM": "find an ATM", "find_branch": "find a branch"},
    "LOAN": {"apply_for_loan": "apply for a loan", "apply_for_mortgage": "apply for a mortgage",
             "cancel_loan": "cancel my loan", "cancel_mortgage": "cancel my mortgage",
             "check_loan_payments": "check my loan payments", "check_mortgage_payments": "check my mortgage payments"},
    "PASSWORD": {"get_password": "recover my password", "set_up_password": "set up a password"},
    "TRANSFER": {"cancel_transfer": "cancel a transfer", "make_transfer": "make a transfer"},
}
FRAMES = [("I want to {t}", "B"), ("how can I {t}?", "I"), ("could you please help me {t}", "P"),
          ("can u help me {t} pls", "Q"), ("{t}", "K"), ("I need to {t} and also check my balance", "C"),
          ("I don't know how to {t}", "N"), ("help me {t} asap", "BE"), ("I'm so annoyed, I need to {t} now!", "W"),
          ("hw do i {t}", "Z"), ("where do I go to {t}", "I"), ("I'd like to {t} today", "BL")]


def typo(s, rng):
    i = rng.randrange(1, max(2, len(s) - 1))
    return s[:i] + s[i + 1:]


def main(n_per_intent: int = 40, seed: int = 0):
    rng = random.Random(seed)
    rows = []
    for cat, intents in INTENTS.items():
        for intent, task in intents.items():
            for j in range(n_per_intent):
                frame, flags = FRAMES[j % len(FRAMES)]
                t = typo(task, rng) if "Z" in flags else task
                ins = frame.format(t=t) + ("" if j < len(FRAMES) else f" #{j}")
                resp = (f"I'm here to help you {task}. 1. Log in to {{{{Banking App}}}}. 2. Go to the relevant section. "
                        f"3. Follow the steps. If you need help, call {{{{Customer Support Phone Number}}}}.")
                rows.append({"flags": flags, "instruction": ins, "category": cat, "intent": intent, "response": resp})
    out = C.RAW / "_standin_bitext.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {len(rows)} stand-in rows -> {out}")


if __name__ == "__main__":
    main()
