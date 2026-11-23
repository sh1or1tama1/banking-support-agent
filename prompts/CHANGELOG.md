# Prompt changelog

Every prompt change gets a new file (`*_v2.md`) and a line here. Never edit a prompt that already
produced reported results; the run files record which version they used.

| Date | File | Version | Change | Why | Author |
|---|---|---|---|---|---|
| 2026-11-08 | policy_v1.md | policy_v1 | Initial shared policy (rules + JSON output) | Same rules for every Task-2 system | Team (AI-assisted, Claude) |
| 2026-11-08 | direct_llm_v1.md | direct_llm@v1 | Initial | Reference system (no retrieval) | Team (AI-assisted, Claude) |
| 2026-11-08 | naive_rag_v1.md | naive_rag@v1 | Initial | Baseline / champion | Team (AI-assisted, Claude) |
| 2026-11-08 | agentic_{analyze,grade,rewrite,generate,validate}_v1.md | agentic_*@v1 | Initial v0 agent prompts | Challenger scaffold; B iterates to v2 on dev runs | Team (AI-assisted, Claude) |
| 2026-11-08 | judge_v1.md | judge@v1 | Rubric identical to the human sheet | LLM-as-a-judge validated against human scores | Team (AI-assisted, Claude) |
| 2026-11-23 | agentic_analyze_v2.md | agentic_analyze@v2 | `needs_clarification` only when no task is named, with examples; vague banking messages count as banking | v1 flagged 29/48 dev queries as vague, which sent clear requests to clarify and skipped rewrite/escalate | Ziming (AI-assisted, Claude) |
| 2026-11-23 | agentic_grade_v2.md | agentic_grade@v2 | Relevant only if the snippet resolves the same situation (ATM dispute ≠ card-payment dispute, etc.) | v1 accepted look-alike snippets for unsupported requests | Ziming (AI-assisted, Claude) |
| 2026-11-23 | agentic_generate_v2.md | agentic_generate@v2 | Explicit action order (clarify if vague → answer if covered → escalate); never ask for secrets in chat | v1 asked clarifying questions on clear, covered requests and on unsupported ones | Ziming (AI-assisted, Claude) |
| 2026-11-23 | agentic_validate_v2.md | agentic_validate@v2 | Checklist with quoted evidence; verdict derived from the checks; knows it speaks for ACME Bank | v1 failed good answers for naming ACME Bank, menu names and placeholder names, then escalated them | Ziming (AI-assisted, Claude) |

**v2 tuning record** (dev set only, `data/processed/dev_queries.parquet`, 48 queries, never the eval set; tool = TF-IDF, threshold 0.6). Three iterations before freezing on 2026-11-23; no reported result uses the intermediate versions.

| Run (outputs/runs/dev/) | Behaviour accuracy | Escalation F1 |
|---|---|---|
| naive_rag | 0.708 | 0.667 |
| agentic_v1 | 0.688 | 0.690 |
| agentic_v2a (first draft) | 0.729 | 0.765 |
| agentic_v2b (+ examples, action order) | 0.792 | 0.737 |
| agentic_v2 (frozen: checklist validator) | 0.875 | 0.813 |

`agentic_rewrite` stays at v1. `policy_v1` is unchanged, so all Task 2 systems still share the same rules. **Agent prompts frozen at v2.**
