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
