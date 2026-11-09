version: agentic_rewrite@v1
### SYSTEM
The previous knowledge-base search did not find useful snippets. Write a better search query for the same customer request: use standard banking vocabulary, fix typos, and keep only the core task.

Return JSON with keys query, reason.

### USER
Customer message:
"""{query}"""

Previous search query: {previous}
