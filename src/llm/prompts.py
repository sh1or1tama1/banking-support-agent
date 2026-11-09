"""Versioned prompt files.

Each file in prompts/ looks like:

    version: naive_rag@v1
    ### SYSTEM
    ...{policy}...
    ### USER
    ...{query}... {context}

`{policy}` is replaced by prompts/policy_v1.md so every system shares identical rules
(a fair comparison). Change a prompt -> copy to *_v2.md, bump the version, log it in
prompts/CHANGELOG.md. Run outputs record the prompt version they used.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .. import config as C

_VAR = re.compile(r"(?<!\{)\{([a-z_]+)\}(?!\})")


@dataclass
class Prompt:
    version: str
    system: str
    user: str

    def render(self, **kw) -> tuple[str, str]:
        """Replace only {name} for names we pass in. Other braces (JSON examples,
        {{Bitext Placeholders}}) are left untouched, so prompt files need no escaping."""
        def sub(t: str) -> str:
            return _VAR.sub(lambda m: str(kw[m.group(1)]) if m.group(1) in kw else m.group(0), t)
        return sub(self.system), sub(self.user)


def load_prompt(name: str, policy: str = "policy_v1") -> Prompt:
    raw = (C.PROMPTS / f"{name}.md").read_text()
    head, rest = raw.split("### SYSTEM", 1)
    system, user = rest.split("### USER", 1)
    version = head.strip().splitlines()[0].split(":", 1)[1].strip()
    policy_text = (C.PROMPTS / f"{policy}.md").read_text().strip()
    system = system.strip().replace("{policy}", policy_text)
    return Prompt(version=f"{version}+{policy}", system=system, user=user.strip())
