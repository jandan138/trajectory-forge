from __future__ import annotations

import json
import os
import re
from typing import Callable, Protocol, Sequence
import urllib.request

from trajectory_forge.core.contracts import (
    ContractValidationError,
    EmbodimentCard,
    ScenarioIntakeBrief,
)
from trajectory_forge.intake.genmanip_intake import render_agent_brief_markdown
from trajectory_forge.solver.diagnostics import (
    AttemptFeedbackPacket,
    render_repair_prompt_markdown,
)


class ControllerSynthesizer(Protocol):
    """Protocol for any human/agent/LLM synthesizer generating `def solve(ctx):` scripts."""

    def propose_initial(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
    ) -> str: ...

    def repair(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        previous_code: str,
        feedback: AttemptFeedbackPacket,
    ) -> str: ...


class StaticControllerSynthesizer:
    """Evaluates a fixed user-provided controller script across privileged + gate stages."""

    def __init__(self, controller_code: str) -> None:
        self.controller_code = controller_code

    def propose_initial(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
    ) -> str:
        _ = brief, embodiment
        return self.controller_code

    def repair(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        previous_code: str,
        feedback: AttemptFeedbackPacket,
    ) -> str:
        _ = brief, embodiment, previous_code, feedback
        return self.controller_code


class ScriptedRepairSynthesizer:
    """Deterministically supplies a sequence of candidate scripts across repair attempts."""

    def __init__(
        self,
        scripts: Sequence[str],
        on_feedback: Callable[[AttemptFeedbackPacket], None] | None = None,
    ) -> None:
        if not scripts:
            raise ContractValidationError("ScriptedRepairSynthesizer requires at least one script")
        self._scripts = list(scripts)
        self._idx = 0
        self._on_feedback = on_feedback

    def propose_initial(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
    ) -> str:
        _ = brief, embodiment
        self._idx = 0
        return self._scripts[0]

    def repair(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        previous_code: str,
        feedback: AttemptFeedbackPacket,
    ) -> str:
        _ = brief, embodiment, previous_code
        if self._on_feedback is not None:
            self._on_feedback(feedback)
        self._idx = min(self._idx + 1, len(self._scripts) - 1)
        return self._scripts[self._idx]


class CallbackControllerSynthesizer:
    """Adapts a pair of user callbacks into a ControllerSynthesizer."""

    def __init__(
        self,
        *,
        propose_fn: Callable[[ScenarioIntakeBrief, EmbodimentCard], str],
        repair_fn: Callable[
            [ScenarioIntakeBrief, EmbodimentCard, str, AttemptFeedbackPacket], str
        ],
    ) -> None:
        self._propose_fn = propose_fn
        self._repair_fn = repair_fn

    def propose_initial(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
    ) -> str:
        return self._propose_fn(brief, embodiment)

    def repair(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        previous_code: str,
        feedback: AttemptFeedbackPacket,
    ) -> str:
        return self._repair_fn(brief, embodiment, previous_code, feedback)


_PYTHON_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_source(response_text: str) -> str:
    """Extract Python source code from a fenced markdown block or raw text."""
    match = _PYTHON_BLOCK_RE.search(response_text)
    if match:
        return match.group(1).strip() + "\n"
    return response_text.strip() + "\n"


class OpenAICompatibleSynthesizer:
    """Stdlib OpenAI-compatible chat completions synthesizer for LLM controller generation."""

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.api_base = (api_base or os.environ.get("TF_LLM_API_BASE", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("TF_LLM_API_KEY", "")
        self.model = model or os.environ.get("TF_LLM_MODEL", "gpt-4o")
        self.timeout_s = timeout_s
        if not self.api_base or not self.api_key:
            raise ContractValidationError(
                "OpenAICompatibleSynthesizer requires api_base and api_key "
                "(or TF_LLM_API_BASE / TF_LLM_API_KEY)"
            )

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = str(body["choices"][0]["message"]["content"])
        return extract_python_source(content)

    def propose_initial(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
    ) -> str:
        system = (
            "You are an expert robotics manipulation programmer writing a privileged "
            "Python controller `def solve(ctx):` for `trajectory_forge`. Return only one "
            "```python ... ``` code block defining `def solve(ctx):`."
        )
        user = render_agent_brief_markdown(brief, embodiment)
        return self._chat(system, user)

    def repair(
        self,
        brief: ScenarioIntakeBrief,
        embodiment: EmbodimentCard,
        previous_code: str,
        feedback: AttemptFeedbackPacket,
    ) -> str:
        system = (
            "You are repairing a `trajectory_forge` `def solve(ctx):` controller based on "
            "live Isaac Sim diagnostics. Fix the blocking stage without using "
            "`restore_snapshot` in continuous execution. Return only one ```python ... ``` block."
        )
        user = (
            render_repair_prompt_markdown(brief, embodiment, feedback)
            + "\n### Previous Controller\n\n```python\n"
            + previous_code.rstrip()
            + "\n```\n"
        )
        return self._chat(system, user)

