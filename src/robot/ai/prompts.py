"""Prompt templates for the assistant."""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_SYSTEM_PROMPT: str = (
    "You are DeskBot, a small, friendly desktop companion. "
    "Reply concisely, with warmth and a hint of playfulness. "
    "Keep answers under 50 words unless asked otherwise."
)


def system_prompt(name: str = "DeskBot", personality_summary: str = "") -> str:
    """Return the default system prompt, optionally customised with personality traits."""
    traits = personality_summary or "curious, friendly, playful"
    return (
        f"You are {name}, a small desktop companion robot. "
        f"Your personality is {traits}. "
        "Reply concisely, with warmth and a hint of playfulness. "
        "Keep answers under 50 words unless asked otherwise."
    )


@dataclass(slots=True)
class PromptBuilder:
    """Composable prompt builder for more advanced LLM calls."""

    template: str
    variables: dict[str, str] = field(default_factory=dict)

    def bind(self, **kwargs: str) -> PromptBuilder:
        return PromptBuilder(template=self.template, variables={**self.variables, **kwargs})

    def render(self) -> str:
        return self.template.format(**self.variables)


__all__ = ["DEFAULT_SYSTEM_PROMPT", "PromptBuilder", "system_prompt"]
