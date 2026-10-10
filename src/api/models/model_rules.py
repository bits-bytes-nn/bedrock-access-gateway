"""Per-model request rules for the Bedrock Converse API.

Bedrock models differ in which OpenAI request features they accept, and newer
Claude releases keep tightening the rules (no sampling parameters, adaptive
thinking only, no assistant prefill, no forced tool choice). The rules live here,
in one table, so supporting a new model family means editing this module and
its tests rather than hunting for substring checks across the request code.

Claude rules are keyed by family and version, and a version newer than every
row inherits the newest rules: a future Opus release gets the Opus 5.5 behaviour,
which is closer to what new models accept than the oldest defaults would be.

All checks take a foundation model id; resolve inference profiles first.
"""

import re
from dataclasses import dataclass
from enum import Enum


class Sampling(Enum):
    """Which of temperature and topP a model accepts."""

    BOTH = "both"
    # Either one, but not both together; temperature is kept.
    ONE = "one"
    # Neither ("`temperature` is deprecated for this model").
    NONE = "none"


class Thinking(Enum):
    """How reasoning_effort is expressed for a model."""

    NONE = "none"
    # reasoning_config with budget_tokens (Claude 3.7 to Opus 4.6). Extended thinking
    # rejects a temperature other than 1, topP, and a forced tool choice.
    BUDGET = "budget"
    # thinking.type adaptive with output_config.effort ("thinking.type.enabled" is rejected).
    ADAPTIVE = "adaptive"
    # reasoning_config as a low/medium/high string (DeepSeek V3).
    EFFORT_STRING = "effort_string"


@dataclass(frozen=True)
class ModelRules:
    sampling: Sampling = Sampling.BOTH
    thinking: Thinking = Thinking.NONE
    # Whether a conversation may end with an assistant message.
    prefill: bool = True
    # Whether toolChoice may be sent at all.
    tool_choice: bool = True
    # Whether toolChoice may force a tool ("any" or a named tool).
    forced_tool_choice: bool = True
    prompt_caching: bool = False


# Current naming: anthropic.claude-opus-4-5-20251101-v1:0, anthropic.claude-sonnet-5-5,
# anthropic.claude-sonnet-4-20250514-v1:0 (4.0). The minor version is one or two digits
# followed by a separator, so a date suffix is not mistaken for it.
_CLAUDE_ID = re.compile(r"anthropic\.claude-([a-z]+)-(\d+)(?:-(\d{1,2}))?(?=[-:]|$)")
# Earlier naming, version first: anthropic.claude-3-5-sonnet-20240620-v1:0, anthropic.claude-3-haiku-20240307-v1:0.
_CLAUDE_LEGACY_ID = re.compile(r"anthropic\.claude-(\d+)(?:-(\d))?-([a-z]+)(?=[-:]|$)")


def _claude_version(model_id: str) -> tuple[str, tuple[int, int]] | None:
    match = _CLAUDE_ID.search(model_id)
    if match:
        family, major, minor = match.groups()
        return family, (int(major), int(minor or 0))
    match = _CLAUDE_LEGACY_ID.search(model_id)
    if match:
        major, minor, family = match.groups()
        return family, (int(major), int(minor or 0))
    return None


def _claude_rules(family: str, version: tuple[int, int]) -> ModelRules:
    # Opus 4.7 and 4.8 adopted the Claude 5 request rules early.
    latest_generation = version >= (5, 0) or (family == "opus" and version >= (4, 7))

    if latest_generation:
        sampling, thinking = Sampling.NONE, Thinking.ADAPTIVE
    elif version >= (4, 5):
        sampling, thinking = Sampling.ONE, Thinking.BUDGET
    else:
        # Extended thinking starts with Claude 3.7 Sonnet.
        sampling, thinking = Sampling.BOTH, Thinking.BUDGET if version >= (3, 7) else Thinking.NONE

    # Fable and Mythos reject forced tool choice in every version; Opus and Sonnet from 5.5.
    forced_tool_choice = not (family in {"fable", "mythos"} or (family in {"opus", "sonnet"} and version >= (5, 5)))

    return ModelRules(
        sampling=sampling,
        thinking=thinking,
        prefill=version < (4, 6),
        forced_tool_choice=forced_tool_choice,
        # Prompt caching starts with Claude 3.7 Sonnet and 3.5 Haiku.
        prompt_caching=version >= (3, 7) or (family == "haiku" and version == (3, 5)),
    )


def rules_for(model_id: str) -> ModelRules:
    """Return the request rules for a foundation model id."""
    model_id = model_id.lower()

    if "anthropic.claude" in model_id:
        parsed = _claude_version(model_id)
        if parsed:
            return _claude_rules(*parsed)
        # Unrecognised Claude ids (instant, v2) predate extended thinking.
        return ModelRules()

    if "amazon.nova" in model_id:
        return ModelRules(prompt_caching=True)
    if "deepseek.v3" in model_id or "deepseek.deepseek-v3" in model_id:
        return ModelRules(thinking=Thinking.EFFORT_STRING)
    if "meta.llama3-1-" in model_id:
        # Llama 3.1 rejects the toolChoice field.
        return ModelRules(tool_choice=False)
    return ModelRules()
