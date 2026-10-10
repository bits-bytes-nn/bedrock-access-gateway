"""The model rules table, checked against behaviour observed on Bedrock.

Each row records what the model accepted or rejected when called directly through
Converse. When a new model family arrives, add its ids here first.
"""

import pytest

from api.models.model_rules import ModelRules, Sampling, Thinking, rules_for

NONE, ONE, BOTH = Sampling.NONE, Sampling.ONE, Sampling.BOTH
ADAPTIVE, BUDGET, NO_THINKING = Thinking.ADAPTIVE, Thinking.BUDGET, Thinking.NONE


@pytest.mark.parametrize(
    "model_id,sampling,thinking,prefill,forced_tool_choice,prompt_caching",
    [
        ("anthropic.claude-opus-5-5", NONE, ADAPTIVE, False, False, True),
        ("anthropic.claude-sonnet-5-5", NONE, ADAPTIVE, False, False, True),
        ("anthropic.claude-haiku-5-5", NONE, ADAPTIVE, False, True, True),
        ("anthropic.claude-opus-5", NONE, ADAPTIVE, False, True, True),
        ("anthropic.claude-sonnet-5", NONE, ADAPTIVE, False, True, True),
        ("anthropic.claude-fable-5-1", NONE, ADAPTIVE, False, False, True),
        ("anthropic.claude-opus-4-8", NONE, ADAPTIVE, False, True, True),
        ("anthropic.claude-opus-4-7", NONE, ADAPTIVE, False, True, True),
        ("anthropic.claude-opus-4-6-v1", ONE, BUDGET, False, True, True),
        ("anthropic.claude-sonnet-4-6", ONE, BUDGET, False, True, True),
        ("anthropic.claude-opus-4-5-20251101-v1:0", ONE, BUDGET, True, True, True),
        ("anthropic.claude-sonnet-4-5-20250929-v1:0", ONE, BUDGET, True, True, True),
        ("anthropic.claude-haiku-4-5-20251001-v1:0", ONE, BUDGET, True, True, True),
        ("anthropic.claude-sonnet-4-20250514-v1:0", BOTH, BUDGET, True, True, True),
        ("anthropic.claude-3-7-sonnet-20250219-v1:0", BOTH, BUDGET, True, True, True),
        ("anthropic.claude-3-5-haiku-20241022-v1:0", BOTH, NO_THINKING, True, True, True),
        # Rejected "unsupported model or your request did not allow prompt caching".
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", BOTH, NO_THINKING, True, True, False),
        ("anthropic.claude-3-5-sonnet-20240620-v1:0", BOTH, NO_THINKING, True, True, False),
        ("anthropic.claude-3-haiku-20240307-v1:0", BOTH, NO_THINKING, True, True, False),
    ],
)
def test_claude_rules_match_bedrock(model_id, sampling, thinking, prefill, forced_tool_choice, prompt_caching):
    rules = rules_for(model_id)
    assert (rules.sampling, rules.thinking, rules.prefill, rules.forced_tool_choice, rules.prompt_caching) == (
        sampling,
        thinking,
        prefill,
        forced_tool_choice,
        prompt_caching,
    )


@pytest.mark.parametrize("model_id", ["anthropic.claude-opus-6", "anthropic.claude-sonnet-6-1-v1:0"])
def test_future_claude_versions_inherit_the_newest_rules(model_id):
    assert rules_for(model_id) == rules_for("anthropic.claude-opus-5-5")


def test_date_suffix_is_not_read_as_a_minor_version():
    # sonnet-4-20250514 is Sonnet 4.0, not 4.20.
    assert rules_for("anthropic.claude-sonnet-4-20250514-v1:0").sampling is BOTH


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("amazon.nova-pro-v1:0", ModelRules(prompt_caching=True)),
        ("deepseek.v3-v1:0", ModelRules(thinking=Thinking.EFFORT_STRING)),
        ("meta.llama3-1-70b-instruct-v1:0", ModelRules(tool_choice=False)),
        ("openai.gpt-oss-120b-1:0", ModelRules()),
        ("anthropic.claude-instant-v1", ModelRules()),
    ],
)
def test_other_providers(model_id, expected):
    assert rules_for(model_id) == expected
