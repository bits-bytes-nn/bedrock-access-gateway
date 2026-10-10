"""Tests for the per-model request and response adjustments in the Converse translation."""

import pytest
from fastapi import HTTPException

from api.models import bedrock as bedrock_module
from api.models.bedrock import BedrockModel
from api.schema import ChatRequest

NEW_CLAUDE = [
    "global.anthropic.claude-opus-5-5",
    "global.anthropic.claude-sonnet-5-5",
    "global.anthropic.claude-haiku-5-5",
    "global.anthropic.claude-opus-4-8",
    "us.anthropic.claude-opus-4-7",
    "anthropic.claude-sonnet-5",
]
OLDER_CLAUDE = [
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "global.anthropic.claude-opus-4-5-20251101-v1:0",
]
PREFILL = [
    {"role": "user", "content": "Count to 5"},
    {"role": "assistant", "content": "1, 2,"},
]


@pytest.fixture
def gateway():
    return BedrockModel()


def build(gateway, **kwargs):
    kwargs.setdefault("messages", [{"role": "user", "content": "hi"}])
    return gateway._parse_request(ChatRequest(**kwargs))


@pytest.mark.parametrize("model_id", NEW_CLAUDE)
def test_sampling_params_dropped_for_models_that_reject_them(gateway, model_id):
    args = build(gateway, model=model_id, max_tokens=100, temperature=0.5, top_p=0.9)
    assert args["inferenceConfig"] == {"maxTokens": 100}


@pytest.mark.parametrize("model_id", OLDER_CLAUDE)
def test_temperature_kept_for_older_models(gateway, model_id):
    args = build(gateway, model=model_id, max_tokens=100, temperature=0.5, top_p=0.9)
    assert args["inferenceConfig"] == {"maxTokens": 100, "temperature": 0.5}


def test_sampling_params_dropped_for_application_profile(gateway, monkeypatch):
    arn = "arn:aws:bedrock:ap-northeast-2:123456789012:application-inference-profile/abc"
    monkeypatch.setitem(bedrock_module.profile_metadata, arn, {"underlying_model_id": "anthropic.claude-opus-5-5"})
    args = build(gateway, model=arn, max_tokens=100, temperature=0.5)
    assert "temperature" not in args["inferenceConfig"]


@pytest.mark.parametrize("model_id", NEW_CLAUDE)
def test_reasoning_effort_maps_to_adaptive_thinking(gateway, model_id):
    args = build(gateway, model=model_id, reasoning_effort="low")
    assert args["additionalModelRequestFields"] == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "low"},
    }


def test_reasoning_effort_uses_budget_for_older_claude(gateway):
    args = build(gateway, model=OLDER_CLAUDE[0], max_tokens=4000, reasoning_effort="medium")
    assert args["additionalModelRequestFields"] == {"reasoning_config": {"type": "enabled", "budget_tokens": 2400}}


def test_reasoning_budget_raised_to_bedrock_minimum(gateway):
    args = build(gateway, model=OLDER_CLAUDE[0], max_tokens=2048, reasoning_effort="low")
    assert args["additionalModelRequestFields"]["reasoning_config"]["budget_tokens"] == 1024


def test_reasoning_rejected_when_max_tokens_cannot_fit_budget(gateway):
    with pytest.raises(HTTPException) as exc:
        build(gateway, model=OLDER_CLAUDE[0], max_tokens=1000, reasoning_effort="low")
    assert exc.value.status_code == 400


@pytest.mark.parametrize("model_id", NEW_CLAUDE + ["global.anthropic.claude-sonnet-4-6"])
def test_continuation_prompt_added_when_prefill_unsupported(gateway, model_id):
    messages = build(gateway, model=model_id, max_tokens=50, messages=PREFILL)["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]


@pytest.mark.parametrize("model_id", OLDER_CLAUDE)
def test_prefill_kept_for_models_that_support_it(gateway, model_id):
    messages = build(gateway, model=model_id, max_tokens=50, messages=PREFILL)["messages"]
    assert messages[-1]["role"] == "assistant"


def test_redacted_reasoning_does_not_break_response(gateway):
    content = [{"reasoningContent": {"redactedContent": b"\x00\x01"}}, {"text": "ok"}]
    response = gateway._create_response(
        model="global.openai.gpt-6.1-sol", message_id="id", content=content, finish_reason="end_turn"
    )
    assert response.choices[0].message.content == "ok"
    assert response.usage.completion_tokens_details is None


def test_reasoning_text_still_wrapped_in_think_tags(gateway):
    content = [{"reasoningContent": {"reasoningText": {"text": "plan"}}}, {"text": "ok"}]
    response = gateway._create_response(
        model="global.moonshotai.kimi-k3", message_id="id", content=content, finish_reason="end_turn"
    )
    assert response.choices[0].message.content == "<think>plan</think>ok"


def test_redacted_reasoning_stream_chunk_is_skipped(gateway):
    chunk = {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"redactedContent": b"x"}}}}
    assert gateway._create_response_stream("global.openai.gpt-6.1-sol", "id", chunk) is None
